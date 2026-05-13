import argparse
import json
import os
import random
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pyodbc
import torch
import torch.nn as nn
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset

try:
    from tqdm.auto import tqdm
except ModuleNotFoundError:
    def tqdm(iterable, **kwargs):
        return iterable

from lms_mon.extract_spike import extract_spike
from lstmae import LSTMAutoencoder


BASE_DIR = Path(__file__).resolve().parent

DEFAULT_COLS = (
    [f"F03_{i:02d}" for i in range(1, 7)]
    + [f"F05_{i:02d}" for i in range(1, 4)]
    + [f"F10_{i:02d}" for i in range(1, 26)]
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train LMS spike autoencoder models for multiple columns."
    )
    parser.add_argument(
        "--cols",
        nargs="+",
        default=DEFAULT_COLS,
        help="Columns to train. Default: F03_01~F03_06, F05_01~F05_03, F10_01~F10_25.",
    )
    parser.add_argument("--env", default="dev", help="Environment file suffix.")
    parser.add_argument(
        "--env-file",
        default=None,
        help="Explicit .env path. Default: .env.{env}, then lms_mon/.env.{env}.",
    )
    parser.add_argument(
        "--table",
        default="kamtec_test.dbo.lmsCurrent_Copy",
        help="Source table name.",
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seq-len", type=int, default=20)
    parser.add_argument("--pre-offset", type=int, default=5)
    parser.add_argument("--slope-factor", type=float, default=3.0)
    parser.add_argument("--min-jump", type=float, default=0.3)
    parser.add_argument("--dedup-gap", type=int, default=10)
    parser.add_argument("--threshold-percentile", type=float, default=99.5)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument(
        "--save-root",
        default=str(BASE_DIR / "model"),
        help="Root directory where model/{COL} artifacts are saved.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Training device.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip a column if model.pth, scaler.pkl, and threshold.npy already exist.",
    )
    return parser.parse_args()


def load_env(args):
    if args.env_file:
        env_path = Path(args.env_file)
    else:
        env_path = BASE_DIR / f".env.{args.env}"
        if not env_path.exists():
            env_path = BASE_DIR / "lms_mon" / f".env.{args.env}"

    if not env_path.exists():
        raise FileNotFoundError(f"Env file not found: {env_path}")

    load_dotenv(env_path, override=True)
    return env_path


def make_conn_str():
    driver = os.getenv("DRIVER", "ODBC Driver 18 for SQL Server").strip()
    server = os.getenv("SERVER", "").strip()
    port = os.getenv("PORT", "").strip()
    database = os.getenv("DATABASE", "").strip()
    username = os.getenv("USERNAME", "").strip()
    password = os.getenv("PASSWORD", "").strip()

    if not all([server, database, username, password]):
        raise ValueError("SERVER, DATABASE, USERNAME, and PASSWORD are required.")

    server_part = f"{server},{port}" if port else server
    return (
        f"DRIVER={driver};"
        f"SERVER={server_part};"
        f"DATABASE={database};"
        f"UID={username};"
        f"PWD={password};"
        "Encrypt=no;"
        "TrustServerCertificate=yes;"
    )


def resolve_device(device_arg):
    if device_arg == "cpu":
        return torch.device("cpu")
    if device_arg == "cuda":
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_column_data(conn_str, table, col):
    query = f"""
        SELECT {col}
        FROM {table} WITH (NOLOCK)
        WHERE {col} > 0
        ORDER BY id ASC
        """

    with pyodbc.connect(conn_str, timeout=10) as conn:
        return pd.read_sql(query, conn)


def get_errors(model, loader, device):
    model.eval()
    errors = []

    with torch.no_grad():
        for batch_x, _ in loader:
            batch_x = batch_x.to(device)
            output = model(batch_x)
            loss = torch.mean((batch_x - output) ** 2, dim=(1, 2))
            errors.extend(loss.cpu().numpy())

    return np.array(errors)


def train_one_col(conn_str, col, args, device):
    save_dir = Path(args.save_root) / col
    model_path = save_dir / "model.pth"
    scaler_path = save_dir / "scaler.pkl"
    threshold_path = save_dir / "threshold.npy"

    if args.skip_existing and all(
        path.exists() for path in [model_path, scaler_path, threshold_path]
    ):
        print(f"[{col}] skip existing artifacts")
        return {"col": col, "status": "skipped"}

    print("=" * 80)
    print(f"[{col}] load data")

    df = load_column_data(conn_str, args.table, col)
    if df.empty:
        print(f"[{col}] no source data")
        return {"col": col, "status": "no_data"}

    print(f"[{col}] rows: {len(df)}")

    scaler = MinMaxScaler()
    df["scaled"] = scaler.fit_transform(df[[col]])

    sequences, indices = extract_spike(
        df["scaled"].values.reshape(-1),
        seq_len=args.seq_len,
        pre_offset=args.pre_offset,
        slope_factor=args.slope_factor,
        min_jump=args.min_jump,
        dedup_gap=args.dedup_gap,
    )

    print(f"[{col}] spikes: {len(sequences)}")

    if len(sequences) < 2:
        print(f"[{col}] not enough spikes")
        return {"col": col, "status": "not_enough_spikes", "spikes": len(sequences)}

    train_sequences, test_sequences = train_test_split(
        sequences,
        test_size=args.test_size,
        random_state=args.random_state,
        shuffle=True,
    )

    x_train = torch.tensor(train_sequences, dtype=torch.float32).unsqueeze(-1)
    x_test = torch.tensor(test_sequences, dtype=torch.float32).unsqueeze(-1)

    train_loader = DataLoader(
        TensorDataset(x_train, x_train),
        batch_size=args.batch_size,
        shuffle=True,
    )
    test_loader = DataLoader(
        TensorDataset(x_test, x_test),
        batch_size=args.batch_size,
        shuffle=False,
    )

    model = LSTMAutoencoder(args.seq_len, 1).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    print(f"[{col}] train start: epochs={args.epochs}, device={device}")

    for epoch in range(args.epochs):
        model.train()
        epoch_losses = []

        for batch_x, _ in tqdm(train_loader, desc=f"{col} epoch {epoch + 1}"):
            batch_x = batch_x.to(device)

            optimizer.zero_grad()
            output = model(batch_x)
            loss = criterion(output, batch_x)
            loss.backward()
            optimizer.step()

            epoch_losses.append(loss.item())

        print(f"[{col}] epoch {epoch + 1}/{args.epochs} loss={np.mean(epoch_losses):.8f}")

    train_errors = get_errors(model, train_loader, device)
    test_errors = get_errors(model, test_loader, device)
    threshold = float(np.percentile(train_errors, args.threshold_percentile))
    test_anomalies = int(np.sum(test_errors > threshold))

    save_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), model_path)
    joblib.dump(scaler, scaler_path)
    np.save(threshold_path, threshold)

    metadata = {
        "col": col,
        "status": "trained",
        "table": args.table,
        "rows": int(len(df)),
        "spikes": int(len(sequences)),
        "train_sequences": int(len(train_sequences)),
        "test_sequences": int(len(test_sequences)),
        "seq_len": args.seq_len,
        "pre_offset": args.pre_offset,
        "slope_factor": args.slope_factor,
        "min_jump": args.min_jump,
        "dedup_gap": args.dedup_gap,
        "embedding_dim": 64,
        "output_activation": "sigmoid",
        "loss_type": "mse",
        "threshold_percentile": args.threshold_percentile,
        "threshold": threshold,
        "train_error_min": float(train_errors.min()),
        "train_error_max": float(train_errors.max()),
        "train_error_mean": float(train_errors.mean()),
        "test_error_min": float(test_errors.min()),
        "test_error_max": float(test_errors.max()),
        "test_error_mean": float(test_errors.mean()),
        "test_anomalies": test_anomalies,
        "scaler_data_min": float(scaler.data_min_[0]),
        "scaler_data_max": float(scaler.data_max_[0]),
        "scaler_data_range": float(scaler.data_range_[0]),
        "model_path": str(model_path),
        "scaler_path": str(scaler_path),
        "threshold_path": str(threshold_path),
        "trained_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    with (save_dir / "metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print(f"[{col}] saved: {save_dir}")
    print(f"[{col}] threshold={threshold:.8f}, test_anomalies={test_anomalies}")

    return metadata


def main():
    args = parse_args()
    set_seed(args.random_state)
    env_path = load_env(args)
    conn_str = make_conn_str()
    device = resolve_device(args.device)

    print("batch train start")
    print(f"env: {env_path}")
    print(f"table: {args.table}")
    print(f"device: {device}")
    print(f"cols: {', '.join(args.cols)}")

    results = []
    for col in args.cols:
        try:
            results.append(train_one_col(conn_str, col, args, device))
        except Exception as exc:
            print(f"[{col}] ERROR: {exc}")
            results.append({"col": col, "status": "error", "error": str(exc)})

    summary_path = Path(args.save_root) / "batch_train_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print("=" * 80)
    print(f"batch train done: {summary_path}")


if __name__ == "__main__":
    main()
