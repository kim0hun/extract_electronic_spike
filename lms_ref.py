import time
import torch
import joblib
import numpy as np
import pyodbc
import os
import sys
from dotenv import load_dotenv

from lstmae import LSTMAutoencoder

import warnings

# pandas 학습, numpy 추론 경고 제거
warnings.filterwarnings(
    "ignore",
    message="X does not have valid feature names",
)

# =========================================================
# 설정
# =========================================================
print("============================================================")

os.environ["PYTHONIOENCODING"] = "utf-8"
sys.stdout.reconfigure(encoding="utf-8")

env = os.getenv("NODE_ENV", "dev")

load_dotenv(f".env.{env}", override=True)

COL = os.getenv("COL")

SEQ_LEN = 20
BATCH_SIZE = 100
INTERVAL = 1

MODEL_DIR = f"../model/{COL}"

gpu_cols = [
    "F03_01",
    "F03_02",
    "F03_03",
    "F03_04",
    "F03_05",
    "F03_06",
]

device = torch.device("cuda" if COL not in gpu_cols else "cpu")

print(f"디바이스: {device}")

SERVER = os.getenv("SERVER")
PORT = os.getenv("PORT")
DATABASE = os.getenv("DATABASE")
USERNAME = os.getenv("USERNAME")
PASSWORD = os.getenv("PASSWORD")

conn_str = (
    "DRIVER=FreeTDS;"
    f"SERVER={SERVER};"
    f"PORT={PORT};"
    f"DATABASE={DATABASE};"
    f"UID={USERNAME};"
    f"PWD={PASSWORD};"
    "TDS_Version=7.4;"
)

conn = None

# anomaly 데이터만 recon 저장 여부
SAVE_ONLY_ANOMALY = True


# =========================================================
# DB 유틸
# =========================================================
def get_conn():
    global conn

    if conn is None:
        print("DB 연결 시도")

        conn = pyodbc.connect(
            conn_str,
            timeout=5,
        )

        print("DB 연결 성공")

    return conn


def exec_query(query, params=None, fetch=False):
    global conn

    try:
        conn = get_conn()

        cursor = conn.cursor()

        if params is not None:
            cursor.execute(query, params)
        else:
            cursor.execute(query)

        if fetch:
            rows = cursor.fetchall()
            cols = [c[0] for c in cursor.description]

            return rows, cols

        conn.commit()

    except Exception as e:
        print(f"DB 오류 → 재연결: {e}")

        try:
            conn.close()
        except:
            pass

        conn = None

        raise e


# =========================================================
# 모델 로드
# =========================================================
model = LSTMAutoencoder(SEQ_LEN, 1).to(device)

state_dict = torch.load(
    f"{MODEL_DIR}/model.pth",
    map_location=device,
    weights_only=True,
)

model.load_state_dict(state_dict)

model.eval()

scaler = joblib.load(f"{MODEL_DIR}/scaler.pkl")

threshold = float(np.load(f"{MODEL_DIR}/threshold.npy"))

print("모델 로드 완료")
print(f"threshold: {threshold:.8f}")

# =========================================================
# 메인 루프
# =========================================================
while True:
    try:
        print("============================================================")

        # =====================================================
        # 1. 이벤트 조회
        # =====================================================
        print("이벤트 조회 시작")

        rows, _ = exec_query(
            f"""
            SELECT TOP ({BATCH_SIZE}) id
            FROM spike_event WITH (NOLOCK)
            WHERE name = ? AND is_checked = 0
            ORDER BY id Desc
            """,
            (COL,),
            fetch=True,
        )

        event_ids = [r[0] for r in rows]

        print(f"이벤트 조회 완료: {len(event_ids)}건")

        if not event_ids:
            time.sleep(INTERVAL)
            continue

        # =====================================================
        # 2. 데이터 로드
        # =====================================================
        print("데이터 로드 시작")

        placeholders = ",".join(["?"] * len(event_ids))

        rows, cols = exec_query(
            f"""
            SELECT
                event_id,
                seq,
                actual_value
            FROM spike_data WITH (NOLOCK)
            WHERE event_id IN ({placeholders})
            ORDER BY event_id, seq
            """,
            params=event_ids,
            fetch=True,
        )

        print(f"데이터 로드 완료: {len(rows)}건")

        # =====================================================
        # 3. 시퀀스 구성
        # =====================================================
        print("시퀀스 구성 시작")

        sequences = []
        valid_ids = []

        current_id = None

        current_seq = []
        current_seq_ids = []

        expected_seq = list(range(SEQ_LEN))

        for row in rows:
            eid, seq, val = row

            if current_id != eid:

                # 이전 event 저장
                if len(current_seq) == SEQ_LEN and current_seq_ids == expected_seq:
                    sequences.append(current_seq)
                    valid_ids.append(current_id)

                current_seq = []
                current_seq_ids = []

                current_id = eid

            current_seq.append(val)
            current_seq_ids.append(seq)

        # 마지막 event 처리
        if len(current_seq) == SEQ_LEN and current_seq_ids == expected_seq:
            sequences.append(current_seq)
            valid_ids.append(current_id)

        if len(sequences) == 0:
            print("유효한 시퀀스 없음")

            time.sleep(INTERVAL)
            continue

        sequences = np.array(
            sequences,
            dtype=np.float32,
        )

        print(f"시퀀스 구성 완료: {len(sequences)}건")

        # =====================================================
        # 4. 전처리
        # =====================================================
        print("전처리 시작")

        x = sequences.reshape(
            -1,
            SEQ_LEN,
            1,
        )

        x = scaler.transform(x.reshape(-1, 1)).reshape(
            -1,
            SEQ_LEN,
            1,
        )

        print("전처리 완료")

        # =====================================================
        # 5. 추론
        # =====================================================
        print("추론 시작")

        x_tensor = torch.tensor(
            x,
            dtype=torch.float32,
        ).to(device)

        with torch.no_grad():
            recon = model(x_tensor)

        # =====================================================
        # reconstruction error
        # 학습 threshold는 MSE 기준으로 계산되어 있어 추론도 동일하게 맞춘다.
        # =====================================================
        errors = (x_tensor - recon) ** 2

        # anomaly score
        losses = torch.mean(
            errors,
            dim=(1, 2),
        )

        # numpy 변환
        losses = losses.cpu().numpy()

        x_np = x_tensor.cpu().numpy()
        recon_np = recon.cpu().numpy()
        errors_np = errors.cpu().numpy()

        anomaly_mask = losses > threshold

        print(
            "loss min/max/mean: "
            f"{float(losses.min()):.8f} "
            f"{float(losses.max()):.8f} "
            f"{float(losses.mean()):.8f}"
        )
        print(
            "anomaly count: "
            f"{int(anomaly_mask.sum())}/{len(losses)}"
        )

        # =====================================================
        # 역정규화
        # =====================================================
        x_denorm = scaler.inverse_transform(x_np.reshape(-1, 1)).reshape(-1, SEQ_LEN)

        # MinMaxScaler 입력 범위 밖의 reconstruction은 원본 단위에서 음수 전류를 만들 수 있다.
        # anomaly score는 위에서 원본 recon 기준으로 계산하고, 저장/표시용 복원값만 clamp한다.
        recon_np_clipped = np.clip(recon_np, 0.0, 1.0)

        recon_denorm = scaler.inverse_transform(
            recon_np_clipped.reshape(-1, 1)
        ).reshape(-1, SEQ_LEN)

        print(f"추론 완료: {len(losses)}건")

        # =====================================================
        # 6. spike_event 저장
        # =====================================================
        conn = get_conn()

        cursor = conn.cursor()

        event_update_params = []

        for eid, loss in zip(valid_ids, losses):

            is_anomaly = 1 if loss > threshold else 0

            event_update_params.append(
                (
                    float(loss),
                    is_anomaly,
                    eid,
                )
            )

        anomaly_ids = [eid for _, is_anomaly, eid in event_update_params if is_anomaly]

        if anomaly_ids:
            print(f"이상 탐지 {len(anomaly_ids)}건 " f"→ {anomaly_ids}")
        else:
            print("이상 없음")

        cursor.executemany(
            """
            UPDATE spike_event
            SET
                anomaly_score = ?,
                is_anomaly = ?,
                is_checked = 1
            WHERE id = ?
            """,
            event_update_params,
        )

        # =====================================================
        # 7. spike_recon 저장
        # =====================================================
        recon_insert_params = []

        for idx, eid in enumerate(valid_ids):

            loss = losses[idx]

            for seq in range(SEQ_LEN):

                recon_val = float(recon_denorm[idx][seq])

                error_val = float(abs(x_denorm[idx][seq] - recon_denorm[idx][seq]))

                recon_insert_params.append(
                    (
                        eid,
                        seq,
                        recon_val,
                        error_val,
                    )
                )

        if recon_insert_params:

            cursor.executemany(
                """
                INSERT INTO spike_recon (
                    event_id,
                    seq,
                    recon_value,
                    error_value
                )
                VALUES (?, ?, ?, ?)
                """,
                recon_insert_params,
            )

            print(f"reconstrunction 저장 완료: " f"{len(recon_insert_params)}건")

        conn.commit()

        print("최종 저장 및 업데이트 완료")

    except Exception as e:
        print(f"ERROR 발생: {e}")

        try:
            conn.rollback()
        except:
            pass

    finally:
        time.sleep(INTERVAL)
