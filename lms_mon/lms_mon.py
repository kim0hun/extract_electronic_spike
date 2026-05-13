import time
import joblib
import numpy as np
import pandas as pd
import pyodbc
import os
import sys
from dotenv import load_dotenv

from extract_spike import extract_spike

# ==============================
# 설정
# ==============================

os.environ["PYTHONIOENCODING"] = "utf-8"
sys.stdout.reconfigure(encoding="utf-8")

env = os.getenv("NODE_ENV", "dev")

load_dotenv(f".env.{env}", override=True)

COL = os.getenv("COL")
SEQ_LEN = 20
PRE_OFFSET = 5
INTERVAL = 1

SCALER_PATH = f"../model/{COL}/scaler.pkl"

DRIVER = os.getenv("DRIVER")
SERVER = os.getenv("SERVER")
PORT = os.getenv("PORT")
DATABASE = os.getenv("DATABASE")
USERNAME = os.getenv("USERNAME")
PASSWORD = os.getenv("PASSWORD")

conn_str = (
    f"DRIVER={DRIVER};"
    f"SERVER={SERVER},{PORT};"
    f"DATABASE={DATABASE};"
    f"UID={USERNAME};"
    f"PWD={PASSWORD};"
    "TrustServerCertificate=yes;"
)

conn = None


# ==============================
# DB
# ==============================
def get_conn():
    global conn
    if conn is None:
        print("DB 연결 시도")
        conn = pyodbc.connect(conn_str, timeout=5)
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
        raise


# ==============================
# 타입 변환 (안정)
# ==============================
def to_py(v):
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return float(v)
    if isinstance(v, pd.Timestamp):
        return v.to_pydatetime()
    return v


# ==============================
# 스케일러 로드
# ==============================
scaler = joblib.load(SCALER_PATH)

print("스케일러 로드 완료")

# ==============================
# 메인 루프
# ==============================
last_id = None

print(f"{COL} 모니터링 시작")

while True:
    try:
        print("============================================================")
        # =========================
        # last_id 조회
        # =========================
        if last_id is None:
            print("마지막 lms_id 조회")
            rows, _ = exec_query(
                """
                    SELECT TOP 1 lms_id
                    FROM spike_event WITH (NOLOCK)
                    WHERE name = ?
                    ORDER BY id DESC
                """,
                (COL,),
                fetch=True,
            )
            
            last_id = int(rows[0][0]) if rows else None
        print(f"마지막 lms_id: {last_id or '없음'}")
        
        # =========================
        # 데이터 조회
        # =========================
        print("데이터 조회 시작")

        if last_id is None:
            qry = f"""
                SELECT id, s001, {COL}
                FROM lmsCurrent WITH (NOLOCK)
                -- WHERE s001 >= CAST(GETDATE() AS DATE)
                -- WHERE s001 >= '2026-04-16' AND s001 < '2026-04-17'
                WHERE {COL} > 0
                ORDER BY id ASC
            """
            params = None
        else:
            qry = f"""
                SELECT id, s001, {COL}
                FROM lmsCurrent WITH (NOLOCK)
                WHERE id > ? AND {COL} > 0
                ORDER BY id ASC
            """
            params = (last_id,)

        rows, cols = exec_query(qry, params, fetch=True)

        print(f"데이터 조회 완료: {len(rows)}건")

        if len(rows) < SEQ_LEN:
            print("데이터 부족 → 대기")
            time.sleep(INTERVAL)
            continue

        last_id = int(rows[-SEQ_LEN][0])

        df = pd.DataFrame.from_records(rows, columns=cols)

        # =========================
        # 스케일링
        # =========================
        print("스케일링 시작")
        df["scaled"] = scaler.transform(df[[COL]]).reshape(-1)
        print("스케일링 완료")

        # =========================
        # 스파이크 추출
        # =========================
        print("스파이크 추출 시작")

        sequences, indices = extract_spike(
            df["scaled"].values.reshape(-1),
            seq_len=SEQ_LEN,
            pre_offset=PRE_OFFSET,
            slope_factor=3.0,
            min_jump=0.3,
            dedup_gap=10,
        )

        if not len(indices):
            print("스파이크 없음")
            time.sleep(INTERVAL)
            continue

        spike_results = []

        for idx in indices:
            start = idx - PRE_OFFSET
            end = start + SEQ_LEN

            if start < 0 or end > len(df):
                continue

            raw = df.iloc[start:end]
            first = raw.iloc[0]

            spike_results.append(
                {
                    "lms_id": int(first["id"]),
                    "s001": first["s001"].to_pydatetime(),
                    "sequence": [float(x) for x in raw[COL].values],
                }
            )

        print(f"스파이크 추출 완료: {len(spike_results)}")

        # =========================
        # DB 처리 (단일 트랜잭션)
        # =========================
        conn = get_conn()
        cursor = conn.cursor()

        print(f"spike_event + spike_data INSERT 시작: {len(spike_results)}건")

        event_rows = [
            (COL, to_py(x["lms_id"]), to_py(x["s001"])) for x in spike_results
        ]

        BATCH_SIZE = 500
        event_ids = []

        # 1️⃣ spike_event insert
        for i in range(0, len(event_rows), BATCH_SIZE):
            batch = event_rows[i : i + BATCH_SIZE]

            values_clause = ",".join(["(?, ?, ?)"] * len(batch))

            query = f"""
                DECLARE @tmp TABLE (id INT);

                INSERT INTO spike_event (name, lms_id, s001)
                OUTPUT INSERTED.id INTO @tmp
                VALUES {values_clause};

                SELECT id FROM @tmp;
            """

            params = [v for row in batch for v in row]

            cursor.execute(query, params)

            # 👉 결과셋 이동 (중요)
            while cursor.description is None:
                if not cursor.nextset():
                    raise Exception("결과셋 없음")

            batch_ids = [int(r[0]) for r in cursor.fetchall()]

            if len(batch_ids) != len(batch):
                raise Exception(f"ID 개수 불일치: {len(batch_ids)} != {len(batch)}")

            event_ids.extend(batch_ids)

        # 2️⃣ spike_data insert
        data_rows = [
            (eid, i, v)
            for eid, item in zip(event_ids, spike_results)
            for i, v in enumerate(item["sequence"])
        ]

        print(f"spike_data INSERT: {len(data_rows)}건")

        if data_rows:
            cursor.fast_executemany = True
            cursor.executemany(
                """
                INSERT INTO spike_data (event_id, seq, actual_value)
                VALUES (?, ?, ?)
            """,
                data_rows,
            )

        conn.commit()

        print(f"완료: event {len(event_ids)}, data {len(data_rows)}")

    except Exception as e:
        try:
            conn.rollback()
            print("ROLLBACK 수행")
        except:
            pass

        print(f"ERROR: {e}")

    finally:
        time.sleep(INTERVAL)
