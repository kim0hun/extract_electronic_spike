import torch.nn as nn
import numpy as np

# ==============================
# 스파이크 데이터 추출 함수
# ==============================


def extract_spike(
    signal,
    seq_len=20,
    pre_offset=5,
    slope_factor=3.0,  # 🔥 민감도 (높을수록 엄격)
    min_jump=0.3,  # 🔥 절대 상승량 기준
    dedup_gap=10,  # 🔥 중복 제거
):
    signal = np.asarray(signal)
    diff = np.diff(signal)

    # 🔥 1. 통계 기반 threshold
    slope_th = diff.mean() + slope_factor * diff.std()

    # 🔥 2. 급상승 조건 (두 가지 동시에 만족)
    rise_points = (
        np.where((diff > slope_th) & (diff > min_jump))[  # 통계적으로 큼  # 절대값도 큼
            0
        ]
        + 1
    )

    sequences = []
    indices = []

    last_idx = -dedup_gap

    for i in rise_points:

        # 🔥 3. 중복 제거 (같은 이벤트 여러번 잡는거 방지)
        if i - last_idx < dedup_gap:
            continue

        start = i - pre_offset
        end = start + seq_len

        if start < 0 or end > len(signal):
            continue

        seq = signal[start:end]

        # 🔥 4. 진짜 "상승"인지 확인 (앞뒤 평균 비교)
        before = np.mean(signal[max(0, i - 5) : i])
        after = np.mean(signal[i : i + 5])

        if (after - before) < min_jump:
            continue

        sequences.append(seq)
        indices.append(i)

        last_idx = i

    return np.array(sequences), np.array(indices)
