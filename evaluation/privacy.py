import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import NearestNeighbors

plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

RESULT_DIR = f"result/evaluation/{datetime.now().strftime('%Y%m%d_%H%M%S')}"
os.makedirs(RESULT_DIR, exist_ok=True)
print(f"결과 저장 경로: {RESULT_DIR}")

# ── 설정 ────────────────────────────────────────────────────────
REAL_PATH = "data/preprocessed/health_data_2024_preprocessed.csv"
TOOLS = {
    'Synthpop': "data/synthetic/synthpop/health_data_synthetic_synthpop.csv",
    'SDV':      "data/synthetic/sdv/health_data_synthetic_sdv.csv",
    # 'ARX':   "data/synthetic/arx/health_data_synthetic_arx.csv",    # 교수님 제공 예정
    # 'MASQ':  "data/synthetic/masq/health_data_synthetic_masq.csv",   # 교수님 제공 예정
}
TOOL_COLORS = {'Synthpop': '#E74C3C', 'SDV': '#3498DB',
               'ARX': '#F39C12', 'MASQ': '#9B59B6'}
SAMPLE_N    = 2000   # DCR/NNDR 계산용 샘플 크기 (메모리/속도 조정)
DCR_THRESHOLD = 0.01  # 고위험 레코드 판단 임계값

# ── 1. 데이터 로드 ─────────────────────────────────────────────
df_real = pd.read_csv(REAL_PATH, encoding="utf-8-sig")
if '이름' in df_real.columns:
    df_real = df_real.drop(columns=['이름'])
print(f"원본 데이터: {df_real.shape}")

synth_dfs = {}
for tool, path in TOOLS.items():
    if os.path.exists(path):
        df_s = pd.read_csv(path, encoding="utf-8-sig")
        if '이름' in df_s.columns:
            df_s = df_s.drop(columns=['이름'])
        synth_dfs[tool] = df_s
        print(f"로드 완료: {tool} ({df_s.shape[0]:,}행)")
    else:
        print(f"파일 없음 (건너뜀): {path}")

if not synth_dfs:
    print("평가할 합성 데이터가 없습니다. 먼저 synthesis 스크립트를 실행하세요.")
    exit()

# 수치형 공통 컬럼 추출
num_cols = set(df_real.select_dtypes(include='number').columns)
for df_s in synth_dfs.values():
    num_cols &= set(df_s.select_dtypes(include='number').columns)
NUMERIC_COLS = [c for c in df_real.select_dtypes(include='number').columns
                if c in num_cols]
print(f"\n분석 컬럼 ({len(NUMERIC_COLS)}개): {NUMERIC_COLS}")

# ── 2. MinMax 정규화 & NearestNeighbors 준비 ──────────────────
X_real = df_real[NUMERIC_COLS].dropna().values
scaler = MinMaxScaler()
X_real_scaled = scaler.fit_transform(X_real)

rng = np.random.default_rng(42)
n_real_sample = min(SAMPLE_N, len(X_real_scaled))
idx = rng.choice(len(X_real_scaled), n_real_sample, replace=False)
X_real_sample = X_real_scaled[idx]

# k=2: 1번째 거리 → DCR, 2번째 거리 → NNDR 분모
nn = NearestNeighbors(n_neighbors=2, algorithm='ball_tree', n_jobs=-1)
nn.fit(X_real_sample)

# ── 3. DCR / NNDR 계산 ──────────────────────────────────────────
print(f"\n[DCR / NNDR 계산] 샘플 크기: {n_real_sample}")
print("  DCR  : 합성 레코드 → 가장 가까운 실제 레코드까지의 거리 (클수록 안전)")
print("  NNDR : 1st / 2nd 최근접 거리 비율 (1에 가까울수록 안전)")

dcr_dict  = {}
nndr_dict = {}
summary_rows = []

for tool, df_s in synth_dfs.items():
    X_syn = df_s[NUMERIC_COLS].dropna().values
    X_syn_scaled = scaler.transform(X_syn)
    n_syn_sample = min(SAMPLE_N, len(X_syn_scaled))
    idx_s = rng.choice(len(X_syn_scaled), n_syn_sample, replace=False)
    X_syn_sample = X_syn_scaled[idx_s]

    distances, _ = nn.kneighbors(X_syn_sample)
    dcr  = distances[:, 0]
    nndr = distances[:, 0] / (distances[:, 1] + 1e-10)

    dcr_dict[tool]  = dcr
    nndr_dict[tool] = nndr

    risk_rate = (dcr < DCR_THRESHOLD).mean() * 100
    print(f"\n  [{tool}]")
    print(f"    DCR  — 평균: {dcr.mean():.4f}  중앙값: {np.median(dcr):.4f}  최솟값: {dcr.min():.6f}")
    print(f"    NNDR — 평균: {nndr.mean():.4f}  중앙값: {np.median(nndr):.4f}")
    print(f"    고위험 레코드 비율 (DCR < {DCR_THRESHOLD}): {risk_rate:.2f}%")

    summary_rows.append({
        '툴':               tool,
        'DCR_평균':          round(float(dcr.mean()), 4),
        'DCR_중앙값':         round(float(np.median(dcr)), 4),
        'DCR_최솟값':         round(float(dcr.min()), 6),
        'NNDR_평균':         round(float(nndr.mean()), 4),
        'NNDR_중앙값':        round(float(np.median(nndr)), 4),
        f'고위험_비율(DCR<{DCR_THRESHOLD})(%)': round(risk_rate, 2),
    })

df_summary = pd.DataFrame(summary_rows)
df_summary.to_csv(f"{RESULT_DIR}/privacy_metrics.csv",
                  index=False, encoding="utf-8-sig")
print(f"\n[요약]")
print(df_summary.to_string(index=False))

# ── 4. DCR / NNDR 분포 히스토그램 ─────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

ax = axes[0]
for tool, dcr in dcr_dict.items():
    ax.hist(dcr, bins=60, alpha=0.6, label=tool,
            color=TOOL_COLORS.get(tool, 'gray'), density=True)
ax.axvline(DCR_THRESHOLD, color='black', linestyle='--',
           linewidth=1.2, label=f'임계값 ({DCR_THRESHOLD})')
ax.set_xlabel("DCR (Distance to Closest Record)")
ax.set_ylabel("밀도")
ax.set_title("DCR 분포 비교\n(값이 클수록 원본과 달라 프라이버시 안전)")
ax.legend()

ax = axes[1]
for tool, nndr in nndr_dict.items():
    ax.hist(nndr, bins=60, alpha=0.6, label=tool,
            color=TOOL_COLORS.get(tool, 'gray'), density=True)
ax.axvline(0.5, color='black', linestyle='--',
           linewidth=1.2, label='기준값 (0.5)')
ax.set_xlabel("NNDR")
ax.set_ylabel("밀도")
ax.set_title("NNDR 분포 비교\n(1에 가까울수록 프라이버시 안전)")
ax.legend()

fig.suptitle("프라이버시 안전성 평가 — DCR / NNDR", fontsize=13)
plt.tight_layout()
plt.savefig(f"{RESULT_DIR}/privacy_dcr_nndr.png", dpi=150, bbox_inches='tight')
plt.show()
print(f"저장: {RESULT_DIR}/privacy_dcr_nndr.png")

# ── 5. 프라이버시 지표 요약 막대그래프 ─────────────────────────
metrics   = ['DCR_평균', 'DCR_중앙값', 'NNDR_평균', 'NNDR_중앙값']
x_pos     = np.arange(len(metrics))
bar_width = 0.7 / len(df_summary)

fig, ax = plt.subplots(figsize=(10, 5))
for i, row in df_summary.iterrows():
    bars = ax.bar(x_pos + i * bar_width,
                  [row[m] for m in metrics],
                  bar_width,
                  label=row['툴'],
                  color=TOOL_COLORS.get(row['툴'], 'gray'),
                  alpha=0.85)
    for bar, val in zip(bars, [row[m] for m in metrics]):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.005,
                f"{val:.3f}", ha='center', fontsize=8)

ax.set_xticks(x_pos + bar_width * (len(df_summary) - 1) / 2)
ax.set_xticklabels(metrics, fontsize=10)
ax.set_ylabel("값")
ax.set_title("프라이버시 지표 요약 비교\n(DCR↑, NNDR↑ 일수록 안전)")
ax.legend()
plt.tight_layout()
plt.savefig(f"{RESULT_DIR}/privacy_summary.png", dpi=150, bbox_inches='tight')
plt.show()
print(f"저장: {RESULT_DIR}/privacy_summary.png")

# ── 6. 고위험 레코드 비율 막대그래프 ──────────────────────────
risk_col = [c for c in df_summary.columns if '고위험' in c][0]
fig, ax = plt.subplots(figsize=(6, 4))
bars = ax.bar(df_summary['툴'], df_summary[risk_col],
              color=[TOOL_COLORS.get(t, 'gray') for t in df_summary['툴']],
              alpha=0.85, width=0.4)
for bar, val in zip(bars, df_summary[risk_col]):
    ax.text(bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.1,
            f"{val:.2f}%", ha='center', fontsize=11)
ax.set_ylabel("고위험 레코드 비율 (%)")
ax.set_title(f"고위험 레코드 비율 비교\n(DCR < {DCR_THRESHOLD}, 낮을수록 안전)")
plt.tight_layout()
plt.savefig(f"{RESULT_DIR}/privacy_risk_ratio.png", dpi=150, bbox_inches='tight')
plt.show()
print(f"저장: {RESULT_DIR}/privacy_risk_ratio.png")

print("\n[안전성 평가 완료]")
print(f"  {RESULT_DIR}/privacy_metrics.csv")
print(f"  {RESULT_DIR}/privacy_dcr_nndr.png")
print(f"  {RESULT_DIR}/privacy_summary.png")
print(f"  {RESULT_DIR}/privacy_risk_ratio.png")
