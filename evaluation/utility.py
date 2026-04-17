import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score
from sklearn.preprocessing import LabelEncoder

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
TOOL_COLORS  = {'Synthpop': '#E74C3C', 'SDV': '#3498DB',
                'ARX': '#F39C12', 'MASQ': '#9B59B6'}
CATEGORY_COLS = ['성별코드', '연령대코드(5세단위)', '시도코드', '흡연상태', '음주여부']
TARGET_COL    = '음주여부'

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

# 실제 + 모든 합성 데이터에 공통으로 존재하는 수치형 컬럼만 사용
num_cols = set(df_real.select_dtypes(include='number').columns)
for df_s in synth_dfs.values():
    num_cols &= set(df_s.select_dtypes(include='number').columns)
NUMERIC_COLS = [c for c in df_real.select_dtypes(include='number').columns if c in num_cols]
print(f"\n평가 컬럼 ({len(NUMERIC_COLS)}개): {NUMERIC_COLS}")

# ── 2. 기술통계 & KS 검정 ──────────────────────────────────────
print("\n" + "=" * 60)
print("[기술통계 비교 — KS 검정]")
print("=" * 60)

rows = []
for col in NUMERIC_COLS:
    real_vals = df_real[col].dropna()
    for tool, df_s in synth_dfs.items():
        syn_vals = df_s[col].dropna()
        ks_stat, ks_p = stats.ks_2samp(real_vals, syn_vals)
        rows.append({
            '컬럼': col, '툴': tool,
            '실제_평균': round(real_vals.mean(), 3),
            '합성_평균': round(syn_vals.mean(), 3),
            '평균_차이':  round(abs(real_vals.mean() - syn_vals.mean()), 3),
            'KS_통계량':  round(ks_stat, 4),
            'KS_pvalue':  round(ks_p, 4),
        })

df_stats = pd.DataFrame(rows)
df_stats.to_csv(f"{RESULT_DIR}/utility_stats.csv",
                index=False, encoding="utf-8-sig")
print(df_stats.to_string(index=False))

# ── 3. KS 통계량 막대그래프 ────────────────────────────────────
tools_list = list(synth_dfs.keys())
x     = np.arange(len(NUMERIC_COLS))
width = 0.7 / len(tools_list)

fig, ax = plt.subplots(figsize=(14, 5))
for i, tool in enumerate(tools_list):
    ks_vals = [
        df_stats.loc[(df_stats['컬럼'] == col) & (df_stats['툴'] == tool), 'KS_통계량'].values
        for col in NUMERIC_COLS
    ]
    ks_vals = [v[0] if len(v) > 0 else np.nan for v in ks_vals]
    ax.bar(x + i * width, ks_vals, width,
           label=tool, color=TOOL_COLORS.get(tool, 'gray'), alpha=0.85)

ax.set_xticks(x + width * (len(tools_list) - 1) / 2)
ax.set_xticklabels(NUMERIC_COLS, rotation=30, ha='right', fontsize=9)
ax.axhline(0.05, color='black', linestyle='--', linewidth=0.8, label='기준 0.05')
ax.set_ylabel("KS 통계량 (낮을수록 원본과 유사)")
ax.set_title("수치형 컬럼별 KS 통계량 비교 (원본 vs 합성)")
ax.legend()
plt.tight_layout()
plt.savefig(f"{RESULT_DIR}/utility_ks.png", dpi=150, bbox_inches='tight')
plt.show()
print(f"저장: {RESULT_DIR}/utility_ks.png")

# ── 4. 분포 히스토그램 ──────────────────────────────────────────
ncols = 4
nrows = (len(NUMERIC_COLS) + ncols - 1) // ncols
fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4, nrows * 3))
axes = axes.flatten()

for i, col in enumerate(NUMERIC_COLS):
    ax = axes[i]
    ax.hist(df_real[col].dropna(), bins=40, alpha=0.5,
            label='원본', color='#2ECC71', density=True)
    for tool, df_s in synth_dfs.items():
        ax.hist(df_s[col].dropna(), bins=40, alpha=0.5,
                label=tool, color=TOOL_COLORS.get(tool, 'gray'), density=True)
    ax.set_title(col, fontsize=9)
    ax.legend(fontsize=7)

for j in range(i + 1, len(axes)):
    axes[j].set_visible(False)

fig.suptitle("수치형 컬럼 분포 비교 (원본 vs 합성)", fontsize=13)
plt.tight_layout()
plt.savefig(f"{RESULT_DIR}/utility_distribution.png", dpi=150, bbox_inches='tight')
plt.show()
print(f"저장: {RESULT_DIR}/utility_distribution.png")

# ── 5. 상관관계 행렬 비교 ──────────────────────────────────────
n_panels = 1 + len(synth_dfs)
fig, axes = plt.subplots(1, n_panels, figsize=(5.5 * n_panels, 5))
if n_panels == 1:
    axes = [axes]

corr_real = df_real[NUMERIC_COLS].corr()
sns.heatmap(corr_real, annot=True, fmt=".2f", cmap="RdYlBu_r",
            vmin=-1, vmax=1, center=0, square=True,
            ax=axes[0], annot_kws={"size": 7})
axes[0].set_title("원본", fontsize=11)
axes[0].tick_params(axis='x', rotation=30, labelsize=7)
axes[0].tick_params(axis='y', rotation=0,  labelsize=7)

for i, (tool, df_s) in enumerate(synth_dfs.items()):
    corr_syn = df_s[NUMERIC_COLS].corr()
    frob = np.linalg.norm(corr_real.values - corr_syn.values, 'fro')
    sns.heatmap(corr_syn, annot=True, fmt=".2f", cmap="RdYlBu_r",
                vmin=-1, vmax=1, center=0, square=True,
                ax=axes[i + 1], annot_kws={"size": 7})
    axes[i + 1].set_title(f"{tool}\n(Frobenius: {frob:.3f})", fontsize=11)
    axes[i + 1].tick_params(axis='x', rotation=30, labelsize=7)
    axes[i + 1].tick_params(axis='y', rotation=0,  labelsize=7)

fig.suptitle("상관관계 행렬 비교", fontsize=13, y=1.02)
plt.tight_layout()
plt.savefig(f"{RESULT_DIR}/utility_correlation.png", dpi=150, bbox_inches='tight')
plt.show()
print(f"저장: {RESULT_DIR}/utility_correlation.png")

# ── 6. ML 유틸리티 — TSTR ─────────────────────────────────────
print("\n" + "=" * 60)
print("[ML 유틸리티 평가 — TSTR]")
print(f"  Target: {TARGET_COL} (0/1 이진 분류)")
print("=" * 60)

all_feature_cols = ['혈청지오티(AST)', '혈청지피티(ALT)', '감마지티피']

def prepare_X(df, feature_cols):
    df = df[feature_cols].copy()
    le = LabelEncoder()
    for col in df.columns:
        if df[col].dtype == 'object' or str(df[col].dtype) == 'category':
            df[col] = le.fit_transform(df[col].astype(str))
    return df

X_real = prepare_X(df_real, all_feature_cols)
y_real = df_real[TARGET_COL].astype(int)
X_tr, X_te, y_tr, y_te = train_test_split(
    X_real, y_real, test_size=0.2, random_state=42, stratify=y_real)

# TRTR (기준선)
clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
clf.fit(X_tr, y_tr)
pred = clf.predict(X_te)
ml_rows = [{
    '방식':        'TRTR (기준선)',
    '정확도':       round(accuracy_score(y_te, pred), 4),
    'F1 (weighted)': round(f1_score(y_te, pred, average='weighted'), 4),
}]
print(f"TRTR  정확도: {ml_rows[0]['정확도']}  F1: {ml_rows[0]['F1 (weighted)']}")

# TSTR (각 툴)
for tool, df_s in synth_dfs.items():
    feat_cols = [c for c in all_feature_cols if c in df_s.columns]
    X_syn = prepare_X(df_s, feat_cols)
    y_syn = df_s[TARGET_COL].astype(int)
    shared = [c for c in feat_cols if c in X_te.columns]
    clf_s = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    clf_s.fit(X_syn[shared], y_syn)
    pred_s = clf_s.predict(X_te[shared])
    acc = round(accuracy_score(y_te, pred_s), 4)
    f1  = round(f1_score(y_te, pred_s, average='weighted'), 4)
    print(f"TSTR ({tool})  정확도: {acc}  F1: {f1}")
    ml_rows.append({'방식': f'TSTR ({tool})', '정확도': acc, 'F1 (weighted)': f1})

df_ml = pd.DataFrame(ml_rows)
df_ml.to_csv(f"{RESULT_DIR}/utility_ml.csv", index=False, encoding="utf-8-sig")

bar_colors = ['#2ECC71'] + [TOOL_COLORS.get(t, 'gray') for t in synth_dfs]
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
for ax, metric in zip(axes, ['정확도', 'F1 (weighted)']):
    bars = ax.bar(df_ml['방식'], df_ml[metric],
                  color=bar_colors, alpha=0.85, width=0.5)
    ax.set_ylim(0, 1.15)
    for bar, val in zip(bars, df_ml[metric]):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.02,
                f"{val:.3f}", ha='center', fontsize=10)
    ax.set_ylabel(metric)
    ax.set_title(f"ML 유틸리티 — {metric}")
    ax.tick_params(axis='x', rotation=20)

plt.tight_layout()
plt.savefig(f"{RESULT_DIR}/utility_ml.png", dpi=150, bbox_inches='tight')
plt.show()
print(f"저장: {RESULT_DIR}/utility_ml.png")

print("\n[효용성 평가 완료]")
print(f"  {RESULT_DIR}/utility_stats.csv")
print(f"  {RESULT_DIR}/utility_ks.png")
print(f"  {RESULT_DIR}/utility_distribution.png")
print(f"  {RESULT_DIR}/utility_correlation.png")
print(f"  {RESULT_DIR}/utility_ml.csv")
print(f"  {RESULT_DIR}/utility_ml.png")
