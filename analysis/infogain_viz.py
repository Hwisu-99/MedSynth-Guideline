"""
Entropy & Information Gain 시각화 (발표용)

PANEL 1 : 엔트로피 개념 — 불순도 곡선
PANEL 2 : 계산 예시    — 식전혈당 → 당뇨 단계별 분해
PANEL 3 : 질환별 Info Gain 순위 (가로 막대)

실행:
    python analysis/infogain_viz.py
"""

import os
import time
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from datetime import datetime

warnings.filterwarnings("ignore")

_KOREAN_FONTS = ['Malgun Gothic', 'NanumGothic', 'AppleGothic', 'Noto Sans KR']
_registered   = {f.name for f in fm.fontManager.ttflist}
_korean_font  = next((f for f in _KOREAN_FONTS if f in _registered), None)
if _korean_font:
    plt.rcParams['font.family'] = _korean_font
plt.rcParams['axes.unicode_minus'] = False

RESULT_DIR = f"result/analysis/{datetime.now().strftime('%Y%m%d_%H%M%S')}_infogain"
os.makedirs(RESULT_DIR, exist_ok=True)
print(f"결과 저장 경로: {RESULT_DIR}\n")

REAL_PATH     = "data/preprocessed/health_data_2024_preprocessed.csv"
TARGETS       = ['당뇨', '고혈압', '간기능']
TARGET_COLORS = {'당뇨': '#E74C3C', '고혈압': '#3498DB', '간기능': '#2ECC71'}


# ── 헬퍼 ─────────────────────────────────────────────────────────────────────
def entropy(series: pd.Series) -> float:
    p = series.value_counts(normalize=True).values
    p = p[p > 0]
    return float(-np.sum(p * np.log2(p)))

def info_gain(df, feature, target):
    H_D = entropy(df[target])
    n   = len(df)
    H_after = sum(
        (len(g) / n) * entropy(g[target])
        for _, g in df.groupby(feature, observed=True)
    )
    return H_D - H_after


# ── 데이터 로드 & 이산화 ──────────────────────────────────────────────────────
print("[1] 데이터 로드 및 이산화")
df_raw = pd.read_csv(REAL_PATH, encoding="utf-8-sig")
if '이름' in df_raw.columns:
    df_raw = df_raw.drop(columns=['이름'])

d = pd.DataFrame(index=df_raw.index)
d['성별']       = df_raw['성별코드'].map({1: '남', 2: '여'})
d['연령대']     = pd.cut(df_raw['연령대코드(5세단위)'],
                        bins=[0, 4, 7, 10, 13, 99],
                        labels=['미성년', '청년', '중년', '장년', '노년'])
d['BMI']        = pd.cut(df_raw['BMI'], bins=[0, 18.5, 22.9, 24.9, 999],
                         labels=['저체중', '정상', '과체중', '비만'])
d['수축기혈압'] = pd.cut(df_raw['수축기혈압'], bins=[0, 120, 140, 999],
                        labels=['정상', '주의', '위험'])
d['이완기혈압'] = pd.cut(df_raw['이완기혈압'], bins=[0, 80, 90, 999],
                        labels=['정상', '주의', '위험'])
d['혈색소']     = pd.cut(df_raw['혈색소'], bins=[0, 12, 16, 999],
                        labels=['낮음', '정상', '높음'])
d['식전혈당']   = pd.cut(df_raw['식전혈당(공복혈당)'], bins=[0, 100, 126, 999],
                        labels=['정상', '주의', '위험'])
d['AST']        = pd.cut(df_raw['혈청지오티(AST)'], bins=[0, 40, 60, 999],
                         labels=['정상', '주의', '위험'])
d['ALT']        = pd.cut(df_raw['혈청지피티(ALT)'], bins=[0, 40, 60, 999],
                         labels=['정상', '주의', '위험'])
d['감마지티피'] = pd.cut(df_raw['감마지티피'], bins=[0, 35, 63, 999],
                        labels=['정상', '주의', '위험'])
d['흡연상태']   = df_raw['흡연상태'].map({1: '비흡연', 2: '과거흡연', 3: '현재흡연'})
d['음주여부']   = df_raw['음주여부'].map({0: '비음주', 1: '음주'})
for col in TARGETS:
    d[col] = df_raw[col].map({0: '정상', 1: '주의', 2: '위험'})

FEATURE_COLS = [c for c in d.columns if c not in TARGETS]
print(f"  완료: {d.shape}")


# ── Info Gain 전체 계산 ───────────────────────────────────────────────────────
print("[2] Info Gain 계산")
ig_rows = []
for target in TARGETS:
    H_D = entropy(d[target])
    for feat in FEATURE_COLS:
        ig_rows.append({'타겟': target, '피처': feat,
                        'H(D)': round(H_D, 4),
                        'Info Gain': round(info_gain(d, feat, target), 4)})
df_ig = pd.DataFrame(ig_rows)
print("  완료")


# ══════════════════════════════════════════════════════════════════════════════
# 시각화
# ══════════════════════════════════════════════════════════════════════════════
print("\n[3] 시각화")

fig = plt.figure(figsize=(24, 20))
fig.suptitle("Entropy & Information Gain 기반 중요 변수 선택",
             fontsize=17, fontweight='bold', y=0.98)

gs = gridspec.GridSpec(3, 3, figure=fig,
                       hspace=0.55, wspace=0.38,
                       left=0.07, right=0.97, top=0.93, bottom=0.05)

# ─── PANEL 1 : 엔트로피 개념 곡선 ────────────────────────────────────────────
ax_concept = fig.add_subplot(gs[0, :])

p = np.linspace(0.001, 0.999, 400)

# 이진 엔트로피
H2 = -p * np.log2(p) - (1 - p) * np.log2(1 - p)

# 3클래스 예시 (정상 p1, 주의 p2, 위험 1-p1-p2 with p2=0.15 고정)
p2 = 0.10
p1 = np.linspace(0.001, 0.889, 400)
p3 = 1 - p1 - p2
mask = p3 > 0
H3 = np.full_like(p1, np.nan)
H3[mask] = (
    -p1[mask] * np.log2(p1[mask])
    - p2 * np.log2(p2)
    - p3[mask] * np.log2(p3[mask])
)

ax_concept.plot(p, H2, color='#3498DB', linewidth=2.5, label='이진 분류 (정상/위험)')
ax_concept.plot(p1[mask], H3[mask], color='#E74C3C', linewidth=2.5,
                linestyle='--', label=f'3클래스 (정상/주의/위험, 주의={p2:.0%} 고정)')

ax_concept.axvline(0.5, color='#3498DB', linestyle=':', linewidth=1.2, alpha=0.6)
ax_concept.annotate('최대 불순도\n(완전 혼합)',
                    xy=(0.5, 1.0), xytext=(0.58, 0.88),
                    arrowprops=dict(arrowstyle='->', color='gray'),
                    fontsize=10, color='#3498DB')

ax_concept.annotate('순수 집합\n(엔트로피=0)',
                    xy=(0.02, 0.14), xytext=(0.12, 0.35),
                    arrowprops=dict(arrowstyle='->', color='gray'),
                    fontsize=10, color='#555')

ax_concept.set_xlabel("클래스 비율 (p)", fontsize=12)
ax_concept.set_ylabel("엔트로피 H(D)", fontsize=12)
ax_concept.set_title(
    "PANEL 1 — 엔트로피 개념\n"
    "H(D) = -sum( pi * log2(pi) )    |    불순도가 높을수록(=클래스가 섞일수록) 값이 커짐",
    fontsize=12, loc='left')
ax_concept.legend(fontsize=10)
ax_concept.set_ylim(0, 1.7)
ax_concept.grid(axis='y', alpha=0.3)

# 실제 H(D) 값 표시
for target in TARGETS:
    hd = df_ig[df_ig['타겟'] == target]['H(D)'].iloc[0]
    ax_concept.axhline(hd, color=TARGET_COLORS[target], linewidth=1.2,
                       linestyle='-.', alpha=0.85,
                       label=f'실제 H({target}) = {hd:.3f}')
ax_concept.legend(fontsize=9, ncol=2)


# ─── PANEL 2 : 계산 예시 — 식전혈당 → 당뇨 ──────────────────────────────────
ax_ex = fig.add_subplot(gs[1, :2])

feat_ex, tgt_ex = '식전혈당', '당뇨'
H_D_ex = entropy(d[tgt_ex])
groups = d.groupby(feat_ex, observed=True)[tgt_ex]
n_total = len(d)

bar_w   = 0.22
x_ticks = []
x_labels = []
x = 0

palette = {'정상': '#2ECC71', '주의': '#F39C12', '위험': '#E74C3C'}
class_order = ['정상', '주의', '위험']

# 전체(D) 바
sub_full = d[tgt_ex].value_counts(normalize=True).reindex(class_order, fill_value=0)
bottom = 0
for cls in class_order:
    ax_ex.bar(x, sub_full[cls], bar_w * 1.3,
              bottom=bottom, color=palette[cls], edgecolor='white', linewidth=0.5)
    if sub_full[cls] > 0.04:
        ax_ex.text(x, bottom + sub_full[cls] / 2, f"{sub_full[cls]:.2f}",
                   ha='center', va='center', fontsize=8, color='white', fontweight='bold')
    bottom += sub_full[cls]
ax_ex.text(x, 1.04, f"H={H_D_ex:.3f}", ha='center', fontsize=9, fontweight='bold')
x_ticks.append(x); x_labels.append(f"전체 D\n(n={n_total:,})")
x += 0.55

# 구분선
ax_ex.axvline(x - 0.15, color='gray', linestyle='--', linewidth=1, alpha=0.5)
ax_ex.text(x - 0.15, 1.12, "분기 후", ha='center', fontsize=9, color='gray')

# 각 그룹 바
grp_order = ['정상', '주의', '위험']
for grp_name in grp_order:
    if grp_name not in groups.groups:
        continue
    grp = groups.get_group(grp_name)
    n_grp = len(grp)
    H_grp = entropy(grp)
    dist  = grp.value_counts(normalize=True).reindex(class_order, fill_value=0)
    weight = n_grp / n_total

    bottom = 0
    for cls in class_order:
        ax_ex.bar(x, dist[cls], bar_w,
                  bottom=bottom, color=palette[cls],
                  edgecolor='white', linewidth=0.5, alpha=0.88)
        if dist[cls] > 0.06:
            ax_ex.text(x, bottom + dist[cls] / 2, f"{dist[cls]:.2f}",
                       ha='center', va='center', fontsize=7.5, color='white', fontweight='bold')
        bottom += dist[cls]

    ax_ex.text(x, 1.04,
               f"H={H_grp:.3f}\n×{weight:.2f}",
               ha='center', fontsize=8.5, fontweight='bold',
               color=palette.get(grp_name, 'black'))
    x_ticks.append(x)
    x_labels.append(f"혈당{grp_name}\n(n={n_grp:,})")
    x += 0.42

IG_ex = info_gain(d, feat_ex, tgt_ex)
ax_ex.set_xticks(x_ticks)
ax_ex.set_xticklabels(x_labels, fontsize=10)
ax_ex.set_ylabel("클래스 비율", fontsize=11)
ax_ex.set_ylim(0, 1.25)
ax_ex.set_title(
    f"PANEL 2 — 계산 예시: [{feat_ex}] → [{tgt_ex}]\n"
    f"IG = H(D) - sum( |Dv|/|D| * H(Dv) )  =  {H_D_ex:.3f} - (가중합)  =  {IG_ex:.4f}",
    fontsize=11, loc='left')

legend_patches = [mpatches.Patch(color=palette[c], label=f'당뇨 {c}') for c in class_order]
ax_ex.legend(handles=legend_patches, loc='upper right', fontsize=9)
ax_ex.grid(axis='y', alpha=0.25)


# ─── PANEL 2-R : IG 공식 텍스트 박스 ─────────────────────────────────────────
ax_formula = fig.add_subplot(gs[1, 2])
ax_formula.axis('off')

formula_lines = [
    ("엔트로피 (불순도 측정)", 13, 'black', True),
    ("", 6, 'black', False),
    ("H(D) = -sum( pi * log2(pi) )", 13, '#2C3E50', True),
    ("  pi : 클래스 i의 비율", 10, '#555', False),
    ("  값이 클수록 = 더 혼잡(예측 어려움)", 10, '#555', False),
    ("", 6, 'black', False),
    ("정보 이득 (분기 후 불순도 감소량)", 13, 'black', True),
    ("", 6, 'black', False),
    ("IG(D,A) = H(D) - sum( |Dv|/|D| * H(Dv) )", 12, '#2980B9', True),
    ("  A : 분기 속성", 10, '#555', False),
    ("  Dv : 속성값 v인 부분집합", 10, '#555', False),
    ("  값이 클수록 = A가 유용한 변수", 10, '#555', False),
    ("", 8, 'black', False),
    (f"예시 결과", 12, 'black', True),
    (f"  IG(식전혈당 → 당뇨) = {IG_ex:.4f}", 12, '#E74C3C', True),
    (f"  → 전체 피처 중 상위권", 11, '#E74C3C', False),
]

y_pos = 0.97
for text, size, color, bold in formula_lines:
    if text == "":
        y_pos -= size / 100
        continue
    ax_formula.text(0.05, y_pos, text,
                    transform=ax_formula.transAxes,
                    fontsize=size, color=color,
                    fontweight='bold' if bold else 'normal',
                    va='top')
    y_pos -= (size + 3) / 100

ax_formula.set_facecolor('#F8F9FA')
for spine in ax_formula.spines.values():
    spine.set_visible(True)
    spine.set_color('#DDD')
ax_formula.set_title("공식 요약", fontsize=11, loc='left')


# ─── PANEL 3 : 질환별 Info Gain 순위 ─────────────────────────────────────────
for col_idx, target in enumerate(TARGETS):
    ax = fig.add_subplot(gs[2, col_idx])
    sub = df_ig[df_ig['타겟'] == target].sort_values('Info Gain', ascending=True)

    bars = ax.barh(sub['피처'], sub['Info Gain'],
                   color=TARGET_COLORS[target], alpha=0.85, edgecolor='white')

    # 상위 3개 강조
    top3 = sub.nlargest(3, 'Info Gain').index
    for bar, idx in zip(bars, sub.index):
        if idx in top3:
            bar.set_edgecolor('#2C3E50')
            bar.set_linewidth(1.5)

    for bar, val in zip(bars, sub['Info Gain']):
        ax.text(val + 0.0005, bar.get_y() + bar.get_height() / 2,
                f"{val:.4f}", va='center', fontsize=8.5)

    H_D = sub['H(D)'].iloc[0]
    ax.set_xlabel("Information Gain", fontsize=10)
    ax.set_title(
        f"PANEL 3-{col_idx+1} — [{target}] 변수 순위\n"
        f"H(D) = {H_D:.4f}  |  테두리=상위 3개",
        fontsize=10, loc='left')
    ax.axvline(0, color='gray', linewidth=0.6)
    ax.grid(axis='x', alpha=0.25)
    xlim_max = sub['Info Gain'].max() * 1.18
    ax.set_xlim(0, xlim_max)


out = f"{RESULT_DIR}/infogain_viz.png"
plt.savefig(out, dpi=150, bbox_inches='tight')
plt.show()
print(f"\n저장: {out}")
print(f"완료: {RESULT_DIR}/")
