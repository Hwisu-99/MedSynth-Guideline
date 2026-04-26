"""
Information Gain 변수 추출 과정 시각화 (발표용) — 당뇨 / 고혈압 / 간기능

스토리 흐름 (열 = 질환, 행 = STEP):
  STEP 1 : 전체 분포 → H(D) 계산
  STEP 2 : IG 높은 피처로 분기 → 그룹이 순수해짐
  STEP 3 : IG 낮은 피처로 분기 → 그룹이 여전히 혼잡
  STEP 4 : 모든 피처 반복 → 순위화 → 상위 변수 선택

실행:
    python analysis/infogain_process_viz.py
"""

import os
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

RESULT_DIR = f"result/analysis/{datetime.now().strftime('%Y%m%d_%H%M%S')}_ig_process"
os.makedirs(RESULT_DIR, exist_ok=True)
print(f"결과 저장 경로: {RESULT_DIR}\n")

REAL_PATH   = "data/preprocessed/health_data_2024_preprocessed.csv"
PALETTE     = {'정상': '#27AE60', '주의': '#F39C12', '위험': '#E74C3C'}
CLASS_ORDER = ['정상', '주의', '위험']

# 질환별 설정 (good: IG 높은 피처, bad: IG 낮은 피처)
DISEASE_CFG = {
    '당뇨':   {'color': '#C0392B', 'good': '식전혈당',   'bad': '음주여부'},
    '고혈압': {'color': '#2980B9', 'good': '수축기혈압',  'bad': '음주여부'},
    '간기능': {'color': '#27AE60', 'good': '감마지티피',  'bad': '음주여부'},
}
TARGETS = list(DISEASE_CFG.keys())


# ── 헬퍼 ─────────────────────────────────────────────────────────────────────
def entropy(series):
    p = series.value_counts(normalize=True).values
    p = p[p > 0]
    return float(-np.sum(p * np.log2(p)))

def info_gain(df, feat, target):
    H_D = entropy(df[target])
    n   = len(df)
    return H_D - sum((len(g)/n) * entropy(g[target])
                     for _, g in df.groupby(feat, observed=True))

def stacked_bar(ax, x, series, bar_w, min_show=0.05, alpha=1.0):
    dist   = series.value_counts(normalize=True).reindex(CLASS_ORDER, fill_value=0)
    bottom = 0
    for cls in CLASS_ORDER:
        v = dist[cls]
        ax.bar(x, v, bar_w, bottom=bottom, color=PALETTE[cls],
               edgecolor='white', linewidth=0.6, alpha=alpha)
        if v > min_show:
            ax.text(x, bottom + v/2, f"{v:.2f}",
                    ha='center', va='center', fontsize=8,
                    color='white', fontweight='bold')
        bottom += v
    return entropy(series)


# ── 데이터 로드 & 이산화 ──────────────────────────────────────────────────────
print("[1] 데이터 로드 및 이산화")
df_raw = pd.read_csv(REAL_PATH, encoding="utf-8-sig")
if '이름' in df_raw.columns:
    df_raw = df_raw.drop(columns=['이름'])

d = pd.DataFrame(index=df_raw.index)
d['성별']         = df_raw['성별코드'].map({1: '남', 2: '여'})
d['연령대']       = pd.cut(df_raw['연령대코드(5세단위)'],
                          bins=[0,4,7,10,13,99],
                          labels=['미성년','청년','중년','장년','노년'])
d['BMI']          = pd.cut(df_raw['BMI'], bins=[0,18.5,22.9,24.9,999],
                           labels=['저체중','정상','과체중','비만'])
d['수축기혈압']   = pd.cut(df_raw['수축기혈압'], bins=[0,120,140,999],
                          labels=['정상','주의','위험'])
d['이완기혈압']   = pd.cut(df_raw['이완기혈압'], bins=[0,80,90,999],
                          labels=['정상','주의','위험'])
d['혈색소']       = pd.cut(df_raw['혈색소'], bins=[0,12,16,999],
                          labels=['낮음','정상','높음'])
d['식전혈당']     = pd.cut(df_raw['식전혈당(공복혈당)'], bins=[0,100,126,999],
                          labels=['정상','주의','위험'])
d['혈청크레아티닌'] = pd.cut(df_raw['혈청크레아티닌'], bins=[0,0.7,1.2,999],
                            labels=['낮음','정상','높음'])
d['AST']          = pd.cut(df_raw['혈청지오티(AST)'], bins=[0,40,60,999],
                           labels=['정상','주의','위험'])
d['ALT']          = pd.cut(df_raw['혈청지피티(ALT)'], bins=[0,40,60,999],
                           labels=['정상','주의','위험'])
d['감마지티피']   = pd.cut(df_raw['감마지티피'], bins=[0,35,63,999],
                          labels=['정상','주의','위험'])
d['흡연상태']     = df_raw['흡연상태'].map({1:'비흡연',2:'과거흡연',3:'현재흡연'})
d['음주여부']     = df_raw['음주여부'].map({0:'비음주',1:'음주'})
for t in TARGETS:
    d[t] = df_raw[t].map({0:'정상',1:'주의',2:'위험'})

FEATURE_COLS = [c for c in d.columns if c not in TARGETS]
n_total = len(d)

# 질환별 IG 전체 계산
print("[2] Info Gain 계산")
ig_by_target = {}
for target in TARGETS:
    ig_by_target[target] = sorted(
        [(f, info_gain(d, f, target)) for f in FEATURE_COLS],
        key=lambda x: x[1], reverse=True
    )
print("  완료\n")


# ══════════════════════════════════════════════════════════════════════════════
# 질환별 개별 figure
# 레이아웃: 좌(STEP1) | 중앙상(STEP2) / 중앙하(STEP3) | 우(STEP4)
# ══════════════════════════════════════════════════════════════════════════════
legend_patches = [mpatches.Patch(color=PALETTE[c], label=f'질환 {c}')
                  for c in CLASS_ORDER]
STEP_COLORS = ['#2C3E50', '#1E8449', '#BA4A00', '#6C3483']

def step_badge(ax, step_n, text):
    """STEP 배지를 axes 상단 내부에 오버레이."""
    ax.text(0.01, 0.99, f" STEP {step_n} → {text}",
            transform=ax.transAxes, fontsize=9.5, fontweight='bold',
            color='white', va='top', ha='left',
            bbox=dict(boxstyle='round,pad=0.35',
                      fc=STEP_COLORS[step_n - 1], ec='none', alpha=0.92))


for target in TARGETS:
    cfg   = DISEASE_CFG[target]
    t_col = cfg['color']
    good  = cfg['good']
    bad   = cfg['bad']

    H_D     = entropy(d[target])
    ig_list = ig_by_target[target]

    # ── figure & gridspec ────────────────────────────────────────────────────
    fig = plt.figure(figsize=(24, 13))
    fig.patch.set_facecolor('#F5F6FA')
    fig.suptitle(
        f"Information Gain 기반 중요 변수 추출 과정  |  대상 질환: {target}",
        fontsize=15, fontweight='bold', y=0.995
    )

    # 2행 × 3열: 좌열·우열은 전체 높이, 중앙열은 상하 분리
    gs = gridspec.GridSpec(
        2, 3, figure=fig,
        width_ratios=[0.85, 2.2, 1.5],
        height_ratios=[1, 1],
        hspace=0.42, wspace=0.28,
        left=0.06, right=0.98, top=0.94, bottom=0.06
    )
    ax1 = fig.add_subplot(gs[:, 0])   # STEP 1 — 좌 전체
    ax2 = fig.add_subplot(gs[0, 1])   # STEP 2 — 중앙 상
    ax3 = fig.add_subplot(gs[1, 1])   # STEP 3 — 중앙 하
    ax4 = fig.add_subplot(gs[:, 2])   # STEP 4 — 우 전체

    # H(D) 배지: figure 좌상단
    fig.text(0.005, 0.97, f"H(D) = {H_D:.3f}",
             fontsize=13, fontweight='bold', color='white', va='top',
             bbox=dict(boxstyle='round,pad=0.45', fc=t_col, ec='none'))

    # ── STEP 1: 전체 분포 ────────────────────────────────────────────────────
    ax1.set_facecolor('#FFFFFF')
    H_val = stacked_bar(ax1, 0.5, d[target], bar_w=0.45)
    ax1.set_xlim(0, 1); ax1.set_ylim(0, 1.22)
    ax1.set_xticks([0.5])
    ax1.set_xticklabels([f"전체 D\n(n={n_total:,})"], fontsize=10)
    ax1.set_ylabel("클래스 비율", fontsize=10)
    ax1.legend(handles=legend_patches, loc='upper right', fontsize=9)
    ax1.grid(axis='y', alpha=0.2)
    ax1.set_title(f"전체 [{target}] 분포\n→ H(D) 계산", fontsize=10, pad=4, loc='left')
    ax1.text(0.5, 0.02, "클래스가 골고루\n섞여 있어 불순도 높음",
             ha='center', fontsize=8.5, color='#777', transform=ax1.transAxes)
    step_badge(ax1, 1, f"전체 분포 → H(D) 계산")

    # ── STEP 2: IG 높은 피처 ─────────────────────────────────────────────────
    ax2.set_facecolor('#F0FFF4')
    groups_good = d.groupby(good, observed=True)[target]
    grp_names   = [g for g in CLASS_ORDER if g in groups_good.groups]
    x_pos2      = np.linspace(0.12, 0.88, len(grp_names))
    bw2         = min(0.20, 0.55 / len(grp_names))
    H_after2    = 0

    for xi, grp_name in zip(x_pos2, grp_names):
        grp = groups_good.get_group(grp_name)
        n_g = len(grp); w = n_g / n_total
        H_g = stacked_bar(ax2, xi, grp, bar_w=bw2)
        H_after2 += w * H_g
        ax2.text(xi, 1.04, f"{grp_name}\nn={n_g:,}  H={H_g:.3f}\n가중치={w:.2f}",
                 ha='center', fontsize=8.5, fontweight='bold', va='bottom', color='#1A5276')

    IG_g = H_D - H_after2
    ax2.set_xlim(0, 1); ax2.set_ylim(0, 1.40)
    ax2.set_xticks(x_pos2)
    ax2.set_xticklabels([f"{good}={g}" for g in grp_names], fontsize=9)
    ax2.set_ylabel("클래스 비율", fontsize=9)
    ax2.set_title(
        f"IG 높은 피처 예시: [{good}]  —  각 그룹의 당뇨정상만 남음\n"
        f"IG = H(D) - 가중평균H  =  {H_D:.3f} - {H_after2:.3f}  =  {IG_g:.4f}  (높음 → 유용한 변수)",
        fontsize=9.5, pad=4, loc='left')
    ax2.text(0.99, 0.97, f"IG = {IG_g:.4f}",
             ha='right', va='top', fontsize=13, fontweight='bold', color='white',
             transform=ax2.transAxes,
             bbox=dict(boxstyle='round,pad=0.4', fc='#1E8449', ec='none'))
    ax2.grid(axis='y', alpha=0.2)
    step_badge(ax2, 2, f"IG 높은 피처 예시: [{good}]")

    # ── STEP 3: IG 낮은 피처 ─────────────────────────────────────────────────
    ax3.set_facecolor('#FFF8F0')
    groups_bad  = d.groupby(bad, observed=True)[target]
    bad_grps    = list(groups_bad.groups.keys())
    x_pos3      = np.linspace(0.20, 0.80, len(bad_grps))
    bw3         = min(0.26, 0.5 / len(bad_grps))
    H_after3    = 0

    for xi, grp_name in zip(x_pos3, bad_grps):
        grp = groups_bad.get_group(grp_name)
        n_g = len(grp); w = n_g / n_total
        H_g = stacked_bar(ax3, xi, grp, bar_w=bw3)
        H_after3 += w * H_g
        ax3.text(xi, 1.04, f"{grp_name}\nn={n_g:,}  H={H_g:.3f}\n가중치={w:.2f}",
                 ha='center', fontsize=8.5, fontweight='bold', va='bottom', color='#641E16')

    IG_b = H_D - H_after3
    ax3.set_xlim(0, 1); ax3.set_ylim(0, 1.40)
    ax3.set_xticks(x_pos3)
    ax3.set_xticklabels([f"{bad}={g}" for g in bad_grps], fontsize=9)
    ax3.set_ylabel("클래스 비율", fontsize=9)
    ax3.set_title(
        f"IG 낮은 피처 예시: [{bad}]  —  비율이 비슷하게 혼잡\n"
        f"IG = H(D) - 가중평균H  =  {H_D:.3f} - {H_after3:.3f}  =  {IG_b:.4f}  (낮음 → 비유용한 변수)",
        fontsize=9.5, pad=4, loc='left')
    ax3.text(0.99, 0.97, f"IG = {IG_b:.4f}",
             ha='right', va='top', fontsize=13, fontweight='bold', color='white',
             transform=ax3.transAxes,
             bbox=dict(boxstyle='round,pad=0.4', fc='#BA4A00', ec='none'))
    ax3.grid(axis='y', alpha=0.2)
    step_badge(ax3, 3, f"IG 낮은 피처 예시: [{bad}]")

    # ── 중앙 → 화살표 (figure 좌표) ──────────────────────────────────────────
    for y_fig in [0.73, 0.30]:
        fig.text(0.365, y_fig, '→', fontsize=20, color='#AAA',
                 ha='center', va='center', fontweight='bold')

    # ── STEP 4: 전체 순위 ────────────────────────────────────────────────────
    ax4.set_facecolor('#FFFFFF')
    feats  = [f for f, _ in ig_list]
    values = [v for _, v in ig_list]
    colors = ['#1E8449' if f == good else '#BA4A00' if f == bad else t_col
              for f in feats]

    bars = ax4.barh(range(len(feats)), values,
                    color=colors, alpha=0.87, edgecolor='white', linewidth=0.5)
    ax4.set_yticks(range(len(feats)))
    ax4.set_yticklabels(feats, fontsize=9.5)
    ax4.invert_yaxis()

    for bar, val in zip(bars, values):
        ax4.text(val + max(values)*0.012, bar.get_y() + bar.get_height()/2,
                 f"{val:.4f}", va='center', fontsize=8.5)

    ax4.set_xlabel("Information Gain", fontsize=10)
    ax4.set_xlim(0, max(values) * 1.22)
    ax4.set_title("모든 피처 IG 계산\n→ IG 순으로 정렬 → 상위 피처",
                  fontsize=10, pad=4, loc='left')
    ax4.grid(axis='x', alpha=0.25)
    ax4.legend(handles=[
        mpatches.Patch(color='#1E8449', label=f'{good}  (IG 최고 → 선택)'),
        mpatches.Patch(color='#BA4A00', label=f'{bad}  (IG 최하위권 → 제외)'),
        mpatches.Patch(color=t_col,     label='기타 피처'),
    ], loc='lower right', fontsize=8.5)
    step_badge(ax4, 4, "전체 반복 → IG 순으로 정렬 → 상위 선택")

    out = f"{RESULT_DIR}/infogain_process_{target}.png"
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"저장: {out}")

print(f"\n완료: {RESULT_DIR}/")
