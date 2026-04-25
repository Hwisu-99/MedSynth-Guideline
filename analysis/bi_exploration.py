"""
BI/BR 탐사 스크립트 — 건강검진 데이터셋

STEP 1 : Attribute Removal    — 고유값 기반 속성 필터링
STEP 2 : Discretization       — 의학적 기준 기반 이산화
STEP 3 : Entropy & Info Gain  — 질환별 핵심 변수 도출
STEP 4 : t-weight / d-weight  — 지역 × 질환 교차 분석
STEP 5 : Association Rules    — 연관 규칙 탐사 (FP-Growth)

실행:
    python analysis/bi_exploration.py
"""

import os
import sys
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from datetime import datetime

warnings.filterwarnings("ignore")

try:
    from mlxtend.frequent_patterns import fpgrowth, association_rules as mlxtend_rules
    HAS_MLXTEND = True
except ImportError:
    HAS_MLXTEND = False
    print("[경고] mlxtend 없음 — 연관 규칙 분석 건너뜀.  pip install mlxtend")

plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

RESULT_DIR = f"result/analysis/{datetime.now().strftime('%Y%m%d_%H%M%S')}"
os.makedirs(RESULT_DIR, exist_ok=True)
print(f"결과 저장 경로: {RESULT_DIR}\n")

REAL_PATH = "data/preprocessed/health_data_2024_preprocessed.csv"
TARGETS   = ['당뇨', '고혈압', '간기능']
TARGET_COLORS = {'당뇨': '#E74C3C', '고혈압': '#3498DB', '간기능': '#2ECC71'}


# ────────────────────────────────────────────────────────────────────────────
# 헬퍼
# ────────────────────────────────────────────────────────────────────────────
def _section(title: str, width: int = 68):
    print(f"\n{'=' * width}\n  {title}\n{'=' * width}")


def _entropy(series: pd.Series) -> float:
    probs = series.value_counts(normalize=True).values
    probs = probs[probs > 0]
    return float(-np.sum(probs * np.log2(probs)))


def _info_gain(df: pd.DataFrame, feature: str, target: str) -> float:
    H_D = _entropy(df[target])
    n   = len(df)
    H_after = sum(
        (len(group) / n) * _entropy(group[target])
        for _, group in df.groupby(feature, observed=True)
    )
    return H_D - H_after


# ────────────────────────────────────────────────────────────────────────────
# STEP 1 : 데이터 로드 & Attribute Removal
# ────────────────────────────────────────────────────────────────────────────
_section("STEP 1 — 데이터 로드 & Attribute Removal (U_i 기반)")

df = pd.read_csv(REAL_PATH, encoding="utf-8-sig")
if '이름' in df.columns:
    df = df.drop(columns=['이름'])
n = len(df)
print(f"로드: {n:,}행 × {len(df.columns)}열")

U_THRESHOLD = 0.05   # 고유값 비율이 5% 초과 시 식별자성 속성으로 간주
removal_rows = []
candidate_cols = []

for col in df.columns:
    ratio    = df[col].nunique() / n
    removed  = (ratio > U_THRESHOLD) and (col not in TARGETS)
    removal_rows.append({
        '컬럼':       col,
        '고유값 수':   df[col].nunique(),
        '고유값 비율': f"{ratio:.4f}",
        '역할':       'TARGET' if col in TARGETS else ('제거' if removed else '후보'),
    })
    if not removed:
        candidate_cols.append(col)

df_attr = pd.DataFrame(removal_rows)
print(f"\n{df_attr.to_string(index=False)}")
print(f"\n→ 후보 속성 ({len([c for c in candidate_cols if c not in TARGETS])}개): "
      f"{[c for c in candidate_cols if c not in TARGETS]}")

df_attr.to_csv(f"{RESULT_DIR}/attr_removal.csv", index=False, encoding="utf-8-sig")


# ────────────────────────────────────────────────────────────────────────────
# STEP 2 : 이산화 (의학적 기준)
# ────────────────────────────────────────────────────────────────────────────
_section("STEP 2 — Discretization (의학적 기준)")

def discretize(df: pd.DataFrame) -> pd.DataFrame:
    d = pd.DataFrame(index=df.index)

    # 인구통계
    d['성별'] = df['성별코드'].map({1: '남', 2: '여'})

    # 연령대코드(5세단위) 실제 코드표
    # 1=0-4세, 2=5-9세, ..., 5=20-24세, 8=35-39세, 10=45-49세, 14=65-69세, 18=85세+
    d['연령대'] = pd.cut(df['연령대코드(5세단위)'],
                       bins=[0, 4, 7, 10, 13, 99],
                       labels=['미성년(~19세)', '청년(20-34세)', '중년(35-49세)',
                               '장년(50-64세)', '노년(65세+)'])

    # 시도코드 → 지역명 매핑
    시도_map = {
        11: '서울', 26: '부산', 27: '대구', 28: '인천', 29: '광주',
        30: '대전', 31: '울산', 36: '세종', 41: '경기', 42: '강원',
        43: '충북', 44: '충남', 45: '전북', 46: '전남',
        47: '경북', 48: '경남', 49: '제주',
    }
    d['시도코드'] = df['시도코드'].map(시도_map)

    # 체형
    d['BMI'] = pd.cut(df['BMI'],
                      bins=[0, 18.5, 22.9, 24.9, 999],
                      labels=['저체중(<18.5)', '정상(18.5-22.9)', '과체중(23-24.9)', '비만(≥25)'])

    # 혈압
    d['수축기혈압'] = pd.cut(df['수축기혈압'],
                          bins=[0, 120, 140, 999],
                          labels=['정상(<120)', '주의(120-139)', '위험(≥140)'])
    d['이완기혈압'] = pd.cut(df['이완기혈압'],
                          bins=[0, 80, 90, 999],
                          labels=['정상(<80)', '주의(80-89)', '위험(≥90)'])

    # 혈액
    d['혈색소']       = pd.cut(df['혈색소'],
                             bins=[0, 12, 16, 999],
                             labels=['낮음(<12)', '정상(12-16)', '높음(>16)'])
    d['식전혈당']     = pd.cut(df['식전혈당(공복혈당)'],
                             bins=[0, 100, 126, 999],
                             labels=['정상(<100)', '주의(100-125)', '위험(≥126)'])
    d['혈청크레아티닌'] = pd.cut(df['혈청크레아티닌'],
                              bins=[0, 0.7, 1.2, 999],
                              labels=['낮음(<0.7)', '정상(0.7-1.2)', '높음(>1.2)'])

    # 간 효소
    d['AST']    = pd.cut(df['혈청지오티(AST)'],
                         bins=[0, 40, 60, 999],
                         labels=['정상(≤40)', '주의(41-60)', '위험(>60)'])
    d['ALT']    = pd.cut(df['혈청지피티(ALT)'],
                         bins=[0, 40, 60, 999],
                         labels=['정상(≤40)', '주의(41-60)', '위험(>60)'])
    d['감마지티피'] = pd.cut(df['감마지티피'],
                          bins=[0, 35, 63, 999],
                          labels=['정상(≤35)', '주의(36-63)', '위험(>63)'])

    # 생활습관
    d['흡연상태'] = df['흡연상태'].map({1: '비흡연', 2: '과거흡연', 3: '현재흡연'})
    d['음주여부'] = df['음주여부'].map({0: '비음주', 1: '음주'})

    # 질환 파생변수 (타겟)
    disease_map = {0: '정상', 1: '주의', 2: '위험'}
    for col in ['당뇨', '고혈압', '간기능']:
        d[col] = df[col].map(disease_map)

    return d

df_disc = discretize(df)
FEATURE_COLS = [c for c in df_disc.columns if c not in TARGETS]
print(f"이산화 완료: {df_disc.shape}")
print(f"피처 컬럼: {FEATURE_COLS}")


# ────────────────────────────────────────────────────────────────────────────
# STEP 3 : Entropy & Information Gain
# ────────────────────────────────────────────────────────────────────────────
_section("STEP 3 — Entropy & Information Gain")

ig_rows = []
for target in TARGETS:
    H_D = _entropy(df_disc[target])
    print(f"\n[{target}]  Entropy(D) = {H_D:.4f}")
    for feat in FEATURE_COLS:
        ig = _info_gain(df_disc, feat, target)
        ig_rows.append({'타겟': target, '피처': feat,
                        '엔트로피(D)': round(H_D, 4), 'Info Gain': round(ig, 4)})

df_ig = pd.DataFrame(ig_rows).sort_values(['타겟', 'Info Gain'], ascending=[True, False])
print(f"\n{df_ig.to_string(index=False)}")
df_ig.to_csv(f"{RESULT_DIR}/info_gain.csv", index=False, encoding="utf-8-sig")

# 시각화: 타겟별 Info Gain 막대그래프
fig, axes = plt.subplots(1, len(TARGETS), figsize=(6 * len(TARGETS), 6), sharey=False)
for ax, target in zip(axes, TARGETS):
    sub = df_ig[df_ig['타겟'] == target].sort_values('Info Gain', ascending=True)
    bars = ax.barh(sub['피처'], sub['Info Gain'],
                   color=TARGET_COLORS[target], alpha=0.85)
    ax.set_xlabel("Information Gain")
    ax.set_title(f"[{target}] Info Gain\n(높을수록 예측력 강함)", fontsize=11)
    ax.axvline(0, color='gray', linewidth=0.8)
    for bar, val in zip(bars, sub['Info Gain']):
        ax.text(val + 0.0005, bar.get_y() + bar.get_height() / 2,
                f"{val:.4f}", va='center', fontsize=8)

fig.suptitle("질환별 Information Gain (핵심 변수 도출)", fontsize=13)
plt.tight_layout()
plt.savefig(f"{RESULT_DIR}/info_gain.png", dpi=150, bbox_inches='tight')
plt.show()
print(f"\n저장: {RESULT_DIR}/info_gain.png")


# ────────────────────────────────────────────────────────────────────────────
# STEP 4 : t-weight / d-weight 교차 분석 (다중 조합)
# ────────────────────────────────────────────────────────────────────────────
_section("STEP 4 — t-weight / d-weight 교차 분석")

print("""
  t-weight : 행 기준 내 열 값 비중  → P(열 | 행)
  d-weight : 열 값 전체 중 행 비중  → P(행 | 열)
""")

# 분석할 교차 조합 정의
# (행_컬럼, 열_컬럼, 행_레이블, 열_레이블, 히트맵_가로크기, 히트맵_세로크기)
CROSS_PAIRS = [
    # 질환 관련 조합
    ('시도코드',  '당뇨',   '지역',     '당뇨단계',   8, 7),
    ('시도코드',  '고혈압',  '지역',     '고혈압단계',  8, 7),
    ('시도코드',  '간기능',  '지역',     '간기능단계',  8, 7),
    ('성별',     '당뇨',   '성별',     '당뇨단계',   5, 4),
    ('성별',     '고혈압',  '성별',     '고혈압단계',  5, 4),
    ('성별',     '간기능',  '성별',     '간기능단계',  5, 4),
    ('연령대',   '당뇨',   '연령대',   '당뇨단계',   6, 4),
    ('연령대',   '고혈압',  '연령대',   '고혈압단계',  6, 4),
    ('연령대',   '간기능',  '연령대',   '간기능단계',  6, 4),
    ('흡연상태', '당뇨',   '흡연상태', '당뇨단계',   5, 4),
    ('흡연상태', '고혈압',  '흡연상태', '고혈압단계',  5, 4),
    ('흡연상태', '간기능',  '흡연상태', '간기능단계',  5, 4),
    ('음주여부', '당뇨',   '음주여부', '당뇨단계',   5, 4),
    ('음주여부', '고혈압',  '음주여부', '고혈압단계',  5, 4),
    ('음주여부', '간기능',  '음주여부', '간기능단계',  5, 4),
    ('BMI',     '당뇨',   'BMI단계', '당뇨단계',   6, 4),
    ('BMI',     '고혈압',  'BMI단계', '고혈압단계',  6, 4),
    ('BMI',     '간기능',  'BMI단계', '간기능단계',  6, 4),
    # 생활습관 × 생활습관
    ('흡연상태', '음주여부', '흡연상태', '음주여부',   5, 4),
    # 인구통계 × 인구통계
    ('성별',     '연령대',  '성별',     '연령대',    6, 4),
]

all_weight_rows = []

def _cross_heatmap(row_col, col_col, row_label, col_label,
                   fw, fh, t_or_d='t', cmap='YlOrRd', prefix=''):
    """t-weight 또는 d-weight 히트맵을 그리고 저장."""
    # 이산화된 데이터 사용 (df_disc)
    if row_col in df_disc.columns and col_col in df_disc.columns:
        cross = pd.crosstab(df_disc[row_col], df_disc[col_col])
    else:
        return None, None

    if t_or_d == 't':
        w = cross.div(cross.sum(axis=1), axis=0).round(4)
        title_suffix = f"t-weight\n({row_label} 내 {col_label} 비중)"
    else:
        w = cross.div(cross.sum(axis=0), axis=1).round(4)
        title_suffix = f"d-weight\n({col_label} 내 {row_label} 비중)"

    fig, ax = plt.subplots(figsize=(fw, fh))
    sns.heatmap(w, annot=True, fmt=".3f", cmap=cmap,
                ax=ax, linewidths=0.5,
                cbar_kws={'label': t_or_d + '-weight'})
    ax.set_title(f"[{row_label} × {col_label}] {title_suffix}", fontsize=11)
    ax.set_xlabel(col_label)
    ax.set_ylabel(row_label)
    plt.tight_layout()

    fname = f"{prefix}{t_or_d}w_{row_col}_{col_col}.png"
    plt.savefig(f"{RESULT_DIR}/{fname}", dpi=150, bbox_inches='tight')
    plt.show()
    print(f"  저장: {fname}")
    return cross, w


for row_col, col_col, row_lbl, col_lbl, fw, fh in CROSS_PAIRS:
    if row_col not in df_disc.columns or col_col not in df_disc.columns:
        print(f"  [건너뜀] {row_col} × {col_col} — 컬럼 없음")
        continue

    print(f"\n[{row_lbl} × {col_lbl}]")
    cross = pd.crosstab(df_disc[row_col], df_disc[col_col])
    t_w   = cross.div(cross.sum(axis=1), axis=0).round(4)
    d_w   = cross.div(cross.sum(axis=0), axis=1).round(4)

    # 수치 결과 수집
    for r in cross.index:
        for c in cross.columns:
            all_weight_rows.append({
                '행_컬럼': row_col, '열_컬럼': col_col,
                '행_값':   r,       '열_값':   c,
                '빈도':    cross.loc[r, c],
                't-weight': t_w.loc[r, c],
                'd-weight': d_w.loc[r, c],
            })

    # t-weight 히트맵
    _cross_heatmap(row_col, col_col, row_lbl, col_lbl, fw, fh,
                   t_or_d='t', cmap='YlOrRd')
    # d-weight 히트맵
    _cross_heatmap(row_col, col_col, row_lbl, col_lbl, fw, fh,
                   t_or_d='d', cmap='Blues')

df_weight_all = pd.DataFrame(all_weight_rows)
df_weight_all.to_csv(f"{RESULT_DIR}/weight_analysis.csv", index=False, encoding="utf-8-sig")
print(f"\n전체 교차 분석 결과 저장: weight_analysis.csv")

# 인사이트 요약: 각 질환 위험단계에서 t-weight 최고 행
_section("교차 분석 인사이트 요약 (위험단계 t-weight 최고값)")
for target in TARGETS:
    sub = df_weight_all[df_weight_all['열_컬럼'] == target]
    risk_sub = sub[sub['열_값'].isin(['위험', '2', 2])]
    if risk_sub.empty:
        continue
    top = risk_sub.nlargest(5, 't-weight')[
        ['행_컬럼', '행_값', '빈도', 't-weight', 'd-weight']]
    print(f"\n[{target} 위험단계] t-weight 상위 5:")
    print(top.to_string(index=False))


# ────────────────────────────────────────────────────────────────────────────
# STEP 5 : Association Rule Mining (FP-Growth)
# ────────────────────────────────────────────────────────────────────────────
_section("STEP 5 — Association Rule Mining (FP-Growth)")

if not HAS_MLXTEND:
    print("  mlxtend 미설치 — 건너뜀")
else:
    MIN_SUPPORT    = 0.05   # 5% 이상 (500건+)
    MIN_CONFIDENCE = 0.55   # 신뢰도 55% 이상
    MIN_LIFT       = 1.1    # 리프트 1.1 이상

    # One-hot 인코딩 (이산화된 데이터 전체 사용)
    df_oh = pd.get_dummies(df_disc.astype(str), prefix_sep='=')
    df_oh = df_oh.astype(bool)

    print(f"  FP-Growth 실행 중...  (min_support={MIN_SUPPORT})")
    frequent_items = fpgrowth(df_oh, min_support=MIN_SUPPORT, use_colnames=True)
    print(f"  빈발 항목집합: {len(frequent_items):,}개")

    rules = mlxtend_rules(frequent_items, metric="confidence",
                          min_threshold=MIN_CONFIDENCE)
    rules = rules[rules['lift'] >= MIN_LIFT].copy()
    rules = rules.sort_values('lift', ascending=False).reset_index(drop=True)
    print(f"  최종 규칙 수: {len(rules):,}개  (confidence≥{MIN_CONFIDENCE}, lift≥{MIN_LIFT})")

    # 읽기 쉬운 형태로 변환
    rules['antecedents_str'] = rules['antecedents'].apply(lambda x: ', '.join(sorted(x)))
    rules['consequents_str'] = rules['consequents'].apply(lambda x: ', '.join(sorted(x)))

    display_cols = ['antecedents_str', 'consequents_str', 'support', 'confidence', 'lift']
    print(f"\n[상위 20개 규칙]")
    print(rules[display_cols].head(20).round(4).to_string(index=False))

    rules[display_cols].to_csv(f"{RESULT_DIR}/association_rules.csv",
                               index=False, encoding="utf-8-sig")

    # 질환(당뇨/고혈압/간기능) 관련 규칙만 필터
    disease_pattern = '|'.join(TARGETS)
    rules_disease = rules[
        rules['consequents_str'].str.contains(disease_pattern)
    ].head(30)

    if len(rules_disease):
        print(f"\n[질환 관련 규칙 상위 30개]")
        print(rules_disease[display_cols].round(4).to_string(index=False))
        rules_disease[display_cols].to_csv(
            f"{RESULT_DIR}/association_rules_disease.csv",
            index=False, encoding="utf-8-sig")

    # 시각화: Support vs Confidence, 크기=Lift
    fig, ax = plt.subplots(figsize=(10, 6))
    sc = ax.scatter(rules['support'], rules['confidence'],
                    c=rules['lift'], cmap='RdYlGn',
                    s=rules['lift'] * 15, alpha=0.6, edgecolors='gray', linewidth=0.3)
    plt.colorbar(sc, ax=ax, label='Lift')
    ax.set_xlabel("Support (지지도)")
    ax.set_ylabel("Confidence (신뢰도)")
    ax.set_title("연관 규칙 분포\n(색상·크기 = Lift, 오른쪽 위일수록 강한 규칙)", fontsize=12)
    ax.axhline(MIN_CONFIDENCE, color='red', linestyle='--', linewidth=0.8,
               label=f'최소 신뢰도 {MIN_CONFIDENCE}')
    ax.legend()
    plt.tight_layout()
    plt.savefig(f"{RESULT_DIR}/association_rules.png", dpi=150, bbox_inches='tight')
    plt.show()
    print(f"\n저장: {RESULT_DIR}/association_rules.png")


# ────────────────────────────────────────────────────────────────────────────
# 완료 요약
# ────────────────────────────────────────────────────────────────────────────
_section("탐사 완료 — 저장 파일 목록")
for fname in sorted(os.listdir(RESULT_DIR)):
    print(f"  {RESULT_DIR}/{fname}")
