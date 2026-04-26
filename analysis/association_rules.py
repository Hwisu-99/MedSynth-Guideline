"""
STEP 5 — Association Rule Mining (FP-Growth)

bi_exploration.py 에서 연관 규칙 분석만 단독 실행하는 스크립트

실행:
    python analysis/association_rules.py              # 정상 포함 전체 규칙
    python analysis/association_rules.py --abnormal   # 질환 주의/위험 규칙만
"""

import os
import json
import time
import argparse
import threading
import itertools
import warnings
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()
parser.add_argument('--abnormal', action='store_true',
                    help='질환 주의/위험 포함 규칙만 추출 (정상 제외)')
args = parser.parse_args()


# ── 진행 표시기 ───────────────────────────────────────────────────────────────
class Spinner:
    """백그라운드 스레드로 스피너와 경과 시간을 출력."""

    FRAMES = ['|', '/', '-', '\\']

    def __init__(self, message: str = "실행 중"):
        self.message  = message
        self._stop    = threading.Event()
        self._thread  = threading.Thread(target=self._spin, daemon=True)
        self._start   = None

    def _spin(self):
        for frame in itertools.cycle(self.FRAMES):
            if self._stop.is_set():
                break
            elapsed = time.perf_counter() - self._start
            mins, secs = divmod(int(elapsed), 60)
            time_str = f"{mins}분 {secs:02d}초" if mins else f"{secs}초"
            print(f"\r  {frame}  {self.message} ... ({time_str} 경과)",
                  end='', flush=True)
            time.sleep(0.3)

    def __enter__(self):
        self._start = time.perf_counter()
        self._thread.start()
        return self

    def __exit__(self, *_):
        self._stop.set()
        self._thread.join()
        elapsed = time.perf_counter() - self._start
        mins, secs = divmod(elapsed, 60)
        time_str = f"{int(mins)}분 {secs:.1f}초" if mins else f"{elapsed:.1f}초"
        print(f"\r  ✓  {self.message} 완료  ({time_str})" + " " * 10)


def _step(msg: str):
    """단계 시작 시 타임스탬프와 함께 출력."""
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"\n[{ts}] {msg}")

try:
    from mlxtend.frequent_patterns import fpgrowth, association_rules as mlxtend_rules
except ImportError:
    print("[오류] mlxtend 없음 — pip install mlxtend")
    exit()

plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

RESULT_DIR = f"result/analysis/{datetime.now().strftime('%Y%m%d_%H%M%S')}"
os.makedirs(RESULT_DIR, exist_ok=True)
print(f"결과 저장 경로: {RESULT_DIR}\n")

# ── 0. 파라미터 선정 ───────────────────────────────────────────────────────────────
REAL_PATH = "data/preprocessed/health_data_2024_preprocessed.csv"
TARGETS        = ['당뇨', '고혈압', '간기능']
ABNORMAL_ONLY  = args.abnormal  # --abnormal 플래그로 제어

MIN_SUPPORT    = 0.10   # 지지도
MIN_CONFIDENCE = 0.70   # 신뢰도 
MIN_LIFT       = 1.5    # 리프트 

json.dump({
    'MIN_SUPPORT': MIN_SUPPORT,
    'MIN_CONFIDENCE': MIN_CONFIDENCE,
    'MIN_LIFT': MIN_LIFT,
    'ABNORMAL_ONLY': ABNORMAL_ONLY,
}, open(f"{RESULT_DIR}/params.json", 'w'), ensure_ascii=False, indent=2)
print(f"파라미터 저장: {RESULT_DIR}/params.json")

# ── 1. 데이터 로드 ────────────────────────────────────────────────────────────
_step("데이터 로드")
df = pd.read_csv(REAL_PATH, encoding="utf-8-sig")
if '이름' in df.columns:
    df = df.drop(columns=['이름'])
print(f"  로드: {len(df):,}행 × {len(df.columns)}열")

# ── 2. 이산화 ────────────────────────────────────────────────────────────────
_step("이산화 (Discretization)")
시도_map = {
    11: '서울', 26: '부산', 27: '대구', 28: '인천', 29: '광주',
    30: '대전', 31: '울산', 36: '세종', 41: '경기', 42: '강원',
    43: '충북', 44: '충남', 45: '전북', 46: '전남',
    47: '경북', 48: '경남', 49: '제주',
}

d = pd.DataFrame(index=df.index)
d['성별']       = df['성별코드'].map({1: '남', 2: '여'})
d['연령대']     = pd.cut(df['연령대코드(5세단위)'],
                        bins=[0, 4, 7, 10, 13, 99],
                        labels=['미성년(~19세)', '청년(20-34세)', '중년(35-49세)',
                                '장년(50-64세)', '노년(65세+)'])
# 시도코드는 17개 지역 → one-hot 시 컬럼 폭발로 ARM 제외 (t-weight 분석에서만 사용)
d['BMI']        = pd.cut(df['BMI'],
                         bins=[0, 18.5, 22.9, 24.9, 999],
                         labels=['저체중', '정상', '과체중', '비만'])
d['수축기혈압'] = pd.cut(df['수축기혈압'],
                        bins=[0, 120, 140, 999],
                        labels=['정상', '주의', '위험'])
d['이완기혈압'] = pd.cut(df['이완기혈압'],
                        bins=[0, 80, 90, 999],
                        labels=['정상', '주의', '위험'])
d['혈색소']     = pd.cut(df['혈색소'],
                        bins=[0, 12, 16, 999],
                        labels=['낮음', '정상', '높음'])
d['식전혈당']   = pd.cut(df['식전혈당(공복혈당)'],
                        bins=[0, 100, 126, 999],
                        labels=['정상', '주의', '위험'])
d['혈청크레아티닌'] = pd.cut(df['혈청크레아티닌'],
                           bins=[0, 0.7, 1.2, 999],
                           labels=['낮음', '정상', '높음'])
d['AST']        = pd.cut(df['혈청지오티(AST)'],
                         bins=[0, 40, 60, 999],
                         labels=['정상', '주의', '위험'])
d['ALT']        = pd.cut(df['혈청지피티(ALT)'],
                         bins=[0, 40, 60, 999],
                         labels=['정상', '주의', '위험'])
d['감마지티피'] = pd.cut(df['감마지티피'],
                        bins=[0, 35, 63, 999],
                        labels=['정상', '주의', '위험'])
d['흡연상태']   = df['흡연상태'].map({1: '비흡연', 2: '과거흡연', 3: '현재흡연'})
d['음주여부']   = df['음주여부'].map({0: '비음주', 1: '음주'})
disease_map = {0: '정상', 1: '주의', 2: '위험'}
for col in TARGETS:
    d[col] = df[col].map(disease_map)

df_disc = d
print(f"  이산화 완료: {df_disc.shape}")

# ── 3. One-hot 인코딩 ─────────────────────────────────────────────────────────
_step("One-hot 인코딩")
with Spinner("One-hot 인코딩"):
    df_oh = pd.get_dummies(df_disc.astype(str), prefix_sep='=').astype(bool)
print(f"  One-hot 컬럼 수: {df_oh.shape[1]}개")

# ── 4. FP-Growth — 빈발 항목집합 탐색 ────────────────────────────────────────
_step(f"FP-Growth  (min_support={MIN_SUPPORT})")
with Spinner("FP-Growth 빈발 항목집합 탐색"):
    frequent_items = fpgrowth(df_oh, min_support=MIN_SUPPORT, use_colnames=True)
    frequent_items = frequent_items.sort_values('support', ascending=False)
print(f"  빈발 항목집합: {len(frequent_items):,}개")

# ── 5. 연관 규칙 생성 ─────────────────────────────────────────────────────────
_step(f"연관 규칙 생성  (min_confidence={MIN_CONFIDENCE}, min_lift={MIN_LIFT})")
with Spinner("연관 규칙 생성"):
    rules = mlxtend_rules(frequent_items, metric="confidence",
                          min_threshold=MIN_CONFIDENCE)
    n_rules_raw = len(rules)
    rules = rules[rules['lift'] >= MIN_LIFT]
    n_rules_lift = len(rules)
    rules = rules.sort_values('lift', ascending=False).reset_index(drop=True)
print(f"  최종 규칙 수: {len(rules):,}개  (confidence≥{MIN_CONFIDENCE}, lift≥{MIN_LIFT})")

# 읽기 쉬운 문자열 변환
rules['조건부(if)']  = rules['antecedents'].apply(lambda x: ' & '.join(sorted(x)))
rules['결론부(then)'] = rules['consequents'].apply(lambda x: ' & '.join(sorted(x)))

# 쌍방향 중복 제거: A→B 와 B→A 중 lift가 높은 방향 하나만 유지
# (이미 lift 내림차순 정렬이므로 먼저 등장한 쪽이 더 높은 lift)
seen_pairs: set = set()
keep_idx = []
for idx, row in rules.iterrows():
    key = frozenset(row['antecedents']) | frozenset(row['consequents'])
    if key not in seen_pairs:
        seen_pairs.add(key)
        keep_idx.append(idx)
rules = rules.loc[keep_idx].reset_index(drop=True)
print(f"  중복 제거 후 규칙 수: {len(rules):,}개")

# ── 6. --abnormal 시 전체 rules 자체를 비정상 규칙으로 사전 필터링 ────────────
if ABNORMAL_ONLY:
    abnormal_pattern = '|'.join(f'{t}=주의|{t}=위험' for t in TARGETS)
    # 조건1: 조건부 또는 결론부에 질환 비정상(주의/위험) 포함
    has_disease_abnormal = (rules['조건부(if)'].str.contains(abnormal_pattern) |
                            rules['결론부(then)'].str.contains(abnormal_pattern))
    # 조건2: 조건부/결론부 양쪽 모두 =정상 값이 하나도 없어야 함
    no_normal = (~rules['조건부(if)'].str.contains('=정상') &
                 ~rules['결론부(then)'].str.contains('=정상'))
    rules = rules[has_disease_abnormal & no_normal].reset_index(drop=True)
    filter_label = "질환 비정상(주의/위험) 규칙"
    print(f"  --abnormal 필터 후 규칙 수: {len(rules):,}개")
else:
    filter_label = "질환 관련 규칙 (정상 포함)"

n_rules_final = len(rules)
display_cols = ['조건부(if)', '결론부(then)', 'support', 'confidence', 'lift']

print(f"\n[전체 규칙 상위 20개 — lift 기준]")
print(rules[display_cols].head(20).round(4).to_string(index=False))
rules[display_cols].to_csv(f"{RESULT_DIR}/association_rules_all.csv",
                           index=False, encoding="utf-8-sig")

# ── 7. 질환 관련 규칙 필터 ───────────────────────────────────────────────────
if ABNORMAL_ONLY:
    disease_pattern = abnormal_pattern  # 이미 위에서 정의
else:
    disease_pattern = '|'.join(TARGETS)

rules_disease = rules[
    rules['결론부(then)'].str.contains(disease_pattern) |
    rules['조건부(if)'].str.contains(disease_pattern)
].reset_index(drop=True)
print(f"\n[{filter_label}: {len(rules_disease)}개]")
if len(rules_disease):
    print(rules_disease[display_cols].head(30).round(4).to_string(index=False))
    rules_disease[display_cols].to_csv(
        f"{RESULT_DIR}/association_rules_disease.csv",
        index=False, encoding="utf-8-sig")

# ── 7. 시각화 ─────────────────────────────────────────────────────────────────
# 전체 규칙 산점도 (support × confidence, 색=lift)
fig, ax = plt.subplots(figsize=(10, 6))
sc = ax.scatter(rules['support'], rules['confidence'],
                c=rules['lift'], cmap='RdYlGn_r',
                s=rules['lift'] * 15, alpha=0.6,
                edgecolors='gray', linewidth=0.3)
plt.colorbar(sc, ax=ax, label='Lift')
ax.axhline(MIN_CONFIDENCE, color='red', linestyle='--',
           linewidth=0.8, label=f'최소 신뢰도 {MIN_CONFIDENCE}')
ax.set_xlabel("Support (지지도)")
ax.set_ylabel("Confidence (신뢰도)")
# ax.set_title("전체 연관 규칙 분포\n(색상·크기 = Lift, 오른쪽 위·빨강일수록 강한 규칙)", fontsize=12)
ax.legend()
plt.tight_layout()
plt.savefig(f"{RESULT_DIR}/rules_scatter.png", dpi=150, bbox_inches='tight')
plt.show()
print(f"저장: {RESULT_DIR}/rules_scatter.png")

# 질환 관련 규칙 상위 15개 막대그래프
if len(rules_disease):
    top15 = rules_disease.head(15)
    fig, ax = plt.subplots(figsize=(24, 10))
    colors = ['#E74C3C' if '당뇨' in t else '#3498DB' if '고혈압' in t else '#2ECC71'
              for t in top15['결론부(then)']]
    bars = ax.barh(range(len(top15)), top15['lift'], color=colors, alpha=0.85)
    ax.set_yticks(range(len(top15)))
    ax.set_yticklabels(
        [f"{row['조건부(if)']}  →  {row['결론부(then)']}"
         for _, row in top15.iterrows()],
        fontsize=9)
    for bar, val in zip(bars, top15['lift']):
        ax.text(val + 0.05, bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}", va='center', fontsize=10)
    ax.set_xlabel("Lift (향상도)", fontsize=12)
    ax.set_title("질환 관련 연관 규칙 상위 15개 (Lift 기준)\n"
                 "빨강=당뇨  파랑=고혈압  초록=간기능", fontsize=13)
    ax.invert_yaxis()
    plt.subplots_adjust(left=0.45)
    plt.savefig(f"{RESULT_DIR}/rules_disease_bar.png", dpi=150, bbox_inches='tight')
    plt.show()
    print(f"저장: {RESULT_DIR}/rules_disease_bar.png")


# ── 8. FP-Growth 파이프라인 시각화 (발표용) ──────────────────────────────────
_step("FP-Growth 파이프라인 시각화 (발표용)")

fig = plt.figure(figsize=(22, 14))
fig.suptitle("FP-Growth 연관 규칙 마이닝 파이프라인", fontsize=17, fontweight='bold', y=0.99)

# Panel 1 (좌상): 빈발 단일 항목 Top 15
ax1 = fig.add_subplot(2, 2, 1)
single_items = frequent_items[frequent_items['itemsets'].apply(len) == 1].copy()
single_items['item'] = single_items['itemsets'].apply(lambda x: list(x)[0])
single_items = single_items.nlargest(15, 'support')
bar_colors1 = ['#E74C3C' if any(t in item for t in TARGETS) else '#5B9BD5'
               for item in single_items['item']]
bars1 = ax1.barh(range(len(single_items)), single_items['support'],
                 color=bar_colors1, alpha=0.85)
ax1.set_yticks(range(len(single_items)))
ax1.set_yticklabels(single_items['item'], fontsize=9)
ax1.invert_yaxis()
ax1.axvline(MIN_SUPPORT, color='red', linestyle='--', linewidth=1,
            label=f'min_support={MIN_SUPPORT}')
ax1.set_xlabel("Support (지지도)", fontsize=10)
ax1.set_title(f"① 빈발 단일 항목 Top 15\n(빨강=질환 관련)", fontsize=11, fontweight='bold')
ax1.legend(fontsize=8)
for bar, val in zip(bars1, single_items['support']):
    ax1.text(val + 0.003, bar.get_y() + bar.get_height() / 2,
             f"{val:.3f}", va='center', fontsize=8)

# Panel 2 (우상): 항목집합 크기별 분포
ax2 = fig.add_subplot(2, 2, 2)
size_counts = frequent_items['itemsets'].apply(len).value_counts().sort_index()
cmap2 = plt.cm.Blues
bar_colors2 = [cmap2(0.35 + 0.12 * i) for i in range(len(size_counts))]
bars2 = ax2.bar(size_counts.index, size_counts.values,
                color=bar_colors2, alpha=0.9, edgecolor='white', width=0.6)
ax2.set_xlabel("항목집합 크기 (아이템 수)", fontsize=10)
ax2.set_ylabel("빈발 항목집합 수", fontsize=10)
ax2.set_title(f"② 빈발 항목집합 크기별 분포\n(총 {len(frequent_items):,}개, min_support={MIN_SUPPORT})",
              fontsize=11, fontweight='bold')
ax2.set_xticks(size_counts.index)
for bar, val in zip(bars2, size_counts.values):
    ax2.text(bar.get_x() + bar.get_width() / 2, val + max(size_counts.values) * 0.01,
             f"{val:,}", ha='center', fontsize=9)

# Panel 3 (좌하): 규칙 생성 깔때기
ax3 = fig.add_subplot(2, 2, 3)
funnel_stages = [
    (f"원본 트랜잭션\n{len(df):,}건", len(df)),
    (f"빈발 항목집합\nsupport ≥ {MIN_SUPPORT}\n{len(frequent_items):,}개", len(frequent_items)),
    (f"후보 규칙\nconfidence ≥ {MIN_CONFIDENCE}\n{n_rules_raw:,}개", n_rules_raw),
    (f"Lift 필터\nlift ≥ {MIN_LIFT}\n{n_rules_lift:,}개", n_rules_lift),
    (f"중복 제거 후\n최종 규칙\n{n_rules_final:,}개", n_rules_final),
]
funnel_colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#2ECC71']
max_val = funnel_stages[0][1]
for i, (label, val) in enumerate(funnel_stages):
    width = max(val / max_val, 0.08)
    left = (1 - width) / 2
    ax3.barh(i, width, left=left, color=funnel_colors[i], alpha=0.88, height=0.65)
    ax3.text(0.5, i, label, ha='center', va='center',
             fontsize=9, color='white', fontweight='bold')
    if i > 0:
        prev_val = funnel_stages[i - 1][1]
        ratio = val / prev_val * 100 if prev_val else 0
        ax3.text(0.97, i - 0.5, f"▼ {ratio:.1f}%", ha='right', va='center',
                 fontsize=8, color='gray')
ax3.set_xlim(0, 1)
ax3.set_ylim(-0.5, len(funnel_stages) - 0.5)
ax3.invert_yaxis()
ax3.axis('off')
ax3.set_title("③ 규칙 생성 단계별 필터링 (Funnel)", fontsize=11, fontweight='bold')

# Panel 4 (우하): 최종 규칙 산점도
ax4 = fig.add_subplot(2, 2, 4)
sc4 = ax4.scatter(rules['support'], rules['confidence'],
                  c=rules['lift'], cmap='RdYlGn_r',
                  s=rules['lift'] * 12, alpha=0.65,
                  edgecolors='gray', linewidth=0.2)
plt.colorbar(sc4, ax=ax4, label='Lift')
ax4.axhline(MIN_CONFIDENCE, color='red', linestyle='--', linewidth=0.8,
            label=f'min_confidence={MIN_CONFIDENCE}')
ax4.set_xlabel("Support (지지도)", fontsize=10)
ax4.set_ylabel("Confidence (신뢰도)", fontsize=10)
ax4.set_title(f"④ 최종 규칙 분포\n({n_rules_final:,}개, 빨강=높은 Lift)", fontsize=11, fontweight='bold')
ax4.legend(fontsize=8)

plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.savefig(f"{RESULT_DIR}/fpgrowth_pipeline.png", dpi=150, bbox_inches='tight')
plt.show()
print(f"저장: {RESULT_DIR}/fpgrowth_pipeline.png")

print(f"\n완료: {RESULT_DIR}/")
