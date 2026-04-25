"""
STEP 5 — Association Rule Mining (FP-Growth)

bi_exploration.py 에서 연관 규칙 분석만 단독 실행하는 스크립트

실행:
    python analysis/association_rules.py
"""

import os
import time
import threading
import itertools
import warnings
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

warnings.filterwarnings("ignore")


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

REAL_PATH = "data/preprocessed/health_data_2024_preprocessed.csv"
TARGETS   = ['당뇨', '고혈압', '간기능']

MIN_SUPPORT    = 0.10   # 10% 이상 (1000건+) — 규칙 수 폭발 방지
MIN_CONFIDENCE = 0.60   # 신뢰도 60% 이상
MIN_LIFT       = 1.1    # 리프트 1.1 이상

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
    rules = rules[rules['lift'] >= MIN_LIFT]
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

display_cols = ['조건부(if)', '결론부(then)', 'support', 'confidence', 'lift']

print(f"\n[전체 규칙 상위 20개 — lift 기준]")
print(rules[display_cols].head(20).round(4).to_string(index=False))
rules[display_cols].to_csv(f"{RESULT_DIR}/association_rules_all.csv",
                           index=False, encoding="utf-8-sig")

# ── 6. 질환 관련 규칙 필터 ───────────────────────────────────────────────────
disease_pattern = '|'.join(TARGETS)
rules_disease = rules[
    rules['결론부(then)'].str.contains(disease_pattern)
].reset_index(drop=True)
print(f"\n[질환 관련 규칙: {len(rules_disease)}개]")
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
ax.set_title("전체 연관 규칙 분포\n(색상·크기 = Lift, 오른쪽 위·초록일수록 강한 규칙)", fontsize=12)
ax.legend()
plt.tight_layout()
plt.savefig(f"{RESULT_DIR}/rules_scatter.png", dpi=150, bbox_inches='tight')
plt.show()
print(f"저장: {RESULT_DIR}/rules_scatter.png")

# 질환 관련 규칙 상위 15개 막대그래프
if len(rules_disease):
    top15 = rules_disease.head(15)
    fig, ax = plt.subplots(figsize=(12, 6))
    colors = ['#E74C3C' if '당뇨' in t else '#3498DB' if '고혈압' in t else '#2ECC71'
              for t in top15['결론부(then)']]
    bars = ax.barh(range(len(top15)), top15['lift'], color=colors, alpha=0.85)
    ax.set_yticks(range(len(top15)))
    ax.set_yticklabels(
        [f"{row['조건부(if)']}  →  {row['결론부(then)']}"
         for _, row in top15.iterrows()],
        fontsize=8)
    for bar, val in zip(bars, top15['lift']):
        ax.text(val + 0.01, bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}", va='center', fontsize=8)
    ax.set_xlabel("Lift (향상도)")
    ax.set_title("질환 관련 연관 규칙 상위 15개 (Lift 기준)\n"
                 "빨강=당뇨  파랑=고혈압  초록=간기능", fontsize=11)
    ax.invert_yaxis()
    plt.tight_layout()
    plt.savefig(f"{RESULT_DIR}/rules_disease_bar.png", dpi=150, bbox_inches='tight')
    plt.show()
    print(f"저장: {RESULT_DIR}/rules_disease_bar.png")

print(f"\n완료: {RESULT_DIR}/")
