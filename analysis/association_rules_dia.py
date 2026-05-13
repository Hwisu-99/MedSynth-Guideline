"""
Association Rule Mining — 당뇨 이상군 전용 (DIA)

대상 파일: data/preprocessed/health_data_2024_preprocessed_DIA.csv
  - 당뇨 관련 속성만 포함
  - 정상군(당뇨=0) 제외 → 주의(1) / 위험(2) 만 포함

STEP 1 : Attribute Removal    — 고유값 기반 속성 필터링
STEP 2 : Discretization       — 의학적 기준 기반 이산화 (association_rules.py 동일)
STEP 3 : Association Rules    — 연관 규칙 탐사 (FP-Growth)

실행:
    python analysis/association_rules_dia.py
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

try:
    from mlxtend.frequent_patterns import fpgrowth, association_rules as mlxtend_rules
except ImportError:
    print("[오류] mlxtend 없음 — pip install mlxtend")
    exit()

plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

RESULT_DIR = f"result/analysis/DIA_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
os.makedirs(RESULT_DIR, exist_ok=True)
print(f"결과 저장 경로: {RESULT_DIR}\n")

DATA_PATH    = "data/preprocessed/health_data_2024_preprocessed_DIA.csv"
TARGET       = '당뇨'
TARGET_COLOR = '#E74C3C'


# ── 진행 표시기 ───────────────────────────────────────────────────────────────
class Spinner:
    """백그라운드 스레드로 스피너와 경과 시간을 출력."""

    FRAMES = ['|', '/', '-', '\\']

    def __init__(self, message: str = "실행 중"):
        self.message = message
        self._stop   = threading.Event()
        self._thread = threading.Thread(target=self._spin, daemon=True)
        self._start  = None

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


# ────────────────────────────────────────────────────────────────────────────
# STEP 1 : 데이터 로드 & Attribute Removal
# ────────────────────────────────────────────────────────────────────────────
_step("STEP 1 — 데이터 로드 & Attribute Removal (U_i 기반)")

df = pd.read_csv(DATA_PATH, encoding="utf-8-sig")
if '이름' in df.columns:
    df = df.drop(columns=['이름'])
n = len(df)
print(f"  로드: {n:,}행 × {len(df.columns)}열")
print(f"  당뇨 단계 분포: {df[TARGET].value_counts().sort_index().to_dict()}  (1=주의, 2=위험)")

U_THRESHOLD  = 0.05
removal_rows = []
candidate_cols = []

for col in df.columns:
    ratio   = df[col].nunique() / n
    removed = (ratio > U_THRESHOLD) and (col != TARGET)
    removal_rows.append({
        '컬럼':       col,
        '고유값 수':   df[col].nunique(),
        '고유값 비율': f"{ratio:.4f}",
        '역할':       'TARGET' if col == TARGET else ('제거' if removed else '후보'),
    })
    if not removed:
        candidate_cols.append(col)

df_attr = pd.DataFrame(removal_rows)
print(f"\n{df_attr.to_string(index=False)}")
print(f"\n  → 후보 속성 ({len([c for c in candidate_cols if c != TARGET])}개): "
      f"{[c for c in candidate_cols if c != TARGET]}")

df_attr.to_csv(f"{RESULT_DIR}/attr_removal.csv", index=False, encoding="utf-8-sig")


# ────────────────────────────────────────────────────────────────────────────
# STEP 2 : 이산화 (의학적 기준 — association_rules.py 동일)
# ────────────────────────────────────────────────────────────────────────────
_step("STEP 2 — Discretization (의학적 기준)")

시도_map = {
    11: '서울', 26: '부산', 27: '대구', 28: '인천', 29: '광주',
    30: '대전', 31: '울산', 36: '세종', 41: '경기', 42: '강원',
    43: '충북', 44: '충남', 45: '전북', 46: '전남',
    47: '경북', 48: '경남', 49: '제주',
}

d = pd.DataFrame(index=df.index)
d['성별']         = df['성별코드'].map({1: '남', 2: '여'})
d['연령대']       = pd.cut(df['연령대코드(5세단위)'],
                          bins=[0, 4, 7, 10, 13, 99],
                          labels=['미성년(~19세)', '청년(20-34세)', '중년(35-49세)',
                                  '장년(50-64세)', '노년(65세+)'])
d['시도코드']     = df['시도코드'].map(시도_map)
d['BMI']          = pd.cut(df['BMI'],
                           bins=[0, 18.5, 22.9, 24.9, 999],
                           labels=['저체중', '정상', '과체중', '비만'])
d['수축기혈압']   = pd.cut(df['수축기혈압'],
                          bins=[0, 120, 140, 999],
                          labels=['정상', '주의', '위험'])
d['이완기혈압']   = pd.cut(df['이완기혈압'],
                          bins=[0, 80, 90, 999],
                          labels=['정상', '주의', '위험'])
d['혈색소']       = pd.cut(df['혈색소'],
                          bins=[0, 12, 16, 999],
                          labels=['낮음', '정상', '높음'])
d['식전혈당']     = pd.cut(df['식전혈당(공복혈당)'],
                          bins=[0, 100, 126, 999],
                          labels=['정상', '주의', '위험'])
d['혈청크레아티닌'] = pd.cut(df['혈청크레아티닌'],
                            bins=[0, 0.7, 1.2, 999],
                            labels=['낮음', '정상', '높음'])
d['흡연상태']     = df['흡연상태'].map({1: '비흡연', 2: '과거흡연', 3: '현재흡연'})
d['음주여부']     = df['음주여부'].map({0: '비음주', 1: '음주'})
d[TARGET]         = df[TARGET].map({1: '주의', 2: '위험'})

df_disc = d
print(f"  이산화 완료: {df_disc.shape}")
print(f"  당뇨 단계 분포: {df_disc[TARGET].value_counts().sort_index().to_dict()}")


# ────────────────────────────────────────────────────────────────────────────
# STEP 3 : Association Rule Mining (FP-Growth)
# ────────────────────────────────────────────────────────────────────────────
MIN_SUPPORT    = 0.05
MIN_CONFIDENCE = 0.55
MIN_LIFT       = 1.1
TOP_N          = 15

_step(f"STEP 3 — One-hot 인코딩")
with Spinner("One-hot 인코딩"):
    df_oh = pd.get_dummies(df_disc.astype(str), prefix_sep='=').astype(bool)
print(f"  One-hot 컬럼 수: {df_oh.shape[1]}개")

_step(f"FP-Growth  (min_support={MIN_SUPPORT})")
with Spinner("FP-Growth 빈발 항목집합 탐색"):
    frequent_items = fpgrowth(df_oh, min_support=MIN_SUPPORT, use_colnames=True)
    frequent_items = frequent_items.sort_values('support', ascending=False)
print(f"  빈발 항목집합: {len(frequent_items):,}개")

_step(f"연관 규칙 생성  (min_confidence={MIN_CONFIDENCE}, min_lift={MIN_LIFT})")
with Spinner("연관 규칙 생성"):
    rules = mlxtend_rules(frequent_items, metric="confidence",
                          min_threshold=MIN_CONFIDENCE)
    rules = rules[rules['lift'] >= MIN_LIFT].copy()
    rules = rules.sort_values('lift', ascending=False).reset_index(drop=True)
print(f"  최종 규칙 수: {len(rules):,}개")

rules['조건부(if)']   = rules['antecedents'].apply(lambda x: ' & '.join(sorted(x)))
rules['결론부(then)'] = rules['consequents'].apply(lambda x: ' & '.join(sorted(x)))

display_cols = ['조건부(if)', '결론부(then)', 'support', 'confidence', 'lift']
rules[display_cols].to_csv(f"{RESULT_DIR}/association_rules.csv",
                           index=False, encoding="utf-8-sig")

# 당뇨 관련 규칙 (결론부에 당뇨 포함)
rules_dia = rules[
    rules['결론부(then)'].str.contains(TARGET)
].reset_index(drop=True)

print(f"\n  [당뇨 관련 규칙 수: {len(rules_dia)}개]")
if len(rules_dia):
    print(rules_dia[display_cols].head(TOP_N).round(4).to_string(index=False))
    rules_dia[display_cols].to_csv(f"{RESULT_DIR}/association_rules_dia.csv",
                                   index=False, encoding="utf-8-sig")

# 바차트: 당뇨 관련 규칙 상위 TOP_N
if len(rules_dia):
    top_n  = rules_dia.head(TOP_N)
    labels = (top_n['조건부(if)'] + '  →  ' + top_n['결론부(then)'])
    colors = ['#E74C3C' if '위험' in c else '#E8A090'
              for c in top_n['결론부(then)']]

    fig, ax = plt.subplots(figsize=(12, max(6, TOP_N * 0.7)))
    bars = ax.barh(labels[::-1], top_n['lift'][::-1], color=colors[::-1], alpha=0.85)
    for bar, val in zip(bars, top_n['lift'][::-1]):
        ax.text(val + 0.01, bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}", va='center', fontsize=9)
    ax.set_xlabel("Lift (향상도)")
    ax.set_title(f"당뇨 관련 연관 규칙 상위 {TOP_N}개 (Lift 기준)\n"
                 f"진한색=위험, 연한색=주의", fontsize=12)
    plt.tight_layout()
    plt.savefig(f"{RESULT_DIR}/association_rules_dia_bar.png", dpi=150, bbox_inches='tight')
    plt.show()
    print(f"\n  저장: {RESULT_DIR}/association_rules_dia_bar.png")

print(f"\n완료: {RESULT_DIR}/")
