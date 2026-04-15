# ============================================================
# py-synthpop을 활용한 건강 데이터 합성 데이터 생성
# R의 synthpop 패키지를 파이썬으로 구현한 라이브러리 사용
# 참고: https://r-love-view.tistory.com/14
# ============================================================
# 최초 1회 설치:
#   pip install py-synthpop
# ============================================================

import pandas as pd
from synthpop import Synthpop

# ============================================================
# 1. 데이터 불러오기
# ============================================================
data_path = "data/preprocessed/health_data_2024_preprocessed_1000.csv"
df = pd.read_csv(data_path, encoding="utf-8-sig")

print(f"원본 데이터 크기: {df.shape[0]}행 x {df.shape[1]}열")
print(f"컬럼 목록: {df.columns.tolist()}\n")

# ============================================================
# 2. 개인식별정보(PII) 컬럼 제외
#    - Name, Phone, Address는 합성 대상에서 제외
# ============================================================
df_synth = df.drop(columns=["Name", "Phone", "Address"])

print("PII 제외 후 컬럼 목록:")
print(df_synth.columns.tolist())
print()

# ============================================================
# 3. 범주형 변수 타입 지정
#    - py-synthpop은 int / float / category 3가지 타입을 사용
#    - 범주형으로 처리할 변수는 미리 category로 변환
# ============================================================
category_cols = ["성별코드", "연령대코드(5세단위)", "시도코드", "흡연상태", "음주여부"]

for col in category_cols:
    df_synth[col] = df_synth[col].astype("category")

print("변수 타입 확인:")
print(df_synth.dtypes)
print()

# ============================================================
# 4. py-synthpop용 메타데이터(dtypes 딕셔너리) 생성
#    - pandas dtype을 py-synthpop 형식으로 변환
#    - int64 → "int", float64 → "float", category → "category"
# ============================================================
def convert_dtypes(df):
    """pandas dtype을 py-synthpop이 요구하는 타입 문자열로 변환"""
    type_mapping = {
        "int64":    "int",
        "float64":  "float",
        "category": "category"
    }
    return {col: type_mapping.get(str(dtype), "float")
            for col, dtype in df.dtypes.items()}

dtypes = convert_dtypes(df_synth)

print("py-synthpop 메타데이터(dtypes):")
for col, dtype in dtypes.items():
    print(f"  {col}: {dtype}")
print()

# ============================================================
# 5. Synthpop 모델 학습 (fit)
#    - CART(Classification And Regression Trees) 알고리즘 기반
#    - 실제 데이터의 분포와 변수 간 관계를 학습
# ============================================================
print("Synthpop 모델 학습 중...")
spop = Synthpop()
spop.fit(df_synth, dtypes)
print("학습 완료!\n")

# ============================================================
# 6. 합성 데이터 생성 (generate)
#    - 원본 데이터와 동일한 1000개 행 생성
#    - py-synthpop은 원본 이상의 데이터도 생성 가능
# ============================================================
n_generate = 1000  # 생성할 합성 데이터 행 수

print(f"합성 데이터 {n_generate}개 생성 중...")
df_syn = spop.generate(n_generate)
print(f"생성 완료! 합성 데이터 크기: {df_syn.shape}\n")

# ============================================================
# 7. 원본 vs 합성 데이터 기초 통계 비교
# ============================================================
print("=" * 60)
print("[원본 데이터 기초통계]")
print(df_synth.describe())

print("\n[합성 데이터 기초통계]")
print(df_syn.describe())

# 범주형 변수 분포 비교
print("\n[범주형 변수 분포 비교]")
for col in category_cols:
    print(f"\n  {col}:")
    orig_ratio = df_synth[col].value_counts(normalize=True).sort_index()
    syn_ratio  = df_syn[col].value_counts(normalize=True).sort_index()
    comparison = pd.DataFrame({
        "원본 비율(%)":  (orig_ratio * 100).round(1),
        "합성 비율(%)":  (syn_ratio  * 100).round(1)
    })
    print(comparison.to_string())

# ============================================================
# 8. 합성 데이터 저장
# ============================================================
output_path = "data/synthetic/synthpop/health_data_synthetic_synthpop.csv"
df_syn.to_csv(output_path, index=False, encoding="utf-8-sig")

print(f"\n합성 데이터 저장 완료: {output_path}")
print(f"저장된 행 수: {len(df_syn)}")
