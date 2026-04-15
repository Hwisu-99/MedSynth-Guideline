import pandas as pd
import numpy as np

# ── 1. 데이터 로드 ──────────────────────────────────────────────
df = pd.read_csv(
    "data/raw/health_data_2024.csv",
    encoding="cp949"
)
print(f"원본 데이터: {df.shape[0]:,}행 × {df.shape[1]}열")

# ── 2. 컬럼 필터링: 필요한 11개 컬럼만 선택 ─────────────────────
keep_cols = [
    '성별코드', '연령대코드(5세단위)', '시도코드',
    '신장(5cm단위)', '체중(5kg단위)',
    '수축기혈압', '이완기혈압', '식전혈당(공복혈당)',
    '혈청크레아티닌', '흡연상태', '음주여부'
]
df = df[keep_cols]
print(f"컬럼 필터링 후: {df.shape[1]}개 컬럼")

# ── 3. 결측치 제거: 선택된 컬럼 중 NaN이 있는 행 전체 삭제 ───────
df = df.dropna()
print(f"결측치 제거 후: {df.shape[0]:,}행")

# ── 4. 샘플링: 랜덤하게 10,000행 추출 ───────────────────────────
df = df.sample(n=10_000, random_state=42).reset_index(drop=True)
print(f"샘플링 후: {df.shape[0]:,}행")

# ── 5. 데이터 타입 변환: 수치형 컬럼을 정수형으로 변환 ────────────
# 혈청크레아티닌은 소수점 데이터이므로 float 유지, 나머지는 int 변환
int_cols = [
    '성별코드', '연령대코드(5세단위)', '시도코드',
    '신장(5cm단위)', '체중(5kg단위)',
    '수축기혈압', '이완기혈압', '식전혈당(공복혈당)',
    '흡연상태', '음주여부'
]
df[int_cols] = df[int_cols].astype(int)
# 혈청크레아티닌은 float 유지
df['혈청크레아티닌'] = df['혈청크레아티닌'].astype(float)

# ── 6. 가상 컬럼 추가 ───────────────────────────────────────────
n = len(df)

# Name: '홍길동_0', '홍길동_1', ... 형태의 고유 식별자
df['Name'] = [f'홍길동_{i}' for i in range(n)]

# Phone: '010-XXXX-XXXX' 형태의 랜덤 휴대전화 번호
rng = np.random.default_rng(42)
df['Phone'] = [
    f"010-{rng.integers(1000, 9999)}-{rng.integers(1000, 9999)}"
    for _ in range(n)
]

# Address: 5개 지역 중 랜덤 선택
address_pool = [
    '서울시 강남구', '경기도 성남시', '부산시 해운대구',
    '인천시 연수구', '대전시 유성구'
]
df['Address'] = rng.choice(address_pool, size=n)

# Annual_Income: 평균 4500, 표준편차 1500의 정규분포, 범위 2500~15000, 10단위 절삭
income = rng.normal(loc=4500, scale=1500, size=n)
income = np.clip(income, 2500, 15000)       # 범위 제한
income = (income // 10 * 10).astype(int)    # 10단위 절삭 후 정수 변환
df['Annual_Income'] = income

# ── 7. 컬럼 순서 정렬 ───────────────────────────────────────────
final_cols = [
    'Name', 'Phone', 'Address',
    '성별코드', '연령대코드(5세단위)', '시도코드',
    '신장(5cm단위)', '체중(5kg단위)',
    '수축기혈압', '이완기혈압', '식전혈당(공복혈당)',
    '혈청크레아티닌', '흡연상태', '음주여부',
    'Annual_Income'
]
df = df[final_cols]

# ── 8. 결과 확인 ────────────────────────────────────────────────
print(f"\n최종 데이터: {df.shape[0]:,}행 × {df.shape[1]}열")
print(f"컬럼 목록: {df.columns.tolist()}")
print(f"\n데이터 타입:\n{df.dtypes.to_string()}")
print(f"\n샘플 데이터 (상위 3행):\n{df.head(3).to_string()}")

# ── 9. 저장 ─────────────────────────────────────────────────────
df.to_csv("data/preprocessed/health_data_2024_preprocessed.csv", index=False, encoding="utf-8-sig")
print("\n저장 완료: data/preprocessed/health_data_2024_preprocessed.csv")