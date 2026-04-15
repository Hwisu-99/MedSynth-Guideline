import pandas as pd
import numpy as np

# ── 1. 데이터 로드 ──────────────────────────────────────────────
df = pd.read_csv(
    "data/raw/health_data_2024.csv",
    encoding="cp949"
)
print(f"원본 데이터: {df.shape[0]:,}행 × {df.shape[1]}열")
print(f"원본 컬럼 목록:\n{df.columns.tolist()}")

# ── 2. 컬럼 필터링: 필요한 컬럼만 선택 ─────────────────────
keep_cols = [
    '성별코드', '연령대코드(5세단위)', '시도코드',
    '신장(5cm단위)', '체중(5kg단위)',
    '수축기혈압', '이완기혈압', '혈색소',
    '식전혈당(공복혈당)', '혈청크레아티닌',
    '혈청지오티(AST)', '혈청지피티(ALT)', '감마지티피',
    '흡연상태', '음주여부'
]
df = df[keep_cols]
print(f"컬럼 필터링 후: {df.shape[1]}개 컬럼")

# ── 3. 결측치 제거: 선택된 컬럼 중 NaN이 있는 행 전체 삭제 ───────
df = df.dropna()
print(f"결측치 제거 후: {df.shape[0]:,}행")

# ── 4. 샘플링: 랜덤하게 10,000행 추출 ───────────────────────────
df = df.sample(n=10_000, random_state=42).reset_index(drop=True)
print(f"샘플링 후: {df.shape[0]:,}행")

# ── 5. 데이터 타입 변환 ────────────────────────────────────────────
int_cols = [
    '성별코드', '연령대코드(5세단위)', '시도코드',
    '신장(5cm단위)', '체중(5kg단위)',
    '수축기혈압', '이완기혈압', '식전혈당(공복혈당)',
    '혈청지오티(AST)', '혈청지피티(ALT)', '감마지티피',
    '흡연상태', '음주여부'
]
df[int_cols] = df[int_cols].astype(int)
# 소수점 컬럼 float 유지
df['혈청크레아티닌'] = df['혈청크레아티닌'].astype(float)
df['혈색소'] = df['혈색소'].astype(float)

# BMI 계산: 체중(kg) / (신장(m))²
df['BMI'] = (df['체중(5kg단위)'] / (df['신장(5cm단위)'] / 100) ** 2).round(1)

# ── 6. 가상 컬럼 추가 ───────────────────────────────────────────
n = len(df)
rng = np.random.default_rng(42)

# Name: 랜덤 3글자 한글 이름
last_names = ['김', '이', '박', '최', '정', '강', '조', '윤', '장', '임',
              '한', '오', '서', '신', '권', '황', '안', '송', '류', '전']
mid_chars  = ['민', '서', '지', '현', '준', '승', '예', '도', '수', '하',
              '재', '태', '영', '진', '유', '은', '성', '동', '경', '나']
end_chars  = ['준', '아', '린', '우', '호', '연', '원', '빈', '율', '희',
              '석', '훈', '혁', '진', '영', '민', '현', '솔', '찬', '아']
last_idx = rng.integers(0, len(last_names), size=n)
mid_idx  = rng.integers(0, len(mid_chars),  size=n)
end_idx  = rng.integers(0, len(end_chars),  size=n)
df['이름'] = [last_names[l] + mid_chars[m] + end_chars[e]
              for l, m, e in zip(last_idx, mid_idx, end_idx)]

# ── 7. 컬럼 순서 정렬 ───────────────────────────────────────────
final_cols = [
    '이름', '성별코드', '시도코드', '연령대코드(5세단위)',
    '신장(5cm단위)', '체중(5kg단위)', 'BMI',
    '수축기혈압', '이완기혈압', '혈색소',
    '식전혈당(공복혈당)', '혈청크레아티닌',
    '혈청지오티(AST)', '혈청지피티(ALT)', '감마지티피',
    '흡연상태', '음주여부'
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