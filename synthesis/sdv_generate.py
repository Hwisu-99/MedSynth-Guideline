import os
import pandas as pd
from sdv.metadata import SingleTableMetadata
from sdv.single_table import CTGANSynthesizer
from utils.measure import MeasureResource

os.makedirs("data/synthetic/sdv", exist_ok=True)

# ── 1. 데이터 로드 ─────────────────────────────────────────────
df = pd.read_csv("data/preprocessed/health_data_2024_preprocessed.csv",
                 encoding="utf-8-sig")
print(f"원본 데이터: {df.shape[0]:,}행 × {df.shape[1]}열")

# ── 2. PII 컬럼 제외 ────────────────────────────────────────────
df_synth = df.drop(columns=['이름'])
print(f"PII 제외 후: {df_synth.shape[1]}개 컬럼")
print(f"컬럼 목록: {df_synth.columns.tolist()}\n")

# ── 3. SDV 메타데이터 정의 ─────────────────────────────────────
metadata = SingleTableMetadata()
metadata.detect_from_dataframe(df_synth)

category_cols = ['성별코드', '연령대코드(5세단위)', '시도코드', '흡연상태', '음주여부']
for col in category_cols:
    metadata.update_column(col, sdtype='categorical')

print("[메타데이터 타입]")
for col, info in metadata.columns.items():
    print(f"  {col}: {info['sdtype']}")

# ── 4. SDV(CTGAN) 학습 ─────────────────────────────────────────
synthesizer = CTGANSynthesizer(metadata, epochs=300, verbose=True)

print("\nSDV(CTGAN) 학습 중... (epochs=300, 수 분 소요)")
with MeasureResource("SDV 학습"):
    synthesizer.fit(df_synth)

# ── 5. 합성 데이터 생성 ─────────────────────────────────────────
n_generate = len(df_synth)
print(f"\n합성 데이터 {n_generate:,}개 생성 중...")
with MeasureResource("SDV 생성", n_rows=n_generate):
    df_syn = synthesizer.sample(num_rows=n_generate)
print(f"생성 완료: {df_syn.shape}")

# ── 6. 기초 통계 비교 ───────────────────────────────────────────
print("\n[원본 기초통계]")
print(df_synth.describe().round(2).to_string())
print("\n[합성 기초통계]")
print(df_syn.describe().round(2).to_string())

# ── 7. 저장 ─────────────────────────────────────────────────────
output_csv = "data/synthetic/sdv/health_data_synthetic_sdv.csv"
output_pkl = "data/synthetic/sdv/sdv_model.pkl"

df_syn.to_csv(output_csv, index=False, encoding="utf-8-sig")
synthesizer.save(output_pkl)

print(f"\n저장 완료:")
print(f"  {output_csv}")
print(f"  {output_pkl}")
