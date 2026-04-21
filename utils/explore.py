"""
CSV 파일 기초 탐색 유틸리티

사용법:
    python utils/explore.py data/preprocessed/health_data_2024_preprocessed.csv
    python utils/explore.py data/synthetic/synthpop/health_data_synthetic_synthpop.csv --top 10
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd


# ── 출력 구분선 헬퍼 ────────────────────────────────────────────────────────
def _section(title: str, width: int = 65):
    print(f"\n{'=' * width}")
    print(f"  {title}")
    print(f"{'=' * width}")


# ── 메인 탐색 함수 ─────────────────────────────────────────────────────────
def explore(path: str, top_n: int = 5, sample_rows: int = 5):
    if not os.path.exists(path):
        print(f"[오류] 파일 없음: {path}")
        sys.exit(1)

    print(f"\n파일 경로: {path}")
    df = pd.read_csv(path, encoding="utf-8-sig")

    # ── 1. 기본 정보 ───────────────────────────────────────────────────────
    _section("1. 기본 정보")
    file_size_mb = os.path.getsize(path) / 1024 / 1024
    mem_usage_mb = df.memory_usage(deep=True).sum() / 1024 / 1024
    print(f"  총 레코드 수   : {len(df):,} 행")
    print(f"  총 컬럼 수     : {len(df.columns)} 개")
    print(f"  파일 크기      : {file_size_mb:.2f} MB")
    print(f"  메모리 사용량  : {mem_usage_mb:.2f} MB")
    print(f"  중복 행 수     : {df.duplicated().sum():,} 행")

    # ── 2. 컬럼 목록 & 타입 ───────────────────────────────────────────────
    _section("2. 컬럼 목록 및 데이터 타입")
    dtype_df = pd.DataFrame({
        "컬럼명": df.columns,
        "dtype": df.dtypes.values,
        "결측값": df.isnull().sum().values,
        "결측률(%)": (df.isnull().mean() * 100).round(2).values,
        "고유값 수": df.nunique().values,
    })
    print(dtype_df.to_string(index=False))

    # ── 3. 수치형 컬럼 기술통계 ───────────────────────────────────────────
    num_cols = df.select_dtypes(include="number").columns.tolist()
    if num_cols:
        _section("3. 수치형 컬럼 기술통계")
        desc = df[num_cols].describe(percentiles=[0.25, 0.5, 0.75]).T
        desc["범위"] = desc["max"] - desc["min"]
        desc["왜도"] = df[num_cols].skew().round(4)
        desc["첨도"] = df[num_cols].kurt().round(4)
        display_cols = ["count", "mean", "std", "min", "25%", "50%", "75%", "max", "범위", "왜도", "첨도"]
        print(desc[display_cols].round(4).to_string())

    # ── 4. 범주형 컬럼 빈도 분석 ──────────────────────────────────────────
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    if cat_cols:
        _section("4. 범주형 컬럼 빈도 분석")
        for col in cat_cols:
            vc = df[col].value_counts(dropna=False).head(top_n)
            print(f"\n  [{col}]  (고유값 {df[col].nunique()}개, 상위 {top_n}개)")
            for val, cnt in vc.items():
                pct = cnt / len(df) * 100
                print(f"    {str(val):<20} {cnt:>7,}  ({pct:5.1f}%)")

    # ── 5. 수치형 컬럼 이상치 탐지 (IQR 방식) ────────────────────────────
    if num_cols:
        _section("5. 이상치 탐지 (IQR × 1.5 기준)")
        outlier_rows = []
        for col in num_cols:
            q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
            iqr = q3 - q1
            lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
            n_out = ((df[col] < lower) | (df[col] > upper)).sum()
            outlier_rows.append({
                "컬럼": col,
                "하한": round(lower, 4),
                "상한": round(upper, 4),
                "이상치 수": n_out,
                "이상치율(%)": round(n_out / len(df) * 100, 2),
            })
        print(pd.DataFrame(outlier_rows).to_string(index=False))

    # ── 6. 수치형 컬럼 상관관계 (절댓값 상위 쌍) ─────────────────────────
    if len(num_cols) >= 2:
        _section(f"6. 수치형 컬럼 상관관계 (절댓값 상위 {top_n}쌍)")
        corr = df[num_cols].corr().abs()
        pairs = (
            corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
            .stack()
            .reset_index()
        )
        pairs.columns = ["컬럼1", "컬럼2", "|상관계수|"]
        pairs = pairs.sort_values("|상관계수|", ascending=False).head(top_n)
        print(pairs.to_string(index=False))

    # ── 7. 샘플 미리보기 ──────────────────────────────────────────────────
    _section(f"7. 샘플 데이터 ({sample_rows}행)")
    print(df.head(sample_rows).to_string())

    print(f"\n{'=' * 65}")
    print("  탐색 완료")
    print(f"{'=' * 65}\n")


# ── CLI 진입점 ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CSV 파일 기초 탐색")
    parser.add_argument("path", help="CSV 파일 경로")
    parser.add_argument("--top", type=int, default=5,
                        help="범주형 빈도 / 상관관계 상위 N개 (기본 5)")
    parser.add_argument("--sample", type=int, default=5,
                        help="샘플 미리보기 행 수 (기본 5)")
    args = parser.parse_args()

    explore(args.path, top_n=args.top, sample_rows=args.sample)
