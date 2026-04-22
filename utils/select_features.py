"""
결과 컬럼에 가장 직접적인 영향을 주는 컬럼 탐색 유틸리티

3가지 방법(Pearson 상관계수, Mutual Information, RandomForest 중요도)으로
각각 순위를 매긴 뒤 평균 순위(앙상블)로 최종 상위 컬럼을 선정합니다.

사용법:
    python utils/select_features.py <csv경로> <결과컬럼> [--top N]

예시:
    python utils/select_features.py data/preprocessed/health_data_2024_preprocessed.csv 음주여부
    python utils/select_features.py data/preprocessed/health_data_2024_preprocessed.csv 성별코드 --top 3
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from sklearn.preprocessing import LabelEncoder


# ── 헬퍼 ────────────────────────────────────────────────────────────────────
def _section(title: str, width: int = 65):
    print(f"\n{'=' * width}")
    print(f"  {title}")
    print(f"{'=' * width}")


def _is_classification(y: pd.Series, threshold: int = 10) -> bool:
    return y.nunique() <= threshold or y.dtype == object


def _encode_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    le = LabelEncoder()
    for col in df.select_dtypes(include=["object", "category"]).columns:
        df[col] = le.fit_transform(df[col].astype(str))
    return df


# ── 핵심 분석 함수 ──────────────────────────────────────────────────────────
def select_features(path: str, target: str, top_n: int = 3):
    # 1. 데이터 로드
    if not os.path.exists(path):
        print(f"[오류] 파일 없음: {path}")
        sys.exit(1)

    df = pd.read_csv(path, encoding="utf-8-sig")
    if '이름' in df.columns:
        df = df.drop(columns=['이름'])

    if target not in df.columns:
        print(f"[오류] '{target}' 컬럼이 존재하지 않습니다.")
        print(f"  사용 가능한 컬럼: {list(df.columns)}")
        sys.exit(1)

    feature_cols = [c for c in df.columns if c != target]
    df_clean = df[feature_cols + [target]].dropna()

    X_raw = df_clean[feature_cols]
    y = df_clean[target]

    is_clf = _is_classification(y)
    task_type = "분류 (Classification)" if is_clf else "회귀 (Regression)"

    print(f"\n  파일   : {path}")
    print(f"  타깃   : {target}  (고유값 {y.nunique()}개 → {task_type})")
    print(f"  샘플 수: {len(df_clean):,}행  /  피처 수: {len(feature_cols)}개")

    X = _encode_features(X_raw)
    if is_clf:
        y_enc = LabelEncoder().fit_transform(y.astype(str))
    else:
        y_enc = y.values

    results = pd.DataFrame({"컬럼": feature_cols})

    # ── 방법 1: Pearson 상관계수 ─────────────────────────────────────────
    _section("방법 1 — Pearson 상관계수 (절댓값)")
    pearson = X.corrwith(pd.Series(y_enc, index=X.index)).abs()
    results["Pearson_절댓값"] = pearson.values
    results["Pearson_순위"] = results["Pearson_절댓값"].rank(ascending=False).astype(int)

    top_p = results.nsmallest(top_n, "Pearson_순위")[["컬럼", "Pearson_절댓값", "Pearson_순위"]]
    print(top_p.to_string(index=False))

    # ── 방법 2: Mutual Information ────────────────────────────────────────
    _section("방법 2 — Mutual Information")
    mi_fn = mutual_info_classif if is_clf else mutual_info_regression
    mi_scores = mi_fn(X, y_enc, random_state=42)
    results["MI_점수"] = mi_scores
    results["MI_순위"] = results["MI_점수"].rank(ascending=False).astype(int)

    top_m = results.nsmallest(top_n, "MI_순위")[["컬럼", "MI_점수", "MI_순위"]]
    print(top_m.to_string(index=False))

    # ── 방법 3: RandomForest 중요도 ───────────────────────────────────────
    _section("방법 3 — RandomForest 중요도")
    rf = (RandomForestClassifier if is_clf else RandomForestRegressor)(
        n_estimators=200, random_state=42, n_jobs=-1
    )
    rf.fit(X, y_enc)
    rf_imp = rf.feature_importances_
    results["RF_중요도"] = rf_imp
    results["RF_순위"] = results["RF_중요도"].rank(ascending=False).astype(int)

    top_r = results.nsmallest(top_n, "RF_순위")[["컬럼", "RF_중요도", "RF_순위"]]
    print(top_r.to_string(index=False))

    # ── 최종: 평균 순위 앙상블 ───────────────────────────────────────────
    _section(f"최종 결과 — 평균 순위 기준 상위 {top_n}개 컬럼")
    results["평균_순위"] = (
        results["Pearson_순위"] + results["MI_순위"] + results["RF_순위"]
    ) / 3
    results["최종_순위"] = results["평균_순위"].rank(method="min").astype(int)
    results = results.sort_values("최종_순위")

    display_cols = ["컬럼", "Pearson_절댓값", "Pearson_순위",
                    "MI_점수", "MI_순위", "RF_중요도", "RF_순위", "평균_순위", "최종_순위"]
    print(results[display_cols].head(top_n).round(4).to_string(index=False))

    top_features = results["컬럼"].head(top_n).tolist()
    print(f"\n  ★ 추천 컬럼 (상위 {top_n}개): {top_features}")

    print(f"\n{'=' * 65}\n")
    return top_features


# ── CLI 진입점 ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="결과 컬럼에 영향을 주는 피처 탐색"
    )
    parser.add_argument("path",   help="CSV 파일 경로")
    parser.add_argument("target", help="결과(타깃) 컬럼 이름")
    parser.add_argument("--top",  type=int, default=3,
                        help="추천할 상위 컬럼 수 (기본 3)")
    args = parser.parse_args()

    select_features(args.path, args.target, top_n=args.top)
