import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 한글 폰트 설정 (Windows)
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# ── 1. 원본 데이터 로드 및 샘플링 ───────────────────────────────
df = pd.read_csv("data/health_data_2024.csv", encoding="cp949")
df = df.sample(n=10_000, random_state=42).reset_index(drop=True)
print(f"샘플링 완료: {df.shape[0]:,}행")

# ── 2. 분석 대상 컬럼 선택 및 결측치 제거 ──────────────────────
target_cols = [
    '신장(5cm단위)', '체중(5kg단위)',
    '수축기혈압', '이완기혈압',
    '혈청지오티(AST)', '혈청지피티(ALT)',
]
df = df[target_cols].dropna()

# ── 3. 그룹별 변수 정의 ─────────────────────────────────────────
groups = [
    {
        'cols':  ['수축기혈압', '이완기혈압'],
        'title': '혈압 변수 간 상관관계',
    },
    {
        'cols':  ['혈청지오티(AST)', '혈청지피티(ALT)'],
        'title': '간 효소 변수 간 상관관계',
    },
    {
        'cols':  ['신장(5cm단위)', '체중(5kg단위)'],
        'title': '신체계측 변수 간 상관관계',
    },
]

# ── 4. 상관계수 계산 및 출력 ────────────────────────────────────
for g in groups:
    print(f"\n[{g['title']}]")
    print(df[g['cols']].corr().round(3).to_string())

# ── 5. 그룹별 히트맵 개별 출력 ─────────────────────────────────
file_names = [
    'result/corr_blood_pressure.png',
    'result/corr_liver.png',
    'result/corr_body.png',
]

for g, fname in zip(groups, file_names):
    corr = df[g['cols']].corr()
    n = len(g['cols'])

    fig, ax = plt.subplots(figsize=(2.5 + n * 1.2, 2 + n * 1.2))
    sns.heatmap(
        corr,
        annot=True,
        fmt=".2f",
        cmap="RdYlBu_r",
        vmin=-1, vmax=1,
        center=0,
        square=True,
        linewidths=0.5,
        ax=ax,
        annot_kws={"size": 12},
    )
    ax.set_title(f"{g['title']}\n(n=10,000)", fontsize=13, pad=12)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha='right', fontsize=10)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=10)

    plt.tight_layout()
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    print(f"저장 완료: {fname}")
    plt.show()
