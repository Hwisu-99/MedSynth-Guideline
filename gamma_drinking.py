import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# 한글 폰트 설정 (Windows)
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# ── 1. 원본 데이터 로드 및 샘플링 ───────────────────────────────
df = pd.read_csv("data/health_data_2024.csv", encoding="cp949",
                 usecols=['감마지티피', '음주여부'])
df = df.sample(n=10_000, random_state=42).reset_index(drop=True)

# ── 2. 필요 컬럼 선택 및 결측치 제거 ────────────────────────────
df = df[['감마지티피', '음주여부']].dropna()
df['음주여부'] = df['음주여부'].astype(int)

# 음주여부 레이블 매핑 (1: 음주, 0: 비음주)
df['음주여부_label'] = df['음주여부'].map({1: '음주', 0: '비음주'})

# ── 3. 기술통계 출력 ────────────────────────────────────────────
print("=" * 50)
print("[기술통계] 음주 여부별 감마지티피")
print("=" * 50)
print(df.groupby('음주여부_label')['감마지티피'].describe().round(2).to_string())

# ── 4. 통계 검정 (Mann-Whitney U) ───────────────────────────────
# 감마지티피는 정규분포를 따르지 않는 경우가 많아 비모수 검정 사용
group_drink    = df.loc[df['음주여부'] == 1, '감마지티피']
group_nondrink = df.loc[df['음주여부'] == 0, '감마지티피']

stat, p_value = stats.mannwhitneyu(group_drink, group_nondrink, alternative='two-sided')
print(f"\n[Mann-Whitney U 검정]")
print(f"  U 통계량: {stat:.1f}")
print(f"  p-value : {p_value:.4e}")
print(f"  결론    : {'통계적으로 유의미한 차이 있음 (p < 0.05)' if p_value < 0.05 else '유의미한 차이 없음'}")

# ── 5. 시각화 (3개 subplot) ──────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 6))
fig.suptitle("감마지티피와 음주여부의 관계 (n=10,000)", fontsize=14, y=1.02)

palette = {'음주': '#E74C3C', '비음주': '#3498DB'}
order   = ['비음주', '음주']

# (1) 박스플롯
sns.boxplot(
    data=df, x='음주여부_label', y='감마지티피',
    order=order, palette=palette,
    width=0.5, flierprops=dict(marker='o', markersize=2, alpha=0.3),
    ax=axes[0]
)
axes[0].set_title("Box Plot")
axes[0].set_xlabel("음주여부")
axes[0].set_ylabel("감마지티피 (IU/L)")

# (2) 바이올린 플롯
sns.violinplot(
    data=df, x='음주여부_label', y='감마지티피',
    order=order, palette=palette,
    inner='quartile', cut=0,
    ax=axes[1]
)
axes[1].set_title("Violin Plot")
axes[1].set_xlabel("음주여부")
axes[1].set_ylabel("감마지티피 (IU/L)")

# (3) 평균 ± 95% CI 막대그래프
means = df.groupby('음주여부_label')['감마지티피'].mean()
sems  = df.groupby('음주여부_label')['감마지티피'].sem() * 1.96  # 95% CI
bar_colors = [palette[k] for k in order]

axes[2].bar(
    order, [means[k] for k in order],
    yerr=[sems[k] for k in order],
    color=bar_colors, width=0.5,
    capsize=8, error_kw=dict(elinewidth=1.5)
)
axes[2].set_title("평균 ± 95% CI")
axes[2].set_xlabel("음주여부")
axes[2].set_ylabel("감마지티피 평균 (IU/L)")

# p-value 표시
y_max = max(means[k] + sems[k] for k in order)
axes[2].annotate(
    f"p = {p_value:.4e}",
    xy=(0.5, y_max * 1.05),
    xycoords=('axes fraction', 'data'),
    ha='center', fontsize=8, color='black'
)

plt.tight_layout()
plt.savefig("result/gamma_drinking.png", dpi=150, bbox_inches='tight')
print("\n저장 완료: result/gamma_drinking.png")
plt.show()
