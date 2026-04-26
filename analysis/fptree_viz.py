"""
FP-Tree 구조 시각화 (발표용)

실행:
    python analysis/fptree_viz.py
"""

import os
import warnings
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.font_manager as fm
import networkx as nx

from datetime import datetime

warnings.filterwarnings("ignore")

_KOREAN_FONTS = ['Malgun Gothic', 'NanumGothic', 'AppleGothic', 'Noto Sans KR']
_registered   = {f.name for f in fm.fontManager.ttflist}
_korean_font  = next((f for f in _KOREAN_FONTS if f in _registered), None)
if _korean_font:
    plt.rcParams['font.family'] = _korean_font
plt.rcParams['axes.unicode_minus'] = False

RESULT_DIR = f"result/analysis/{datetime.now().strftime('%Y%m%d_%H%M%S')}_fptree"
os.makedirs(RESULT_DIR, exist_ok=True)
print(f"결과 저장 경로: {RESULT_DIR}\n")

REAL_PATH       = "data/preprocessed/health_data_2024_preprocessed.csv"
TARGETS         = ['당뇨', '고혈압', '간기능']
TOP_N           = 8      # 트리 아이템 수 (적을수록 단순한 트리)
MIN_SUP         = 0.30   # 아이템 선택 최소 지지도
SAMPLE_N        = 3000   # 사용할 행 수
MIN_NODE_COUNT  = 30     # 이 미만 count 노드는 트리에서 제거 (가지치기)


# ── 1. 전처리 ─────────────────────────────────────────────────────────────────
print("[1] 데이터 로드 및 이산화")
df = pd.read_csv(REAL_PATH, encoding="utf-8-sig")
if '이름' in df.columns:
    df = df.drop(columns=['이름'])
df = df.sample(n=min(SAMPLE_N, len(df)), random_state=42)

d = pd.DataFrame(index=df.index)
d['성별']       = df['성별코드'].map({1: '남', 2: '여'})
d['연령대']     = pd.cut(df['연령대코드(5세단위)'],
                        bins=[0, 4, 7, 10, 13, 99],
                        labels=['미성년', '청년', '중년', '장년', '노년'])
d['BMI']        = pd.cut(df['BMI'], bins=[0, 18.5, 22.9, 24.9, 999],
                         labels=['저체중', '정상', '과체중', '비만'])
d['수축기혈압'] = pd.cut(df['수축기혈압'], bins=[0, 120, 140, 999],
                        labels=['혈압정상', '혈압주의', '혈압위험'])
d['이완기혈압'] = pd.cut(df['이완기혈압'], bins=[0, 80, 90, 999],
                        labels=['이혈압정상', '이혈압주의', '이혈압위험'])
d['혈색소']     = pd.cut(df['혈색소'], bins=[0, 12, 16, 999],
                        labels=['혈색소낮음', '혈색소정상', '혈색소높음'])
d['식전혈당']   = pd.cut(df['식전혈당(공복혈당)'], bins=[0, 100, 126, 999],
                        labels=['혈당정상', '혈당주의', '혈당위험'])
d['혈청크레아티닌'] = pd.cut(df['혈청크레아티닌'], bins=[0, 0.7, 1.2, 999],
                           labels=['크레아낮음', '크레아정상', '크레아높음'])
d['AST']        = pd.cut(df['혈청지오티(AST)'], bins=[0, 40, 60, 999],
                         labels=['AST정상', 'AST주의', 'AST위험'])
d['ALT']        = pd.cut(df['혈청지피티(ALT)'], bins=[0, 40, 60, 999],
                         labels=['ALT정상', 'ALT주의', 'ALT위험'])
d['감마지티피'] = pd.cut(df['감마지티피'], bins=[0, 35, 63, 999],
                        labels=['GGT정상', 'GGT주의', 'GGT위험'])
d['흡연상태']   = df['흡연상태'].map({1: '비흡연', 2: '과거흡연', 3: '현재흡연'})
d['음주여부']   = df['음주여부'].map({0: '비음주', 1: '음주'})
for col in TARGETS:
    d[col] = df[col].map({0: f'{col}정상', 1: f'{col}주의', 2: f'{col}위험'})

df_oh = pd.get_dummies(d.astype(str))
item_freq = df_oh.sum().sort_values(ascending=False)

# 상위 N개 빈발 아이템 선택
top_items = item_freq[item_freq / len(df_oh) >= MIN_SUP].head(TOP_N).index.tolist()
print(f"  선택 아이템 {len(top_items)}개: {top_items}")
item_rank = {item: i for i, item in enumerate(top_items)}

# 트랜잭션 구성 (빈도 내림차순 정렬)
transactions = []
for _, row in df_oh.iterrows():
    t = sorted([item for item in top_items if row.get(item, False)],
               key=lambda x: item_rank[x])
    if t:
        transactions.append(t)
print(f"  트랜잭션: {len(transactions):,}개")


# ── 2. FP-Tree 구축 ───────────────────────────────────────────────────────────
print("\n[2] FP-Tree 구축")

class FPNode:
    _id = 0
    def __init__(self, name, parent=None):
        FPNode._id += 1
        self.nid      = FPNode._id
        self.name     = name
        self.count    = 0
        self.parent   = parent
        self.children = {}

root = FPNode('ROOT')
for t in transactions:
    cur = root
    for item in t:
        if item not in cur.children:
            cur.children[item] = FPNode(item, cur)
        cur.children[item].count += 1
        cur = cur.children[item]
print("  완료")


# ── 3. 가지치기 + networkx 변환 ───────────────────────────────────────────────
print("\n[3] 가지치기 및 그래프 변환")

G          = nx.DiGraph()
meta       = {}   # nid → {name, count}

def add_nodes(node):
    G.add_node(node.nid)
    meta[node.nid] = {'name': node.name, 'count': node.count}
    for child in node.children.values():
        if child.count >= MIN_NODE_COUNT:   # 가지치기
            add_nodes(child)
            G.add_edge(node.nid, child.nid)

add_nodes(root)
meta[root.nid]['count'] = len(transactions)
print(f"  가지치기 후 노드 수: {G.number_of_nodes()}")


# ── 4. 재귀적 트리 레이아웃 ──────────────────────────────────────────────────
def subtree_width(node_id):
    children = list(G.successors(node_id))
    if not children:
        return 1
    return sum(subtree_width(c) for c in children)

def assign_pos(node_id, x_left, depth, vert_gap=1.0):
    w = subtree_width(node_id)
    x = x_left + w / 2
    pos[node_id] = (x, -depth * vert_gap)
    children = list(G.successors(node_id))
    cx = x_left
    for child in children:
        cw = subtree_width(child)
        assign_pos(child, cx, depth + 1, vert_gap)
        cx += cw

pos = {}
assign_pos(root.nid, 0, 0)


# ── 5. 색상 / 크기 ────────────────────────────────────────────────────────────
def node_color(name):
    if name == 'ROOT':
        return '#4A4A4A'
    if '주의' in name or '위험' in name:
        return '#E74C3C'
    if any(t in name for t in TARGETS):
        return '#E67E22'
    return '#2980B9'

max_cnt   = max(meta[n]['count'] for n in G.nodes())
node_list = list(G.nodes())
colors    = [node_color(meta[n]['name']) for n in node_list]
sizes     = [max(2000, meta[n]['count'] / max_cnt * 5000) for n in node_list]
labels    = {n: f"{meta[n]['name']}\n({meta[n]['count']})" for n in node_list}
# 노드 크기 → 폰트 크기 매핑 (작은 노드일수록 작은 글씨)
MIN_FONT, MAX_FONT = 7, 11
font_sizes = {
    n: MIN_FONT + (MAX_FONT - MIN_FONT) * (sz - 2000) / max(1, 5000 - 2000)
    for n, sz in zip(node_list, sizes)
}


# ── 6. 시각화 ─────────────────────────────────────────────────────────────────
print("\n[4] 시각화")

fig = plt.figure(figsize=(24, 14))
fig.suptitle(
    f"FP-Tree 구조 시각화\n"
    f"상위 {TOP_N}개 빈발 아이템 | min_support={MIN_SUP} | "
    f"샘플={SAMPLE_N:,}행 | 가지치기 기준 count≥{MIN_NODE_COUNT}",
    fontsize=14, fontweight='bold'
)

# 좌측(3/4): FP-Tree
ax1 = fig.add_axes([0.01, 0.05, 0.68, 0.88])
nx.draw_networkx_edges(G, pos, ax=ax1, arrows=True,
                       arrowstyle='->', arrowsize=12,
                       edge_color='#BBBBBB', width=1.5, alpha=0.8,
                       connectionstyle='arc3,rad=0.0')
nx.draw_networkx_nodes(G, pos, ax=ax1, nodelist=node_list,
                       node_color=colors, node_size=sizes, alpha=0.92)
for n in node_list:
    x, y = pos[n]
    ax1.text(x, y, labels[n],
             ha='center', va='center',
             fontsize=font_sizes[n], fontweight='bold', color='white',
             fontfamily=_korean_font or 'sans-serif',
             bbox=dict(boxstyle='round,pad=0.15', facecolor='none',
                       edgecolor='none', alpha=0))

legend_items = [
    mpatches.Patch(color='#4A4A4A', label='ROOT'),
    mpatches.Patch(color='#E74C3C', label='주의 / 위험'),
    mpatches.Patch(color='#E67E22', label='질환 정상'),
    mpatches.Patch(color='#2980B9', label='비질환 아이템'),
]
ax1.legend(handles=legend_items, loc='lower left', fontsize=10,
           framealpha=0.9)
ax1.set_title("노드 크기 = count  |  위→아래 = 트랜잭션 삽입 순서", fontsize=11)
ax1.axis('off')

# 우측(1/4): 지지도 막대그래프
ax2 = fig.add_axes([0.72, 0.10, 0.26, 0.80])
sup_vals = item_freq[top_items] / len(df_oh)
bar_colors = [node_color(i) for i in top_items]
bars = ax2.barh(range(len(top_items)), sup_vals.values,
                color=bar_colors, alpha=0.88)
ax2.set_yticks(range(len(top_items)))
ax2.set_yticklabels(top_items, fontsize=10)
ax2.invert_yaxis()
ax2.axvline(MIN_SUP, color='red', linestyle='--', linewidth=1.2,
            label=f'min_support={MIN_SUP}')
ax2.set_xlabel("Support (지지도)", fontsize=11)
ax2.set_title("빈발 아이템 지지도\n(트리 삽입 순서 ↓)", fontsize=11)
ax2.legend(fontsize=9)
for bar, val in zip(bars, sup_vals.values):
    ax2.text(val + 0.005, bar.get_y() + bar.get_height() / 2,
             f"{val:.3f}", va='center', fontsize=9)

out = f"{RESULT_DIR}/fptree.png"
plt.savefig(out, dpi=150, bbox_inches='tight')
plt.show()
print(f"저장: {out}")
print(f"\n완료: {RESULT_DIR}/")
