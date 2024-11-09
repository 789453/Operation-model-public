import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from scipy.optimize import linprog


# streamlit run "Operational research1.py"
# 用户通过Streamlit输入


st.title('工厂-中转-商场——两阶段运筹优化模型 1.0版本')
st.markdown(f"<h3 style='color: blue;font-size: 20px;'>制作来源：杨博远    引用请说明，谢谢</h3>", unsafe_allow_html=True)

# 简单的操作说明
st.markdown("""
### 操作说明
1. 使用页面左侧的输入框和滑块来设置工厂、中转站、商店的数量以及各种约束条件。
2. 调整运输成本和定价，查看不同条件下的优化结果。
3. 优化结果以总成本形式显示，并通过桑基图展示工厂到中转站再到商店的货物流动。
""")

# 输入工厂、中转站和商店的数量
m = st.number_input('工厂数量', min_value=1, max_value=50, value=6)
n = st.number_input('中转站数量', min_value=1, max_value=50, value=8)
k = st.number_input('商店数量', min_value=1, max_value=50, value=12)
w = st.slider('总商品数量(此功能暂时不能自动调节，请调节侧栏下方的商品最小需求）', min_value=100, max_value=5000, value=1000)

# 侧边栏用于调整运输成本
st.sidebar.header("调整运输成本和定价")
# 在侧边栏中添加详细指导按钮
with st.sidebar.expander("详细指导", expanded=True):
    st.markdown("""
    ### 操作说明
    1. 使用页面左侧的输入框和滑块来设置工厂、中转站、商店的数量以及各种约束条件。
    2. 调整运输成本和定价，查看不同条件下的优化结果。
    3. 优化结果以总成本形式显示，并通过桑基图展示工厂到中转站再到商店的货物流动。
    """)


q1_min = st.sidebar.slider('工厂到中转站成本最小值', 1, 50, 3)
q1_max = st.sidebar.slider('工厂到中转站成本最大值', 10, 100, 20)
q2_min = st.sidebar.slider('中转站到商店成本最小值', 5, 50, 10)
q2_max = st.sidebar.slider('中转站到商店成本最大值', 20, 150, 50)
q3_min = st.sidebar.slider('中转站到中转站成本最小值', 1, 30, 5)
q3_max = st.sidebar.slider('中转站到中转站成本最大值', 5, 80, 20)




st.sidebar.header("额外约束选项")

# 1. 合作工厂之间的竞争关系
with st.sidebar.expander("合作工厂之间的竞争关系", expanded=False):
    enable_competition = st.checkbox("启用合作工厂之间的竞争关系")
    if enable_competition:
        factory_competition = st.slider("选择工厂之间的竞争程度", 0, 100, 50)

# 2. 某些中转站的流量有限
with st.sidebar.expander("中转站的流量限制", expanded=False):
    enable_flow_limit = st.checkbox("启用中转站流量限制")
    if enable_flow_limit:
        flow_limit = st.slider("选择中转站的最大流量", 0, 10000, 10000)
        st.markdown(f"<h3 style='color: blue;font-size: 17px;'>该部分代码尚未成功编写，所以优化失败（敬请期待2.0版本的修正</h3>",
                    unsafe_allow_html=True)

# 3. 每个工厂和中转站的合作
with st.sidebar.expander("工厂和中转站之间的合作关系", expanded=False):
    enable_factory_hub_coop = st.checkbox("启用工厂与中转站的合作")
    if enable_factory_hub_coop:
        cooperation_factor = st.slider("选择工厂与中转站的合作程度", 0, 100, 50)

# 4. 商场对某些工厂产品的优先需求
with st.sidebar.expander("商场的优先需求", expanded=False):
    enable_mall_priority = st.checkbox("启用商场优先需求")
    if enable_mall_priority:
        priority_demand = st.slider("选择商场优先需求的程度", 0, 100, 50)

# 5. 商场之间的竞争关系
with st.sidebar.expander("商场之间的竞争关系", expanded=False):
    enable_mall_competition = st.checkbox("启用商场之间的竞争关系")
    if enable_mall_competition:
        mall_competition_factor = st.slider("选择商场之间的竞争程度", 0, 100, 50)
        st.markdown(f"<h3 style='color: blue;font-size: 17px;'>该部分代码尚未成功编写，所以优化失败（敬请期待2.0版本的修正</h3>",
                unsafe_allow_html=True)

# 随机生成运输成本
np.random.seed(42)  # 保证可重复性
q1 = np.random.randint(q1_min, q1_max, size=(m, n))  # 工厂到中转站的运输成本
q2 = np.random.randint(q2_min, q2_max, size=(n, k))  # 中转站到商店的运输成本
q3 = np.random.randint(q3_min, q3_max, size=(n, n))  # 中转站到中转站的运输成本

# 调整工厂和中转站的运输成本与合作关系
if enable_factory_hub_coop:
    q1 = q1 * (1 - cooperation_factor / 100)

# 定义目标函数系数：最小化总成本 = 总运输成本 - 总收益
num_vars = m * n + n * k + n * n
c = np.concatenate([q1.flatten(), -q2.flatten(), q3.flatten()])

# 定义约束条件
A = []
B = []



st.sidebar.header("调整生产与需求设置")
# 工厂生产能力设置的展开区域
with st.sidebar.expander("工厂生产力设置"):
    # 约束1：各个工厂的生产能力有限
    for i in range(m):
        constraint = [0] * num_vars
        for j in range(n):
            constraint[i * n + j] = 1  # 工厂 i 向所有中转站 j 供货的总量
        A.append(constraint)

        # 使用侧边栏的滑动条来设置每个工厂的最大生产能力
        max_capacity = st.slider(f'工厂 {i + 1} 的最大生产能力', min_value=300, max_value=5000, value=600)

        # 如果启用了竞争因素，减少工厂的生产能力
        if enable_competition:
            max_capacity *= (1 - factory_competition / 100)

        B.append(max_capacity)

# 商场最小需求设置的展开区域
with st.sidebar.expander("商场需求量设置"):
    # 约束2：每个商场至少有一定的需求量
    for l in range(k):
        constraint = [0] * num_vars
        for j in range(n):
            constraint[m * n + j * k + l] = 1  # 中转站 j 向商场 l 供货的总量

        # 使用侧边栏的滑动条来设置每个商场的最小需求量
        min_demand = st.slider(f'商店 {l + 1} 的最小需求量', min_value=5, max_value=800, value=200)

        # 如果启用了商场优先级，则增加最小需求量
        if enable_mall_priority:
            min_demand *= (1 + priority_demand / 100)

        B.append(min_demand)
        A.append(constraint)

# 约束3：中转站的流入等于流出 + 中转站间的再分配
for j in range(n):
    constraint = [0] * num_vars
    for i in range(m):
        constraint[i * n + j] = 1  # 工厂 i 到中转站 j 的供货
    for l in range(k):
        constraint[m * n + j * k + l] = -1  # 中转站 j 到商店 l 的供货
    for j2 in range(n):
        if j != j2:
            constraint[m * n + n * k + j * n + j2] = -1  # 中转站 j 到其他中转站 j2 的供货
    A.append(constraint)
    B.append(0)  # 流入等于流出

# 约束4：中转站流量限制
if enable_flow_limit:
    for j in range(n):
        A_flow_limit = [0] * num_vars

        # 工厂到中转站的流入
        for i in range(m):
            A_flow_limit[i * n + j] = 1

        # 中转站到商店和其他中转的流出
        for l in range(k):
            A_flow_limit[m * n + j * k + l] = -1

        # 中转站之间的流动，添加负号处理
        for j2 in range(n):
            if j != j2:
                A_flow_limit[m * n + n * k + j * n + j2] = -1

        A.append(A_flow_limit)
        B.append(flow_limit)

# 约束5：商场之间的竞争关系
if enable_mall_competition:
    for l in range(k):
        A_competition = [0] * num_vars
        for j in range(n):
            A_competition[m * n + j * k + l] = 1
        A.append(A_competition)
        B.append(B[m + l] * (1 - mall_competition_factor / 100))


# 约束6: 所有工厂到所有中转站的总运输量不超过商品总量 w
total_constraint = [1] * (m * n) + [0] * (n * k + n * n)  # 只考虑工厂到中转站的流量
#A.append(total_constraint)
#B.append(w)


# 解决线性规划问题
A = np.array(A)
B = np.array(B)
res = linprog(c, A_eq=A, b_eq=B, method='highs')

# 检查优化是否成功
if res.success:
    st.subheader('优化结果')
    total_cost = res.fun
    st.markdown(f"<h3 style='color: blue;font-size: 20px;'>总运输成本: {total_cost:.2f}</h3>",
                unsafe_allow_html=True)

    # 分离不同类型的流动变量
    flow_factory_to_transit = res.x[:m * n].reshape((m, n))
    flow_transit_to_store = res.x[m * n:m * n + n * k].reshape((n, k))
    flow_transit_to_transit = res.x[m * n + n * k:].reshape((n, n))

    labels = [f'工厂 {i + 1}' for i in range(m)] + [f'中转站 {j + 1}' for j in range(n)] + [f'商店 {j + 1}' for j in
                                                                                            range(k)]
    sources, targets, values = [], [], []

    # 工厂到中转站的流动
    for i in range(m):
        for j in range(n):
            if flow_factory_to_transit[i, j] > 0:
                sources.append(i)
                targets.append(m + j)
                values.append(flow_factory_to_transit[i, j])

    # 中转站到商店的流动
    for i in range(n):
        for j in range(k):
            if flow_transit_to_store[i, j] > 0:
                sources.append(m + i)
                targets.append(m + n + j)
                values.append(flow_transit_to_store[i, j])

    # 中转站到中转站的流动
    for i in range(n):
        for j in range(n):
            if i != j and flow_transit_to_transit[i, j] > 0:
                sources.append(m + i)
                targets.append(m + j)
                values.append(flow_transit_to_transit[i, j])

    fig = go.Figure(data=[go.Sankey(
        node=dict(
            pad=15,
            thickness=20,
            line=dict(color="black", width=0.5),
            label=labels,
        ),
        link=dict(
            source=sources,
            target=targets,
            value=values
        )
    )])

    st.subheader('桑基图')
    st.markdown(f"<h3 style='color: green;font-size: 18px;'>图片可放大调整拖动</h3>",
                unsafe_allow_html=True)
    st.plotly_chart(fig)

    # 第二部分：表格展示
    # 工厂到中转站的货物流动表格
    data_factory_to_transit = [
        {'来源': f'工厂 {i + 1}', '目标': f'中转站 {j + 1}', '流量': flow_factory_to_transit[i, j]}
        for i in range(m) for j in range(n) if flow_factory_to_transit[i, j] > 0
    ]

    # 中转站到商店的货物流动表格
    data_transit_to_store = [
        {'来源': f'中转站 {i + 1}', '目标': f'商店 {j + 1}', '流量': flow_transit_to_store[i, j]}
        for i in range(n) for j in range(k) if flow_transit_to_store[i, j] > 0
    ]

    # 中转站到中转站的货物流动表格
    data_transit_to_transit = [
        {'来源': f'中转站 {i + 1}', '目标': f'中转站 {j + 1}', '流量': flow_transit_to_transit[i, j]}
        for i in range(n) for j in range(n) if i != j and flow_transit_to_transit[i, j] > 0
    ]

    # 展示表格
    st.subheader('工厂到中转站的货物流动')
    df_factory_to_transit = pd.DataFrame(data_factory_to_transit)
    st.write(df_factory_to_transit)

    st.subheader('中转站到商店的货物流动')
    df_transit_to_store = pd.DataFrame(data_transit_to_store)
    st.write(df_transit_to_store)

    st.subheader('中转站到中转站的货物流动')
    df_transit_to_transit = pd.DataFrame(data_transit_to_transit)
    st.write(df_transit_to_transit)

else:
    st.subheader('优化失败')
    st.write('无法找到满足约束条件的解，请调整输入参数。')
    st.write("可能的原因包括：生产能力不足、需求过高、约束过多，请查看约束条件的设置。")
    st.write("失败具体数值分析：")
    st.write(f'最终目标函数值：{res.fun}')
    st.write(f'约束矩阵 A 的形状：{A.shape}')
    st.write(f'约束向量 B 的形状：{B.shape}')
    st.write(f'最终优化变量：{res.x}')

# 调试输出：显示约束矩阵和向量
st.write("约束矩阵 A:")
st.write(np.array(A))
st.write("约束向量 B:")
st.write(np.array(B))

st.markdown(f"<h3 style='color: green;font-size: 18px;'>以下可展开查看</h3>",
                unsafe_allow_html=True)
with st.expander("点击展开查看：三个运输成本矩阵"):

    # 调试输出：显示生成的成本矩阵
    st.write("工厂到中转站的运输成本矩阵 (q1):")
    st.write(q1)
    st.write("中转站到商店的运输成本矩阵 (q2):")
    st.write(q2)
    st.write("中转站到中转站的运输成本矩阵 (q3):")
    st.write(q3)