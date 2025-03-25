import streamlit as st
import pandas as pd
import joblib

# **加载 BP 神经网络模型 和 标准化器**
bp_model = joblib.load("res/bp_model.pkl")
scaler = joblib.load("res/scaler.pkl")

# **预定义特征列**
expected_columns = [
    "长表总分", "皮温mean", "△a*", "△B*", "缓慢胃率", "PIF", "Penh", "SCL",
    "饮酒_2", "近视的度数_2", "近视的度数_3", "近视的度数_4",
    "成年期白天的情绪_2", "直系亲属是否有疾病史_2"
]

# **居中显示标题，字体稍微小一点**
st.markdown(
    """
    <h2 style='text-align: center; font-size: 28px;'>
        🏃‍♂️ 基于BP神经网络的大学生运动病风险评估系统
    </h2>
    """,
    unsafe_allow_html=True
)

# **引导文本（更小字体）**
st.markdown(
    "<p style='text-align: center; font-size: 12px; color: gray;'>在左侧侧边栏输入特征值，点击下方按钮开始评估</p>",
    unsafe_allow_html=True
)

# **侧边栏输入**
with st.sidebar:
    st.markdown("## 📥 请输入特征值")

    # **个人基本信息（双列布局）**
    st.markdown("### 📋 个人基本信息")
    col1, col2 = st.columns(2)
    with col1:
        饮酒 = st.radio("饮酒", [1, 2], format_func=lambda x: "不饮酒" if x == 1 else "饮酒")
        成年期白天的情绪 = st.radio("成年期白天的情绪", [1, 2], format_func=lambda x: "正常" if x == 1 else "低落")
    with col2:
        直系亲属是否有疾病史 = st.radio("直系亲属是否有运动病史", [1, 2], format_func=lambda x: "无" if x == 1 else "有")
        近视的度数 = st.radio(
            "视力", 
            [1, 2, 3, 4], 
            format_func=lambda x: {
                1: "正常(0度)",
                2: "轻度近视(1-300度)",
                3: "中度近视(301-600度)",
                4: "重度近视(>600度)"
            }[x]
        )

    # **分隔线**
    st.markdown("---")

    # **问卷**
    st.markdown("### 📝 问卷")
    长表总分 = st.number_input("中文简化版MSSQ-L总分", value=18.6)

    # **分隔线**
    st.markdown("---")

    # **生理指标（双列布局）**
    st.markdown("### 📊 生理指标")
    col3, col4 = st.columns(2)
    with col3:
        皮温mean = st.number_input("皮温 Mean", value=30.5)
        缓慢胃率 = st.number_input("缓慢胃率", value=0)
        delta_a = st.number_input("△a", value=1.78)
        delta_b = st.number_input("△b", value=0.89)
    with col4:
        PIF = st.number_input("PIF", value=1.2)
        Penh = st.number_input("Penh", value=0.5)
        SCL = st.number_input("SCL", value=3.4)

# **评估按钮居中**
st.markdown("<div style='text-align: center;'>", unsafe_allow_html=True)
if st.button("✅ 开始评估", use_container_width=True):
    # **处理输入数据**
    input_data = {
        "饮酒": 饮酒,
        "近视的度数": 近视的度数,
        "成年期白天的情绪": 成年期白天的情绪,
        "直系亲属是否有疾病史": 直系亲属是否有疾病史,
        "长表总分": 长表总分,
        "皮温mean": 皮温mean,
        "△a*": delta_a,
        "△B*": delta_b,  
        "缓慢胃率": 缓慢胃率,
        "PIF": PIF,
        "Penh": Penh,
        "SCL": SCL
    }

    # **转换输入为 DataFrame**
    input_df = pd.DataFrame([input_data])

    # **独热编码**
    discrete_mapping = {
        "饮酒": "饮酒_2",
        "近视的度数": {2: "近视的度数_2", 3: "近视的度数_3", 4: "近视的度数_4"},
        "成年期白天的情绪": "成年期白天的情绪_2",
        "直系亲属是否有疾病史": "直系亲属是否有疾病史_2"
    }

    for col, mapping in discrete_mapping.items():
        if isinstance(mapping, dict):
            for val, col_name in mapping.items():
                input_df[col_name] = 1 if input_df[col].iloc[0] == val else 0
        else:
            input_df[mapping] = 1 if input_df[col].iloc[0] == 2 else 0

    input_df.drop(columns=["饮酒", "近视的度数", "成年期白天的情绪", "直系亲属是否有疾病史"], inplace=True)

    # **补充缺失的列**
    missing_cols = set(expected_columns) - set(input_df.columns)
    for col in missing_cols:
        input_df[col] = 0  

    input_df = input_df[expected_columns]

    # **标准化数据**
    input_scaled = scaler.transform(input_df)

    # **进行评估**
    prediction = bp_model.predict(input_scaled)

    # **显示结果**
    st.markdown("<br>", unsafe_allow_html=True)
    if prediction[0] == 1:
        st.warning("⚠️ 评估结果：该学生可能有运动病风险！请注意防护措施。")
    else:
        st.success("✅ 评估结果：该学生无明显运动病风险！请保持健康生活习惯。")
st.markdown("</div>", unsafe_allow_html=True)