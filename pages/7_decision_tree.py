import streamlit as st
import pandas as pd
import numpy as np
import os
import pickle
import json
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.metrics import accuracy_score
from pages._prepare import render_sidebar, data_uploader

# 页面配置（统一风格）
st.set_page_config(page_title="决策树", layout="wide", initial_sidebar_state="collapsed")
st.markdown("""
    <style>
        [data-testid="stSidebarNav"] {display: none;}
    </style>
""", unsafe_allow_html=True)
render_sidebar("pages/7_decision_tree.py")
st.title("🌳 决策树分类模型")

# 统一模型保存路径
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)
MODEL_PATH = os.path.join(MODEL_DIR, "dt_model.pkl")
CONFIG_PATH = os.path.join(MODEL_DIR, "dt_config.json")

# 功能介绍
with st.expander("📢 功能介绍", expanded=True):
    st.markdown("""
    ### 决策树分类模型（带可解释性）
    1. **智能预处理**：自动识别分类特征 + **独热编码**，数值特征标准化
    2. **可解释性**：**展示每一层决策规则**、划分特征、阈值、熵/基尼系数
    3. **参数可调**：树深度、划分标准（熵/基尼）
    4. **模型持久化**：自动保存/加载，无需重复训练
    5. **预测功能**：输入新数据，输出分类结果
    """)

# 加载数据
df = data_uploader()
if df is None:
    st.warning("⚠️ 请先上传数据！")
    st.stop()

# 数据预处理
df_clean = df.dropna()
if len(df_clean) < 10 or df_clean.shape[1] < 2:
    st.error("❌ 数据无效！需要至少10行有效数据")
    st.stop()

# 自动区分：数值列 / 分类列（用于独热编码）
numeric_cols = df_clean.select_dtypes(include=[np.number]).columns.tolist()
categorical_cols = df_clean.select_dtypes(exclude=[np.number]).columns.tolist()

st.subheader("⚙️ 模型参数配置")
col1, col2, col3 = st.columns(3)
with col1:
    # 选择目标列
    target_col = st.selectbox("选择分类目标列（y）", df_clean.columns, index=len(df_clean.columns)-1)
with col2:
    # 决策树深度
    max_depth = st.number_input("决策树最大深度", min_value=2, max_value=10, value=3, step=1)
with col3:
    # 划分标准：熵 / 基尼
    criterion = st.selectbox("划分标准", ["entropy", "gini"], index=0)
    st.info(f"✅ 数值特征：{len(numeric_cols)} 个 | 分类特征：{len(categorical_cols)} 个")

# 特征列（排除目标列）
feature_cols = [col for col in df_clean.columns if col != target_col]
n_features = len(feature_cols)
n_samples = len(df_clean)
n_classes = len(df_clean[target_col].unique())

# 基础校验
if n_features == 0:
    st.warning("⚠️ 无可用特征列！")
    st.stop()
if n_classes < 2:
    st.warning("⚠️ 目标列至少需要2个类别！")
    st.stop()

#  模型加载函数 
def load_saved_dt():
    if all(os.path.exists(p) for p in [MODEL_PATH, CONFIG_PATH]):
        try:
            with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
                config = json.load(f)
            
            with open(MODEL_PATH, 'rb') as f:
                pipeline = pickle.load(f)

            st.session_state.dt_pipeline = pipeline
            st.session_state.dt_features = feature_cols
            st.session_state.dt_target = target_col
            st.session_state.dt_criterion = criterion
            return True
        except Exception as e:
            st.error(f"加载模型失败：{str(e)}")
            return False
    return False

# 自动加载模型
if "dt_pipeline" not in st.session_state:
    load_success = load_saved_dt()
    if load_success:
        st.toast("✅ 自动加载已保存的决策树模型！", icon="🎉")

# 解析决策树层级（展示每层特征+熵）
def show_tree_rules(model, feature_names):
    tree = model.named_steps["classifier"]
    tree_ = tree.tree_
    rules = export_text(tree, feature_names=feature_names)
    st.markdown("### 🌿 决策树层级规则（特征 + 熵/基尼）")
    st.code(rules, language="text")

    # 详细节点信息
    st.markdown("### 📊 每层节点详细信息")
    node_info = []
    for i in range(tree_.node_count):
        node_info.append({
            "节点ID": i,
            "划分特征": feature_names[tree_.feature[i]] if tree_.feature[i] != -2 else "叶子节点",
            "划分阈值": round(tree_.threshold[i], 4) if tree_.feature[i] != -2 else "-",
            f"{'信息熵' if criterion=='entropy' else '基尼系数'}值": round(tree_.impurity[i], 4),
            "样本数量": int(tree_.n_node_samples[i]),
            "节点类型": "内部节点" if tree_.children_left[i] != -1 else "叶子节点"
        })
    st.dataframe(pd.DataFrame(node_info), use_container_width=True)

#  模型训练 
st.subheader("🚀 模型训练")
train_col, clear_col = st.columns(2)
with train_col:
    if st.button("开始训练决策树", type="primary", use_container_width=True):
        with st.spinner("训练中（自动独热编码+训练）..."):
            # 1. 数据划分
            X = df_clean[feature_cols]
            y = df_clean[target_col]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # 2. 核心：预处理流水线（独热编码 + 标准化）
            preprocessor = ColumnTransformer(
                transformers=[
                    ("num", StandardScaler(), [c for c in numeric_cols if c in feature_cols]),
                    ("cat", OneHotEncoder(handle_unknown="ignore"), [c for c in categorical_cols if c in feature_cols])
                ]
            )

            # 3. 决策树流水线
            pipeline = Pipeline([
                ("preprocess", preprocessor),
                ("classifier", DecisionTreeClassifier(
                    criterion=criterion,
                    max_depth=max_depth,
                    random_state=42
                ))
            ])

            # 4. 训练
            pipeline.fit(X_train, y_train)

            # 5. 评估
            y_pred = pipeline.predict(X_test)
            acc = accuracy_score(y_test, y_pred)

            # 6. 保存会话
            st.session_state.dt_pipeline = pipeline
            st.session_state.dt_features = feature_cols
            st.session_state.dt_target = target_col
            st.session_state.dt_criterion = criterion
            st.session_state.dt_acc = acc

            # 7. 保存模型文件
            with open(MODEL_PATH, 'wb') as f:
                pickle.dump(pipeline, f)
            with open(CONFIG_PATH, 'w', encoding='utf-8') as f:
                json.dump({
                    "features": feature_cols,
                    "target": target_col,
                    "criterion": criterion,
                    "max_depth": max_depth
                }, f, ensure_ascii=False)

            st.success(f"✅ 训练完成！测试集准确率 = {acc:.4f}")

# 清除模型
with clear_col:
    if st.button("清除已保存决策树模型", use_container_width=True):
        for p in [MODEL_PATH, CONFIG_PATH]:
            if os.path.exists(p):
                os.remove(p)
        keys = ["dt_pipeline", "dt_features", "dt_target", "dt_criterion", "dt_acc"]
        for k in keys:
            if k in st.session_state:
                del st.session_state[k]
        st.warning("已清除决策树模型！")

#  决策树可解释性展示 
st.subheader("🌿 决策树规则与层级熵展示")
if "dt_pipeline" not in st.session_state:
    st.warning("请先训练模型！")
else:
    show_tree_rules(st.session_state.dt_pipeline, feature_cols)

#预测功能 
st.subheader("🎯 新数据预测")
if "dt_pipeline" not in st.session_state:
    st.warning("请先训练模型！")
else:
    model = st.session_state.dt_pipeline
    features = st.session_state.dt_features

    st.info("✅ 模型加载成功，请输入特征值：")
    input_data = {}
    cols = st.columns(len(features))
    
    for i, col in enumerate(cols):
        feat = features[i]
        # 分类特征下拉框，数值特征输入框
        if feat in categorical_cols:
            options = df_clean[feat].unique().tolist()
            val = col.selectbox(f"特征：{feat}", options, key=f"dt_{i}")
        else:
            val = col.number_input(f"特征：{feat}", value=0.0, step=0.1, key=f"dt_{i}")
        input_data[feat] = [val]

    if st.button("执行分类预测", use_container_width=True):
        input_df = pd.DataFrame(input_data)
        pred = model.predict(input_df)[0]
        st.success(f"🎯 预测结果：{pred}")