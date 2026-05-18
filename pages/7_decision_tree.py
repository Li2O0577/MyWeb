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
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, export_text
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, mean_squared_error, mean_absolute_error, r2_score
import plotly.graph_objects as go
from pages._prepare import render_sidebar, data_uploader

# 页面配置（统一风格）
st.set_page_config(page_title="决策树", layout="wide", initial_sidebar_state="collapsed")
st.markdown("""
    <style>
        [data-testid="stSidebarNav"] {display: none;}
    </style>
""", unsafe_allow_html=True)
render_sidebar("pages/7_decision_tree.py")
st.title("🌳 决策树模型")

# 统一模型保存路径
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)
MODEL_PATH = os.path.join(MODEL_DIR, "dt_model.pkl")
CONFIG_PATH = os.path.join(MODEL_DIR, "dt_config.json")

# 功能介绍
with st.expander("📢 功能介绍", expanded=True):
    st.markdown("""
    ### 决策树模型（分类 & 回归 · 带可解释性）
    1. **双任务支持**：分类任务 + 回归任务，一键切换
    2. **智能预处理**：自动识别分类特征 + **独热编码**，数值特征标准化
    3. **可解释性**：**展示每一层决策规则**、划分特征、阈值、不纯度
    4. **模型评估**：分类 — 混淆矩阵 + P/R/F1；回归 — R² + MAE + RMSE
    5. **参数可调**：树深度、划分标准（分类：熵/基尼，回归：平方误差等）
    6. **模型持久化**：自动保存/加载，无需重复训练
    7. **预测功能**：输入新数据，输出分类/回归结果
    """)

# 加载数据
df = data_uploader()
if df is None:
    st.warning("⚠️ 请先上传数据！")
    st.stop()

# 数据预处理
df_before = len(df)
df_clean = df.dropna()
dropped = df_before - len(df_clean)
if dropped > 0:
    st.warning(f"⚠️ 已自动丢弃 {dropped} 行含缺失值的数据（剩余 {len(df_clean)} 行 / {df_before} 行）。可在「数据处理」页面手动处理缺失值。")
if len(df_clean) < 10 or df_clean.shape[1] < 2:
    st.error("❌ 数据无效！需要至少10行有效数据")
    st.stop()

# 自动区分：数值列 / 分类列（用于独热编码）
numeric_cols = df_clean.select_dtypes(include=[np.number]).columns.tolist()
categorical_cols = df_clean.select_dtypes(exclude=[np.number]).columns.tolist()

# 初始化任务类型
if "dt_task_type" not in st.session_state:
    st.session_state.dt_task_type = "classification"

# 任务类型选择
st.subheader("⚙️ 模型参数配置")

target_col = st.selectbox("选择目标列（y）", df_clean.columns, index=len(df_clean.columns)-1)

task_col, depth_col, crit_col = st.columns(3)
with task_col:
    task_type = st.selectbox("任务类型", ["分类 (Classification)", "回归 (Regression)"],
                             index=0 if st.session_state.dt_task_type == "classification" else 1)
    st.session_state.dt_task_type = "classification" if "分类" in task_type else "regression"
with depth_col:
    max_depth = st.number_input("决策树最大深度", min_value=2, max_value=15, value=3, step=1)
with crit_col:
    if st.session_state.dt_task_type == "classification":
        criterion = st.selectbox("划分标准", ["entropy", "gini"], index=0)
    else:
        criterion = st.selectbox("划分标准", ["squared_error", "friedman_mse", "absolute_error", "poisson"], index=0)
    st.info(f"✅ 数值特征：{len(numeric_cols)} 个 | 分类特征：{len(categorical_cols)} 个")

# 特征列（排除目标列）
feature_cols = [col for col in df_clean.columns if col != target_col]
n_features = len(feature_cols)
n_samples = len(df_clean)

# 基础校验
if n_features == 0:
    st.warning("⚠️ 无可用特征列！")
    st.stop()

if st.session_state.dt_task_type == "classification":
    n_classes = len(df_clean[target_col].unique())
    if n_classes < 2:
        st.warning("⚠️ 目标列至少需要2个类别！")
        st.stop()
    if n_classes > n_samples * 0.5:
        st.warning(f"⚠️ 类别数 ({n_classes}) 接近样本数 ({n_samples})，可能过拟合")
else:
    if target_col not in numeric_cols:
        st.warning("⚠️ 回归任务请选择数值型目标列！")
        st.stop()
    n_classes = None

#  模型加载函数 
def load_saved_dt():
    if all(os.path.exists(p) for p in [MODEL_PATH, CONFIG_PATH]):
        try:
            with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
                config = json.load(f)
            
            with open(MODEL_PATH, 'rb') as f:
                pipeline = pickle.load(f)

            st.session_state.dt_pipeline = pipeline
            st.session_state.dt_features = config.get("features", feature_cols)
            st.session_state.dt_target = config.get("target", target_col)
            st.session_state.dt_criterion = config.get("criterion", criterion)
            st.session_state.dt_task_type_saved = config.get("task_type", "classification")
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

# 解析决策树层级
def show_tree_rules(model):
    # 找到非预处理步骤（分类器或回归器）
    tree = None
    for step_name, step in model.named_steps.items():
        if hasattr(step, 'tree_'):
            tree = step
            break
    if tree is None:
        st.warning("无法解析决策树结构。")
        return

    tree_ = tree.tree_
    preprocessor = model.named_steps["preprocess"]

    try:
        expanded_names = list(preprocessor.get_feature_names_out())
    except Exception:
        expanded_names = [f"x{i}" for i in range(tree_.n_features_)]

    rules = export_text(tree, feature_names=expanded_names)
    st.markdown("### 🌿 决策树层级规则")
    st.code(rules, language="text")

    # 不纯度名称映射
    criterion_map = {
        "entropy": "信息熵", "gini": "基尼系数",
        "squared_error": "平方误差", "friedman_mse": "Friedman MSE",
        "absolute_error": "绝对误差", "poisson": "Poisson偏差"
    }
    criterion_name = criterion_map.get(tree.criterion, tree.criterion)

    st.markdown("### 📊 每层节点详细信息")
    node_info = []
    for i in range(tree_.node_count):
        node_info.append({
            "节点ID": i,
            "划分特征": expanded_names[tree_.feature[i]] if tree_.feature[i] != -2 else "叶子节点",
            "划分阈值": round(tree_.threshold[i], 4) if tree_.feature[i] != -2 else "-",
            f"{criterion_name}值": round(tree_.impurity[i], 4),
            "样本数量": int(tree_.n_node_samples[i]),
            "节点类型": "内部节点" if tree_.children_left[i] != -1 else "叶子节点"
        })
    st.dataframe(pd.DataFrame(node_info), use_container_width=True)

#  模型训练
is_cls = (st.session_state.dt_task_type == "classification")

st.subheader("🚀 模型训练")
train_col, clear_col = st.columns(2)
with train_col:
    btn_label = "开始训练" if "dt_pipeline" not in st.session_state else "重新训练"
    if st.button(btn_label, type="primary", use_container_width=True):
        with st.spinner("训练中（自动独热编码+训练）..."):
            # 1. 数据划分
            X = df_clean[feature_cols]
            y = df_clean[target_col]
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42,
                stratify=y if is_cls else None
            )

            # 2. 预处理流水线（按实际列类型动态构建，避免空 transformer 报错）
            num_cols = [c for c in numeric_cols if c in feature_cols]
            cat_cols = [c for c in categorical_cols if c in feature_cols]
            transformers = []
            if num_cols:
                transformers.append(("num", StandardScaler(), num_cols))
            if cat_cols:
                transformers.append(("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols))
            preprocessor = ColumnTransformer(transformers=transformers, remainder='passthrough')

            # 3. 决策树流水线（分类 or 回归）
            if is_cls:
                tree_model = DecisionTreeClassifier(criterion=criterion, max_depth=max_depth, random_state=42)
            else:
                tree_model = DecisionTreeRegressor(criterion=criterion, max_depth=max_depth, random_state=42)

            pipeline = Pipeline([
                ("preprocess", preprocessor),
                ("tree", tree_model)
            ])

            # 4. 训练
            pipeline.fit(X_train, y_train)

            # 5. 评估
            y_pred = pipeline.predict(X_test)

            # 保存会话
            st.session_state.dt_pipeline = pipeline
            st.session_state.dt_features = feature_cols
            st.session_state.dt_target = target_col
            st.session_state.dt_criterion = criterion
            st.session_state.dt_task_type_saved = st.session_state.dt_task_type

            if is_cls:
                acc = accuracy_score(y_test, y_pred)
                cm = confusion_matrix(y_test, y_pred)
                unique_test_labels = sorted(set(y_test) | set(y_pred))
                label_names = [str(l) for l in unique_test_labels]
                st.session_state.dt_acc = acc
                st.session_state.dt_cm = cm
                st.session_state.dt_label_names = label_names
                st.session_state.dt_y_test = y_test
                st.session_state.dt_y_pred = y_pred
                # 清除回归 keys
                for k in ["dt_r2", "dt_mae", "dt_rmse", "dt_reg_y_test", "dt_reg_y_pred"]:
                    st.session_state.pop(k, None)
            else:
                r2 = r2_score(y_test, y_pred)
                mae = mean_absolute_error(y_test, y_pred)
                rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                st.session_state.dt_r2 = r2
                st.session_state.dt_mae = mae
                st.session_state.dt_rmse = rmse
                st.session_state.dt_reg_y_test = y_test
                st.session_state.dt_reg_y_pred = y_pred
                # 清除分类 keys
                for k in ["dt_acc", "dt_cm", "dt_label_names", "dt_y_test", "dt_y_pred"]:
                    st.session_state.pop(k, None)

            # 6. 保存模型文件
            with open(MODEL_PATH, 'wb') as f:
                pickle.dump(pipeline, f)
            with open(CONFIG_PATH, 'w', encoding='utf-8') as f:
                json.dump({
                    "features": feature_cols,
                    "target": target_col,
                    "criterion": criterion,
                    "max_depth": max_depth,
                    "task_type": st.session_state.dt_task_type
                }, f, ensure_ascii=False)

            if is_cls:
                st.success(f"✅ 训练完成！测试集准确率 = {acc:.4f}")
            else:
                st.success(f"✅ 训练完成！R² = {r2:.4f} | MAE = {mae:.4f} | RMSE = {rmse:.4f}")

# 清除模型
with clear_col:
    if st.button("清除已保存决策树模型", use_container_width=True):
        for p in [MODEL_PATH, CONFIG_PATH]:
            if os.path.exists(p):
                os.remove(p)
        keys = ["dt_pipeline", "dt_features", "dt_target", "dt_criterion", "dt_task_type_saved",
                "dt_acc", "dt_cm", "dt_label_names", "dt_y_test", "dt_y_pred",
                "dt_r2", "dt_mae", "dt_rmse", "dt_reg_y_test", "dt_reg_y_pred"]
        for k in keys:
            if k in st.session_state:
                del st.session_state[k]
        st.warning("已清除决策树模型！")

#  模型评估可视化
saved_task = st.session_state.get("dt_task_type_saved", st.session_state.dt_task_type)
is_cls_eval = (saved_task == "classification")

if is_cls_eval and "dt_cm" in st.session_state:
    st.subheader("📊 模型评估")
    cm = st.session_state.dt_cm
    label_names = st.session_state.dt_label_names
    y_test = st.session_state.dt_y_test
    y_pred = st.session_state.dt_y_pred

    col_cm, col_report = st.columns([1, 1])
    with col_cm:
        st.caption("混淆矩阵")
        fig_cm = go.Figure(data=go.Heatmap(
            z=cm, x=label_names, y=label_names,
            text=cm, texttemplate="%{text}", textfont=dict(size=14),
            colorscale="Blues", showscale=False
        ))
        fig_cm.update_layout(xaxis_title="预测值", yaxis_title="实际值",
                             height=300, margin=dict(l=0, r=0, t=0, b=0))
        st.plotly_chart(fig_cm, use_container_width=True)

    with col_report:
        st.caption("分类报告")
        try:
            report = classification_report(y_test, y_pred, target_names=label_names,
                                           output_dict=True, zero_division=0)
            report_df = pd.DataFrame(report).transpose()
            display_df = report_df[['precision', 'recall', 'f1-score', 'support']]
            st.dataframe(display_df.style.format("{:.3f}", subset=['precision', 'recall', 'f1-score']),
                         use_container_width=True)
        except Exception:
            st.text(classification_report(y_test, y_pred, zero_division=0))

if not is_cls_eval and "dt_r2" in st.session_state:
    st.subheader("📊 模型评估")
    r2 = st.session_state.dt_r2
    mae = st.session_state.dt_mae
    rmse = st.session_state.dt_rmse
    y_test = st.session_state.dt_reg_y_test
    y_pred = st.session_state.dt_reg_y_pred

    col1, col2 = st.columns(2)
    with col1:
        st.caption("回归指标")
        m_col1, m_col2, m_col3 = st.columns(3)
        with m_col1:
            st.metric("R² Score", f"{r2:.4f}")
        with m_col2:
            st.metric("MAE", f"{mae:.4f}")
        with m_col3:
            st.metric("RMSE", f"{rmse:.4f}")

    with col2:
        st.caption("预测 vs 实际值")
        fig_scatter = go.Figure()
        fig_scatter.add_trace(go.Scatter(
            x=y_test, y=y_pred, mode='markers',
            marker=dict(size=6, opacity=0.6),
            name='预测值'
        ))
        min_val = min(y_test.min(), y_pred.min())
        max_val = max(y_test.max(), y_pred.max())
        fig_scatter.add_trace(go.Scatter(
            x=[min_val, max_val], y=[min_val, max_val],
            mode='lines', line=dict(dash='dash', color='gray'),
            name='理想线'
        ))
        fig_scatter.update_layout(
            xaxis_title="实际值", yaxis_title="预测值",
            height=300, margin=dict(l=0, r=0, t=0, b=0),
            template="plotly_white", showlegend=False
        )
        st.plotly_chart(fig_scatter, use_container_width=True)

st.divider()

#  决策树可解释性展示
st.subheader("🌿 决策树规则与节点信息")
if "dt_pipeline" not in st.session_state:
    st.warning("请先训练模型！")
else:
    show_tree_rules(st.session_state.dt_pipeline)

#预测功能
st.subheader("🎯 新数据预测")
if "dt_pipeline" not in st.session_state:
    st.warning("请先训练模型！")
else:
    model = st.session_state.dt_pipeline
    features = st.session_state.dt_features
    predict_is_cls = (st.session_state.get("dt_task_type_saved", "classification") == "classification")

    st.info(f"✅ 模型加载成功 | 任务类型：{'分类' if predict_is_cls else '回归'}")
    st.write("请输入特征值进行预测：")
    input_data = {}
    cols = st.columns(min(len(features), 5))
    for i, col in enumerate(cols):
        feat = features[i]
        if feat in categorical_cols:
            options = df_clean[feat].unique().tolist()
            val = col.selectbox(f"{feat}", options, key=f"dt_{i}")
        else:
            val = col.number_input(f"{feat}", value=0.0, step=0.1, key=f"dt_{i}")
        input_data[feat] = [val]

    if len(features) > 5:
        for row_start in range(5, len(features), 5):
            cols = st.columns(5)
            for j, col in enumerate(cols):
                idx = row_start + j
                if idx < len(features):
                    feat = features[idx]
                    if feat in categorical_cols:
                        options = df_clean[feat].unique().tolist()
                        val = col.selectbox(f"{feat}", options, key=f"dt_{idx}")
                    else:
                        val = col.number_input(f"{feat}", value=0.0, step=0.1, key=f"dt_{idx}")
                    input_data[feat] = [val]

    btn_label = "执行分类预测" if predict_is_cls else "执行回归预测"
    if st.button(btn_label, use_container_width=True):
        input_df = pd.DataFrame(input_data)
        pred = model.predict(input_df)[0]
        if predict_is_cls:
            st.success(f"🎯 预测类别：{pred}")
        else:
            st.success(f"🎯 预测结果：{pred:.4f}")