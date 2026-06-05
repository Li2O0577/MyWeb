"""DIY MLP page — Flask backend for training, Streamlit for UI."""
import streamlit as st
import pandas as pd
import numpy as np
from pages._prepare import render_sidebar, data_uploader
from pages._mlp_common import render_device_selector, check_constant_features, plot_loss_curve, validate_input_array, render_version_selector, render_task_mismatch_warning, render_ml_status_bar, render_ml_workbench_overview, render_stable_prediction_panel, render_small_dataset_warning, render_class_balance_warning, render_risk_notice, task_mismatch_message, prepare_batch_prediction, render_batch_prediction_output
from pages._api import train_diy_mlp, predict_diy_mlp, batch_predict_diy_mlp, clear_diy_mlp, ensure_session, diy_mlp_status, backend_status_badge, render_backend_sync_panel, list_diy_mlp_versions, activate_diy_mlp_version, delete_diy_mlp_version
from pages._ui_common import render_page_header, render_section_header

st.set_page_config(page_title="自定义 MLP", layout="wide", initial_sidebar_state="collapsed")
st.markdown("""<style>[data-testid="stSidebarNav"] {display: none;}</style>""", unsafe_allow_html=True)
render_sidebar("pages/6_diy_mlp.py")
render_page_header("自定义神经网络", "配置 MLP 网络结构，支持分类和回归两类任务。")

backend_status_badge()
device = render_device_selector()

with st.expander("功能介绍", expanded=False):
    st.markdown("""
    ### 自定义神经网络构建器
    1. **自由设计**：自行添加隐藏层，设定每层神经元数、激活函数、批归一化、Dropout
    2. **双任务支持**：回归预测 + 分类决策，自动适配输出层与损失函数
    3. **前后端分离**：Flask 后端训练，Streamlit 前端展示
    """)

df = data_uploader()
if df is None:
    st.warning("⚠️ 请先上传数据！")
    st.stop()

numeric_df = df.select_dtypes(include=[np.number]).dropna().copy()
if len(numeric_df) < 2 or len(numeric_df.columns) < 1:
    st.error("❌ 数据无效！需要至少 1 列数值数据作为特征")
    st.stop()

df_clean = df.dropna()
if len(df_clean) < 10:
    st.error("❌ 数据无效！删除缺失值后不足10行")
    st.stop()

backend_synced = render_backend_sync_panel(df_clean, compact=True)

numeric_df.columns = numeric_df.columns.astype(str)
all_num_cols = list(numeric_df.columns)
all_cols = list(df.columns)

render_section_header("任务配置", "选择任务类型、目标列和输入特征。")
task_col1, task_col2 = st.columns(2)
with task_col1:
    task_type = st.selectbox("任务类型", ["回归 (Regression)", "分类 (Classification)"], index=0)
with task_col2:
    if "分类" in task_type:
        target_options = all_cols
    else:
        target_options = all_num_cols
    target_col = st.selectbox("选择目标列", target_options, index=len(target_options) - 1)

feature_cols = [c for c in all_num_cols if c != target_col]
n_features = len(feature_cols)
n_samples = len(df_clean)

if n_features == 0:
    st.warning("⚠️ 警告：选择的目标列覆盖了所有列，无可用特征列！请重新选择目标列")
    st.stop()

if task_type == "分类 (Classification)":
    n_classes = len(df_clean[target_col].unique())
    if n_classes < 2:
        st.error("❌ 目标列类别数 < 2，无法分类！")
        st.stop()
    if n_classes > n_samples * 0.5:
        render_risk_notice("类别过多", f"类别数 ({n_classes}) 接近样本数 ({n_samples})，容易过拟合。")
    st.info(f"✅ 数据：{n_samples} 行 | {n_features} 特征 | {n_classes} 类别")
    render_class_balance_warning(df_clean[target_col])
else:
    n_classes = None
    st.info(f"✅ 数据：{n_samples} 行 | {n_features} 特征")

render_small_dataset_warning(n_samples)
check_constant_features(numeric_df, feature_cols)

# Version selector
active_vid = render_version_selector(
    "DIY MLP",
    list_diy_mlp_versions,
    activate_diy_mlp_version,
    delete_diy_mlp_version,
    prediction_keys=["diy_last_prediction"],
)

# Auto-detect saved model — reloads when active version changes
status = diy_mlp_status()
render_ml_status_bar(df_clean, backend_synced, status, "DIY MLP")
current_vid = status.get("version_id", "") if status else ""
if "diy_result" not in st.session_state or st.session_state.get("diy_version_id") != current_vid:
    if status and status.get("has_model"):
        st.session_state.diy_result = {"train_losses": [], "val_losses": []}
        st.session_state.diy_features = status.get("features", [])
        st.session_state.diy_target = status.get("target", "")
        saved_params = status.get("params", {})
        saved_task = saved_params.get("task") or ("classification" if saved_params.get("n_classes") else "regression")
        st.session_state.diy_task = saved_task
        if saved_task == "classification":
            st.session_state.diy_n_classes = status.get("params", {}).get("n_classes", 2)
            st.session_state.diy_reverse_label_map = {}
        st.session_state.diy_version_id = current_vid
        # Restore layers from saved config
        saved_layers = status.get("params", {}).get("layers", [])
        if saved_layers and "diy_layers" not in st.session_state:
            st.session_state.diy_layers = saved_layers
        ds = status.get("dataset_name", "")
        created = status.get("created_at", "")[:16].replace("T", " ")
        st.success(f"已加载 DIY MLP 模型版本（{ds} | {created} | 目标列：{st.session_state.diy_target}）")

current_task_str = "classification" if "分类" in task_type else "regression"
render_task_mismatch_warning("DIY MLP", current_task_str, st.session_state.get("diy_task"))

render_ml_workbench_overview(
    "DIY MLP",
    df_clean,
    backend_synced=backend_synced,
    model_status=status,
    task_text="分类" if current_task_str == "classification" else "回归",
    target_text=str(target_col),
    feature_count=n_features,
    sample_count=n_samples,
    extra_items=[
        ("类别数", f"{n_classes} 个" if n_classes is not None else "不适用"),
        ("设备", str(device)),
    ],
)

# ── Layer builder UI ──
render_section_header("网络结构设计", "逐层配置神经元数量、激活函数、Dropout 和 BatchNorm。")
st.caption("自定义隐藏层。输入层和输出层会根据数据和任务自动生成。")

if "diy_layers" not in st.session_state:
    st.session_state.diy_layers = [
        {"neurons": max(8, n_features * 2), "activation": "ReLU", "bn": False, "dropout": 0.0}
    ]

ACT_OPTIONS = ["ReLU", "LeakyReLU", "GELU", "Tanh", "Sigmoid", "ELU", "SELU", "无激活"]

layer_cols = st.columns([3, 1, 1, 1, 0.8])
with layer_cols[0]: st.caption("**神经元数**")
with layer_cols[1]: st.caption("**激活函数**")
with layer_cols[2]: st.caption("**BatchNorm**")
with layer_cols[3]: st.caption("**Dropout**")
with layer_cols[4]: st.caption("**操作**")

layers_to_remove = []
for i, layer in enumerate(st.session_state.diy_layers):
    cols = st.columns([3, 1, 1, 1, 0.8])
    with cols[0]:
        layer["neurons"] = st.number_input(f"隐藏层 {i+1}", min_value=1, max_value=2048, value=layer["neurons"], step=1, key=f"neurons_{i}")
    with cols[1]:
        layer["activation"] = st.selectbox("激活", ACT_OPTIONS, index=ACT_OPTIONS.index(layer["activation"]) if layer["activation"] in ACT_OPTIONS else 0, key=f"act_{i}", label_visibility="collapsed")
    with cols[2]:
        layer["bn"] = st.checkbox("BN", value=layer["bn"], key=f"bn_{i}")
    with cols[3]:
        layer["dropout"] = st.slider("Dropout", 0.0, 0.8, layer["dropout"], 0.05, key=f"drop_{i}", label_visibility="collapsed")
    with cols[4]:
        if st.button("🗑️", key=f"del_{i}", use_container_width=True):
            layers_to_remove.append(i)

for i in sorted(layers_to_remove, reverse=True):
    if len(st.session_state.diy_layers) > 1:
        st.session_state.diy_layers.pop(i)

btn_col1, btn_col2, btn_col3 = st.columns([1, 1, 3])
with btn_col1:
    if st.button("➕ 添加隐藏层", use_container_width=True):
        prev_neurons = st.session_state.diy_layers[-1]["neurons"]
        st.session_state.diy_layers.append({"neurons": max(2, prev_neurons // 2), "activation": "ReLU", "bn": False, "dropout": 0.0})
        st.rerun()
with btn_col2:
    if st.button("🔄 重置", use_container_width=True):
        st.session_state.diy_layers = [{"neurons": max(8, n_features * 2), "activation": "ReLU", "bn": False, "dropout": 0.0}]
        st.rerun()

# Architecture summary
render_section_header("网络结构摘要", "检查当前网络深度、参数规模和潜在训练风险。")
layer_dims = [n_features]
for layer in st.session_state.diy_layers:
    layer_dims.append(layer["neurons"])
output_dim_display = 1 if task_type == "回归 (Regression)" or n_classes == 2 else n_classes
layer_dims.append(output_dim_display)

total_params = 0
for i in range(len(layer_dims) - 1):
    w = layer_dims[i] * layer_dims[i+1]
    b = layer_dims[i+1]
    bn_p = 2 * layer_dims[i+1] if (i < len(st.session_state.diy_layers) and st.session_state.diy_layers[i]["bn"]) else 0
    total_params += w + b + bn_p

arch_parts = [f"Input({n_features})"]
for i, layer in enumerate(st.session_state.diy_layers):
    extras = []
    if layer["bn"]: extras.append("BN")
    if layer["dropout"] > 0: extras.append(f"Drop({layer['dropout']:.1f})")
    extra_str = " + " + " + ".join(extras) if extras else ""
    arch_parts.append(f"Dense({layer['neurons']}) + {layer['activation']}{extra_str}")
arch_parts.append(f"Output({output_dim_display}) [{'Linear' if '回归' in task_type else 'Logits'}]")
st.code("  →  ".join(arch_parts), language=None)
st.caption(f"总参数量：**{total_params:,}**  |  训练样本：**{n_samples}**  |  参数量/样本比：**{total_params / max(n_samples, 1):.2f}**")

warnings = []
if total_params > n_samples * 2:
    warnings.append(f"🔴 严重过拟合风险：参数量 ({total_params:,}) 远超训练样本数 ({n_samples}) 的两倍。")
elif total_params > n_samples * 0.5:
    warnings.append(f"🟡 过拟合风险：参数量 ({total_params:,}) 超过训练样本数 ({n_samples}) 的一半。")
elif total_params < max(8, n_features):
    warnings.append(f"🟡 欠拟合风险：参数量 ({total_params:,}) 过少。")

# Check for gradient vanishing risk (3+ sigmoid/tanh layers)
vanishing_acts = ["Sigmoid", "Tanh"]
n_vanishing = sum(1 for l in st.session_state.diy_layers if l["activation"] in vanishing_acts)
if n_vanishing >= 3:
    warnings.append(f"⚠️ 梯度消失风险：{n_vanishing} 层使用 Sigmoid/Tanh，深层网络可能无法有效训练。建议中间层改用 ReLU/GELU/LeakyReLU。")

# Check for linear degeneration (all layers have no activation)
has_nonlinear = any(l["activation"] != "无激活" for l in st.session_state.diy_layers)
if not has_nonlinear:
    warnings.append(f"⚠️ 线性退化：所有隐藏层均无激活函数，整个网络等价于线性回归/逻辑回归，无法学习非线性模式。")

for w in warnings: st.warning(w)
if not warnings: st.success("✅ 网络规模与数据量匹配良好。")

# Training params
render_section_header("训练参数", "设置优化器、学习率、训练轮数、batch size 和早停策略。")
hp_col1, hp_col2, hp_col3, hp_col4 = st.columns(4)
with hp_col1:
    learning_rate = st.selectbox("学习率 (LR)", [0.01, 0.005, 0.001, 0.0005, 0.0001], index=1)
with hp_col2:
    optimizer_name = st.selectbox("优化器", ["Adam", "AdamW", "SGD", "RMSprop"], index=0)
with hp_col3:
    epochs = st.slider("最大训练轮数", 20, 500, 100, 20)
with hp_col4:
    batch_size = st.selectbox("批次大小", [4, 8, 16, 32, 64, 128], index=2)

val_split = st.slider("验证集比例", 0.1, 0.4, 0.2, 0.05)
patience = st.slider("早停耐心 (轮)", 5, 50, 15, 5)

# Training
render_section_header("模型训练", "启动训练或清除当前 DIY MLP 模型。")
train_col, clear_col = st.columns(2)

with train_col:
    if st.button("开始训练 / 重新训练", type="primary", use_container_width=True):
        if task_type == "分类 (Classification)":
            class_counts = df_clean[target_col].value_counts()
            if class_counts.min() < 2:
                st.toast("❌ 每个类别至少需要 2 个样本！", icon="❌")
                st.stop()

        train_df = df_clean if "分类" in task_type else numeric_df
        with st.spinner("训练中（后端 Flask 计算）..."):
            sid = ensure_session(train_df)
            if not sid:
                st.toast("无法连接到 Flask 后端 (http://localhost:5001)。请确保后端已启动。", icon="❌")
                st.stop()
            device_str = "cuda" if "CUDA" in str(device) else "cpu"
            task_str = "classification" if "分类" in task_type else "regression"
            result = train_diy_mlp(
                sid,
                target_col, feature_cols,
                st.session_state.diy_layers,
                task_str, n_classes or 0,
                learning_rate, optimizer_name,
                epochs, batch_size, val_split, patience, device_str
            )
            if result and "train_losses" in result:
                st.session_state.diy_result = result
                st.session_state.diy_features = feature_cols
                st.session_state.diy_target = target_col
                st.session_state.diy_task = task_str
                st.session_state.diy_version_id = result.get("version_id", "")
                if task_str == "classification":
                    st.session_state.diy_n_classes = result["n_classes"]
                    st.session_state.diy_reverse_label_map = result["reverse_label_map"]

                if task_str == "regression":
                    st.toast(f"训练完成！R² = {result['r2']:.4f} | MAE = {result['mae']:.4f} | RMSE = {result['rmse']:.4f} | 版本: {result.get('version_id', '?')[:20]}...", icon="✅")
                else:
                    st.toast(f"训练完成！准确率 = {result['acc']:.4f} | 版本: {result.get('version_id', '?')[:20]}...", icon="✅")

                plot_loss_curve(result["train_losses"], result["val_losses"])
            elif result is not None:
                st.toast("训练没有完成：后端返回的结果不完整，请查看后端终端日志。", icon="❌")

with clear_col:
    if st.button("清除已保存模型", use_container_width=True):
        clear_diy_mlp()
        for k in ["diy_result", "diy_features", "diy_target", "diy_task", "diy_n_classes", "diy_reverse_label_map", "diy_version_id", "diy_last_prediction"]:
            if k in st.session_state: del st.session_state[k]
        st.warning("已清除所有保存的模型！")

# Prediction
render_section_header("数据预测", "使用当前激活版本进行单条预测。")
if "diy_result" not in st.session_state:
    st.warning("请先完成模型训练！")
else:
    features = st.session_state.diy_features
    target = st.session_state.diy_target
    task = st.session_state.diy_task
    st.info(f"✅ 模型已训练 | 任务：{'回归' if task == 'regression' else '分类'} | 目标：{target}")

    st.write("请输入特征值进行预测：")
    input_data = []
    cols = st.columns(min(len(features), 5))
    for i, col in enumerate(cols):
        val = col.number_input(f"{features[i]}", value=0.0, step=0.1, key=f"pred_{i}")
        input_data.append(val)
    if len(features) > 5:
        for row_start in range(5, len(features), 5):
            cols = st.columns(5)
            for j, col in enumerate(cols):
                idx = row_start + j
                if idx < len(features):
                    val = col.number_input(f"{features[idx]}", value=0.0, step=0.1, key=f"pred_{idx}")
                    input_data.append(val)

    if st.button("执行预测", use_container_width=True, type="primary"):
        if len(input_data) != len(features):
            st.toast("输入特征数与模型特征数不匹配！", icon="❌")
        else:
            device_str = "cuda" if "CUDA" in str(device) else "cpu"
            version_id = st.session_state.get("diy_version_id")
            err = validate_input_array(np.array([input_data]), "单条预测", expected_features=len(features))
            if err:
                st.toast(err, icon="❌")
            else:
                result = predict_diy_mlp(input_data, device_str, version_id=version_id)
                if result:
                    if task == "regression":
                        st.session_state.diy_last_prediction = {
                            "main": f"{result['result']:.4f}",
                            "details": [
                                ("任务", "回归"),
                                ("预测目标", target),
                                ("模型版本", (version_id or "当前激活版本")[:24]),
                            ],
                            "task": "regression",
                        }
                        st.toast(f"预测结果：**{result['result']:.4f}**", icon="✅")
                    else:
                        n_cls = st.session_state.diy_n_classes
                        reverse_label_map = st.session_state.diy_reverse_label_map
                        pred_idx = result["pred_idx"]
                        pred_class = result.get("pred_class") or reverse_label_map.get(str(pred_idx), pred_idx)
                        probs = result.get("all_probs", [])
                        labels = result.get("label_names") or [reverse_label_map.get(str(i), i) for i in range(len(probs))]
                        st.session_state.diy_last_prediction = {
                            "main": str(pred_class),
                            "details": [
                                ("任务", "分类"),
                                ("预测目标", target),
                                ("置信度", f"{result['prob']:.4f}"),
                                ("模型版本", (version_id or "当前激活版本")[:24]),
                            ],
                            "task": "classification",
                            "probabilities": probs,
                            "probability_labels": labels,
                        }
                        st.toast(f"预测类别：**{pred_class}** | 置信度：**{result['prob']:.4f}**", icon="✅")

    if st.session_state.get("diy_last_prediction"):
        pred = st.session_state.diy_last_prediction
        if isinstance(pred, dict):
            render_stable_prediction_panel(
                "预测结果",
                pred["main"],
                details=pred["details"],
                model_status=status,
                fallback_result=st.session_state.get("diy_result"),
                probabilities=pred.get("probabilities"),
                probability_labels=pred.get("probability_labels"),
                mismatch_message=task_mismatch_message("DIY MLP", current_task_str, st.session_state.get("diy_task")),
            )
        else:
            st.session_state.pop("diy_last_prediction", None)

    st.divider()
    render_section_header("批量预测", "上传包含相同特征列的 CSV 文件并批量生成预测结果。")
    batch_file = st.file_uploader("上传包含特征列的 CSV 文件", type=["csv"], key="diy_batch")
    if batch_file is not None:
        batch_df, batch_X, batch_err = prepare_batch_prediction(batch_file, features, "DIY MLP 批量预测")
        if batch_err:
            st.toast(batch_err, icon="❌")
        else:
            device_str = "cuda" if "CUDA" in str(device) else "cpu"
            version_id = st.session_state.get("diy_version_id")
            result = batch_predict_diy_mlp(batch_X.tolist(), device_str, version_id=version_id)
            if result:
                if task == "regression":
                    result_df = batch_df.copy()
                    result_df[f"预测_{target}"] = result["predictions"]
                else:
                    reverse_label_map = st.session_state.diy_reverse_label_map
                    result_df = batch_df.copy()
                    result_df["预测类别"] = [reverse_label_map.get(str(i), i) for i in result["pred_indices"]]
                    result_df["置信度"] = result["confidences"]
                render_batch_prediction_output(result_df, "diy_mlp_predictions.csv")
