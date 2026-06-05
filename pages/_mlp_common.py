"""Shared utilities for MLP-based pages (regression, classification, DIY)."""
import streamlit as st
import os
import pickle
import json
import html
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from pages._ui_common import render_kv_panel, render_notice, render_prediction_panel, render_status_strip

try:
    import torch
except ImportError:
    torch = None


TASK_NAMES = {"classification": "分类", "regression": "回归"}


# ── Shared ML page UI ──

def render_ml_status_bar(df, backend_synced=False, model_status=None, model_label="模型"):
    """Render a compact, consistent status strip for ML pages."""
    meta = st.session_state.get("session_meta", {}) or {}
    dataset_name = str(meta.get("source_name", "当前数据"))
    rows = meta.get("rows", len(df) if df is not None else 0)
    cols = meta.get("n_columns", len(df.columns) if df is not None else 0)

    has_model = bool(model_status and model_status.get("has_model"))
    if has_model:
        version_id = model_status.get("version_id", "")
        created = (model_status.get("created_at", "") or "")[:16].replace("T", " ")
        target = model_status.get("target", "")
        model_text = f"{model_label}：{version_id[:18]}..."
        if target:
            model_text += f" | 目标：{target}"
        if created:
            model_text += f" | {created}"
    else:
        model_text = f"{model_label}：暂无已加载版本"

    render_status_strip(
        [
            ("当前数据集", f"{dataset_name} · {rows} 行 · {cols} 列"),
            ("后端同步状态", "已同步，可以训练" if backend_synced else "未同步或后端不可用"),
            ("当前模型版本", model_text),
        ]
    )


def render_ml_workbench_overview(
    model_label,
    df,
    backend_synced=False,
    model_status=None,
    task_text="",
    target_text="",
    feature_count=None,
    sample_count=None,
    extra_items=None,
):
    """Render the unified ML workbench overview below the top status strip."""
    sample_count = sample_count if sample_count is not None else (len(df) if df is not None else 0)
    feature_text = "未选择" if feature_count is None else f"{feature_count} 个"
    data_items = [
        ("样本数", f"{sample_count} 行"),
        ("特征数", feature_text),
        ("目标/任务", target_text or task_text or "待选择"),
    ]

    has_model = bool(model_status and model_status.get("has_model"))
    if has_model:
        version_id = model_status.get("version_id", "")
        created = (model_status.get("created_at", "") or "")[:16].replace("T", " ")
        metrics = model_status.get("metrics", {}) or {}
        metric_text = "、".join(
            f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
            for k, v in metrics.items()
        ) or "暂无指标"
        model_items = [
            ("状态", "已加载"),
            ("版本", version_id[:20] or "未知"),
            ("指标", metric_text),
            ("创建时间", created or "未知"),
        ]
    else:
        model_items = [
            ("状态", "未训练"),
            ("版本", "暂无"),
            ("指标", "训练后显示"),
            ("创建时间", "暂无"),
        ]

    train_items = [
        ("后端", "已同步" if backend_synced else "未同步/不可用"),
        ("训练状态", "可训练" if backend_synced else "需先同步后端"),
    ]
    if extra_items:
        train_items.extend(extra_items)

    col_data, col_train, col_model = st.columns(3, gap="medium")
    with col_data:
        render_kv_panel("训练数据", data_items, "当前页面将使用的数据与任务。", tone="data")
    with col_train:
        render_kv_panel("训练准备", train_items, "后端、参数和风险提示会影响训练能否完成。", tone="train")
    with col_model:
        render_kv_panel(f"{model_label}版本", model_items, "当前激活模型会用于预测区。", tone="model")


def render_risk_notice(title, detail, level="warning"):
    """Render risk hints in one consistent format."""
    render_notice(f"风险提示｜{title}", detail, level=level)


def render_small_dataset_warning(n_samples, threshold=50):
    if n_samples < threshold:
        render_risk_notice(
            "样本量偏小",
            f"当前只有 {n_samples} 行有效数据，训练指标波动会较大，建议补充数据或优先使用简单模型。",
        )


def render_class_balance_warning(series):
    counts = series.value_counts(dropna=True)
    if len(counts) < 2:
        return
    min_count = int(counts.min())
    max_count = int(counts.max())
    if min_count < 3 or (max_count and min_count / max_count < 0.2):
        render_risk_notice(
            "类别不均衡",
            f"最少类别 {min_count} 条，最多类别 {max_count} 条，分类结果可能偏向多数类。",
        )


# ── Device selector ──

def render_device_selector():
    """Render GPU/CPU device selection UI. Returns torch.device."""
    st.subheader("⚙️ 计算设备选择")
    device_choice = st.selectbox("选择运行设备", ["CPU", "CUDA (GPU)"], index=0)
    if torch is None:
        st.warning("当前环境未安装 PyTorch，训练页面需要安装依赖后才能训练；已按 CPU 展示页面。")
        return "cpu"
    if device_choice == "CUDA (GPU)":
        if not torch.cuda.is_available():
            st.warning("当前环境未安装 CUDA/GPU，已自动切换为 CPU。如需使用 GPU，请安装 CUDA 版 PyTorch。")
            return torch.device("cpu")
        device = torch.device("cuda")
        st.success("✅ 已启用：CUDA GPU")
    else:
        device = torch.device("cpu")
        st.info("⚙️ 已启用：CPU")
    return device


# ── Input validation ──

def validate_input_array(arr, context="", expected_features=None):
    """Return a user-friendly error message for prediction input arrays."""
    prefix = f"{context}: " if context else ""

    try:
        raw = np.asarray(arr)
    except Exception:
        return f"⚠️ {prefix}输入数据无法读取，请检查格式。"

    if raw.size == 0:
        return f"⚠️ {prefix}输入为空，请先填写特征值。"

    if raw.ndim == 1:
        n_features = raw.shape[0]
    elif raw.ndim == 2:
        n_features = raw.shape[1]
    else:
        return f"⚠️ {prefix}输入维度不正确，请使用一行或二维表格数据。"

    if expected_features is not None and n_features != expected_features:
        return (
            f"⚠️ {prefix}特征数量不匹配：模型需要 {expected_features} 个特征，"
            f"当前输入为 {n_features} 个。"
        )

    try:
        values = raw.astype(float)
    except (TypeError, ValueError):
        return f"⚠️ {prefix}输入包含非数值内容，请先转换为数值或在数据处理页完成编码。"

    if np.isnan(values).any():
        return f"⚠️ {prefix}输入包含空值（NaN），请补全后再预测。"
    if np.isinf(values).any():
        return f"⚠️ {prefix}输入包含无穷值（Inf），请清洗后再预测。"
    return None


def render_task_mismatch_warning(model_label, selected_task, model_task):
    """Warn when the loaded model task differs from the current page setting."""
    if not selected_task or not model_task or selected_task == model_task:
        return False
    task_names = {"classification": "分类", "regression": "回归"}
    st.warning(
        f"⚠️ 当前页面选择的是{task_names.get(selected_task, selected_task)}任务，"
        f"但已加载的{model_label}模型是{task_names.get(model_task, model_task)}任务。"
        "预测区会按已加载模型的任务类型执行；如果要训练当前任务，请点击重新训练。"
    )
    return True


def task_mismatch_message(model_label, selected_task, model_task):
    """Return a warning string when selected task and loaded model task differ."""
    if not selected_task or not model_task or selected_task == model_task:
        return ""
    task_names = {"classification": "分类", "regression": "回归"}
    return (
        f"当前页面选择的是{task_names.get(selected_task, selected_task)}任务，"
        f"但已加载的{model_label}模型是{task_names.get(model_task, model_task)}任务。"
        "预测区会按已加载模型任务执行；若要使用当前任务，请重新训练。"
    )


def model_metric_items(model_status=None, fallback_result=None):
    """Return readable model metric items for prediction panels."""
    metrics = {}
    if model_status:
        metrics.update(model_status.get("metrics", {}) or {})
    if fallback_result:
        for key in ("r2", "mae", "rmse", "acc", "silhouette", "inertia"):
            if key in fallback_result and fallback_result[key] is not None:
                metrics.setdefault(key, fallback_result[key])
    names = {
        "r2": "R²",
        "mae": "MAE",
        "rmse": "RMSE",
        "acc": "准确率",
        "silhouette": "轮廓系数",
        "inertia": "惯性值",
        "n_clusters": "簇数量",
    }
    items = []
    for key, value in metrics.items():
        label = names.get(key, key)
        if isinstance(value, float):
            if abs(value) >= 1000:
                text = f"{value:,.2f}"
            else:
                text = f"{value:.4f}"
        else:
            text = value
        items.append((label, text))
    return items


def probability_rows(probabilities, labels=None):
    """Build a probability table payload from model probabilities."""
    if probabilities is None:
        return []
    probabilities = list(probabilities)
    if not probabilities:
        return []
    labels = labels or [str(i) for i in range(len(probabilities))]
    rows = []
    for label, prob in zip(labels, probabilities):
        try:
            prob_f = float(prob)
        except (TypeError, ValueError):
            continue
        rows.append({"类别": str(label), "概率": prob_f})
    rows.sort(key=lambda item: item["概率"], reverse=True)
    return rows


def render_probability_table(rows):
    """Render class probability rows if available."""
    if not rows:
        return
    display = pd.DataFrame(rows)
    display["概率"] = display["概率"].map(lambda v: f"{v:.4f}")
    st.caption("类别概率")
    st.dataframe(display, use_container_width=True, hide_index=True)


def render_stable_prediction_panel(
    title,
    main_value,
    details=None,
    model_status=None,
    fallback_result=None,
    probabilities=None,
    probability_labels=None,
    mismatch_message="",
    description="最近一次单条预测结果会保留在页面内。",
):
    """Render a complete stable prediction result block."""
    if mismatch_message:
        render_notice("版本/任务不匹配", mismatch_message, level="warning")
    merged_details = list(details or [])
    merged_details.extend(model_metric_items(model_status, fallback_result))
    render_prediction_panel(title, main_value, merged_details, description=description)
    render_probability_table(probability_rows(probabilities, probability_labels))


def prepare_batch_prediction(file_obj, required_features, context="批量预测", max_rows=10000):
    """Read and validate a batch prediction CSV before calling backend."""
    if file_obj is None:
        return None, None, None

    try:
        batch_df = pd.read_csv(file_obj)
    except Exception:
        return None, None, f"⚠️ {context}: CSV 文件读取失败，请确认文件编码、分隔符和内容格式。"

    if batch_df.empty:
        return batch_df, None, f"⚠️ {context}: CSV 文件为空，请上传至少 1 行数据。"
    if len(batch_df) > max_rows:
        return (
            batch_df,
            None,
            f"⚠️ {context}: 单次最多支持 {max_rows} 行，当前为 {len(batch_df)} 行。请拆分文件后重试。",
        )

    duplicated = batch_df.columns[batch_df.columns.duplicated()].tolist()
    if duplicated:
        names = "、".join(str(c) for c in duplicated[:8])
        return batch_df, None, f"⚠️ {context}: CSV 包含重复列名：{names}。请先重命名后再预测。"

    required = [str(c) for c in required_features]
    batch_df.columns = [str(c) for c in batch_df.columns]
    missing_cols = [c for c in required if c not in batch_df.columns]
    if missing_cols:
        names = "、".join(missing_cols[:12])
        return batch_df, None, f"⚠️ {context}: 缺少模型需要的特征列：{names}。"

    batch_x = batch_df[required].values
    err = validate_input_array(batch_x, context, expected_features=len(required))
    if err:
        return batch_df, None, err
    return batch_df, batch_x, None


def render_batch_prediction_output(result_df, filename="predictions.csv"):
    """Render a stable batch prediction table and download action."""
    st.caption(f"批量预测完成：{len(result_df)} 行")
    st.dataframe(result_df, use_container_width=True)
    csv = result_df.to_csv(index=False).encode("utf-8-sig")
    st.download_button("📥 下载预测结果", csv, filename, "text/csv", use_container_width=True)


def cluster_explanation(cluster_label, cluster_result=None):
    """Return a short explanation for a predicted cluster."""
    if cluster_result:
        counts = cluster_result.get("cluster_counts", {}) or {}
        count = counts.get(cluster_label, counts.get(str(cluster_label)))
        if count is not None:
            return f"该样本被分配到簇 {cluster_label}，训练集中该簇包含 {count} 条样本。"
    return f"该样本被分配到簇 {cluster_label}。簇编号只表示分组，不代表好坏或大小顺序。"


# ── Constant feature detection ──

def check_constant_features(df, feature_cols):
    """Warn if any feature column has zero variance (constant value)."""
    numeric_cols = [c for c in feature_cols if c in df.select_dtypes(include=[np.number]).columns]
    bad = []
    for col in numeric_cols:
        vals = df[col].dropna()
        if len(vals) < 2 or vals.nunique() <= 1:
            bad.append((col, vals.iloc[0] if len(vals) > 0 else "NaN"))
    if bad:
        names = ", ".join(f"「{c}」" for c, _ in bad)
        render_risk_notice("常量特征", f"{names} 方差为 0，建议在数据处理页移除。")
    return bad


# ── Loss curve ──

def plot_loss_curve(train_losses, val_losses, y_label="损失值", title="训练 & 验证损失曲线"):
    """Render a Plotly train/val loss curve."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=train_losses, mode='lines', name='训练损失', line=dict(color='#1f77b4')))
    fig.add_trace(go.Scatter(y=val_losses, mode='lines', name='验证损失', line=dict(color='#ff7f0e')))
    fig.update_layout(
        title=title, xaxis_title="训练轮次", yaxis_title=y_label,
        template="plotly_white", height=350, margin=dict(l=0, r=0, t=40, b=0)
    )
    st.plotly_chart(fig, use_container_width=True)


# ── Model version selector ──

def render_version_selector(model_label, list_fn, activate_fn, delete_fn, prediction_keys=None):
    """Render a model version selector UI.

    Returns the active version_id (str or None). Call after training or on page load.
    Side effects: calls activate_fn when user picks a version; deletes on user request.
    """
    data = list_fn()
    if not data:
        st.caption("暂无已保存的模型版本。")
        return None

    versions = data.get("versions", [])
    active = data.get("active")

    if not versions:
        st.caption("暂无已保存的模型版本。")
        return None

    prediction_keys = prediction_keys or []
    with st.expander(f"📦 {model_label} 模型版本管理（共 {len(versions)} 个版本）", expanded=False):
        # Build version options
        options = []
        vid_to_idx = {}
        vid_to_meta = {}
        for i, v in enumerate(versions):
            vid = v.get("version_id", "")
            ds = v.get("dataset_name", "未知数据集")
            created = v.get("created_at", "")[:16].replace("T", " ")
            target = v.get("target", "")
            metrics = v.get("metrics", {})
            metric_str = ", ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
                                   for k, v in metrics.items()) or "暂无指标"
            active_mark = " ★" if vid == active else ""
            label = f"{ds} | {target or '无目标列'} | {metric_str} | {created}{active_mark}"
            options.append(label)
            vid_to_idx[label] = vid
            vid_to_meta[vid] = v

        col1, col2, col3 = st.columns([5, 1, 1])
        with col1:
            default_idx = 0
            for label, vid in vid_to_idx.items():
                if vid == active:
                    default_idx = options.index(label)
                    break
            selected_label = st.selectbox(
                f"选择{model_label}版本",
                options, index=default_idx,
                key=f"version_selector_{model_label}",
                label_visibility="collapsed"
            )
            selected_vid = vid_to_idx.get(selected_label)

        selected_meta = vid_to_meta.get(selected_vid, {}) if selected_vid else {}
        selected_metrics = selected_meta.get("metrics", {}) or {}
        metric_text = "、".join(
            f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
            for k, v in selected_metrics.items()
        ) or "暂无指标"
        selected_params = selected_meta.get("params", {}) or {}
        task_hint = selected_params.get("task_type") or selected_params.get("task") or selected_params.get("algorithm") or "默认"
        recommendation = "当前激活版本" if selected_vid == active else "可切换版本"
        if versions and selected_vid == versions[0].get("version_id"):
            recommendation += " / 最新版本"
        render_kv_panel(
            "选中版本详情",
            [
                ("推荐状态", recommendation),
                ("数据集", selected_meta.get("dataset_name", "未知数据集")),
                ("目标/任务", selected_meta.get("target") or task_hint),
                ("指标", metric_text),
                ("创建时间", (selected_meta.get("created_at", "") or "")[:16].replace("T", " ") or "未知"),
            ],
            "切换版本后会清空当前页面的旧预测结果，避免版本和结果不一致。",
            tone="model",
        )

        with col2:
            if st.button("✅ 激活", key=f"activate_{model_label}", use_container_width=True):
                if selected_vid and selected_vid != active:
                    activate_fn(selected_vid)
                    for key in prediction_keys:
                        st.session_state.pop(key, None)
                    st.toast(f"已激活版本 {selected_vid[:20]}...", icon="✅")
                    st.rerun()
                else:
                    st.toast("该版本已经是当前激活版本。", icon="ℹ️")

        with col3:
            if selected_vid and len(versions) > 1:
                confirm_delete = st.checkbox(
                    "确认删除",
                    key=f"confirm_delete_{model_label}_{selected_vid}",
                    help="删除后会移除该版本记录和文件。",
                )
                if st.button("🗑️ 删除", key=f"delete_{model_label}", use_container_width=True):
                    if not confirm_delete:
                        st.toast("请先勾选“确认删除”。", icon="⚠️")
                    else:
                        delete_fn(selected_vid)
                        for key in prediction_keys:
                            st.session_state.pop(key, None)
                        st.warning(f"已删除版本 {selected_vid[:20]}...")
                        st.rerun()

    return active
