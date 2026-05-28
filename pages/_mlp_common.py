"""Shared utilities for MLP-based pages (regression, classification, DIY)."""
import streamlit as st
import torch
import os
import pickle
import json
import html
import numpy as np
import plotly.graph_objects as go


TASK_NAMES = {"classification": "分类", "regression": "回归"}


# ── Shared ML page UI ──

def render_ml_status_bar(df, backend_synced=False, model_status=None, model_label="模型"):
    """Render a compact, consistent status strip for ML pages."""
    meta = st.session_state.get("session_meta", {}) or {}
    dataset_name = html.escape(str(meta.get("source_name", "当前数据")))
    rows = meta.get("rows", len(df) if df is not None else 0)
    cols = meta.get("n_columns", len(df.columns) if df is not None else 0)

    has_model = bool(model_status and model_status.get("has_model"))
    if has_model:
        version_id = model_status.get("version_id", "")
        created = (model_status.get("created_at", "") or "")[:16].replace("T", " ")
        target = model_status.get("target", "")
        model_text = f"{html.escape(str(model_label))}：{html.escape(str(version_id[:18]))}..."
        if target:
            model_text += f" | 目标：{html.escape(str(target))}"
        if created:
            model_text += f" | {html.escape(str(created))}"
    else:
        model_text = f"{html.escape(str(model_label))}：暂无已加载版本"

    st.markdown(
        f"""
        <div style="display:grid;grid-template-columns:1.25fr 1fr 1.6fr;gap:8px;margin:8px 0 14px 0;">
          <div style="border:1px solid #d0d7de;border-radius:6px;padding:8px 10px;">
            <div style="font-size:12px;color:#6b7280;">当前数据集</div>
            <div style="font-size:14px;">{dataset_name} · {rows} 行 · {cols} 列</div>
          </div>
          <div style="border:1px solid #d0d7de;border-radius:6px;padding:8px 10px;">
            <div style="font-size:12px;color:#6b7280;">后端同步状态</div>
            <div style="font-size:14px;">{'已同步，可以训练' if backend_synced else '未同步或后端不可用'}</div>
          </div>
          <div style="border:1px solid #d0d7de;border-radius:6px;padding:8px 10px;">
            <div style="font-size:12px;color:#6b7280;">当前模型版本</div>
            <div style="font-size:14px;">{model_text}</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_risk_notice(title, detail, level="warning"):
    """Render risk hints in one consistent format."""
    text = f"⚠️ 风险提示｜{title}：{detail}"
    if level == "info":
        st.info(text)
    else:
        st.warning(text)


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
    if device_choice == "CUDA (GPU)":
        if not torch.cuda.is_available():
            st.toast("❌ 设备选择错误：当前环境未安装 CUDA/GPU，无法使用 CUDA！请切换为 CPU", icon="❌")
            st.stop()
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

def render_version_selector(model_label, list_fn, activate_fn, delete_fn):
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

    with st.expander(f"📦 {model_label} 模型版本管理（共 {len(versions)} 个版本）", expanded=False):
        # Build version options
        options = []
        vid_to_idx = {}
        for i, v in enumerate(versions):
            vid = v.get("version_id", "")
            ds = v.get("dataset_name", "未知数据集")
            created = v.get("created_at", "")[:16].replace("T", " ")
            metrics = v.get("metrics", {})
            metric_str = ", ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
                                   for k, v in metrics.items())
            active_mark = " ★" if vid == active else ""
            label = f"{ds} | {metric_str} | {created}{active_mark}"
            options.append(label)
            vid_to_idx[label] = vid

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

        with col2:
            if st.button("✅ 激活", key=f"activate_{model_label}", use_container_width=True):
                if selected_vid and selected_vid != active:
                    activate_fn(selected_vid)
                    st.toast(f"已激活版本 {selected_vid[:20]}...", icon="✅")
                    st.rerun()

        with col3:
            if selected_vid and len(versions) > 1:
                if st.button("🗑️ 删除", key=f"delete_{model_label}", use_container_width=True):
                    delete_fn(selected_vid)
                    st.warning(f"已删除版本 {selected_vid[:20]}...")
                    st.rerun()

    return active
