"""Shared utilities for MLP-based pages (regression, classification, DIY)."""
import streamlit as st
import torch
import os
import pickle
import json
import numpy as np
import plotly.graph_objects as go


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

def validate_input_array(arr, context=""):
    """Return error message if array contains NaN/Inf, else None."""
    if not np.isfinite(arr).all():
        prefix = f"{context}: " if context else ""
        return f"⚠️ {prefix}输入包含无效值（NaN 或 Inf），请检查输入数据。"
    return None


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
        st.warning(f"⚠️ 检测到常数列（方差为 0）：{names}。StandardScaler 对这些列无效，建议在数据处理页移除。")
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

