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
            st.error("❌ 设备选择错误：当前环境未安装 CUDA/GPU，无法使用 CUDA！请切换为 CPU")
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


def _to_python_type(val):
    """Convert numpy scalar to native Python type for JSON serialization."""
    if isinstance(val, (np.integer,)):
        return int(val)
    if isinstance(val, (np.floating,)):
        return float(val)
    if isinstance(val, (np.bool_,)):
        return bool(val)
    return val


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
