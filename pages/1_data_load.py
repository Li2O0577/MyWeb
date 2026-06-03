import streamlit as st
import pandas as pd
import numpy as np
from pages._prepare import data_uploader, render_sidebar
from pages._ui_common import render_page_header, render_section_header, render_status_strip
from backend.services.data_service import detect_outliers

st.set_page_config(page_title="数据加载", layout="wide", initial_sidebar_state="collapsed")
st.markdown("""<style>[data-testid="stSidebarNav"] {display: none;}</style>""", unsafe_allow_html=True)
render_sidebar("pages/1_data_load.py")
render_page_header("数据加载", "上传 CSV/Excel 数据，检查缺失值和异常值，并保存为当前工作数据集。")


def _apply_fix(df):
    """Persist data changes and trigger re-render. Outlier re-detection is handled by _prepare.py on reload."""
    st.session_state.main_df = df
    st.session_state._data_cleaned = True
    st.session_state.pop("outliers", None)  # force re-detect on rerun
    st.rerun()


# ── Coefficient selector ──
if "outlier_coefficient" not in st.session_state:
    st.session_state.outlier_coefficient = 1.5

coeff_label = st.radio(
    "异常值检测灵敏度",
    options=[
        "1.5× IQR（标准 — 标记明显偏离的数据点）",
        "3.0× IQR（宽松 — 仅标记极端异常值）"
    ],
    index=0 if st.session_state.outlier_coefficient == 1.5 else 1,
    horizontal=True,
    help="1.5×IQR 是统计学的标准阈值。如果你的数据本身波动较大（如金融数据），可选择 3.0×IQR 减少假阳性。"
)
new_coeff = 1.5 if coeff_label.startswith("1.5") else 3.0
if st.session_state.outlier_coefficient != new_coeff:
    st.session_state.outlier_coefficient = new_coeff
    st.session_state.pop("outliers", None)  # force re-detect with new coefficient

df = data_uploader(warn_outliers=False, force_cached=st.session_state.get("_data_cleaned", False))

if df is not None:
    if len(df) == 0:
        st.warning("⚠️ 当前数据为空（所有行已被删除）。请重新上传数据。")
        st.stop()

    outlier_info = st.session_state.get("outliers", {})
    coeff = st.session_state.outlier_coefficient

    total_outliers = sum(v.get("count", 0) for v in outlier_info.values())
    total_nans = sum(v.get("nan_count", 0) for v in outlier_info.values())

    if outlier_info:
        n_cols_out = sum(1 for v in outlier_info.values() if v.get("count", 0) > 0)
        n_cols_nan = sum(1 for v in outlier_info.values() if v.get("nan_count", 0) > 0)
        n_cols_mad = sum(1 for v in outlier_info.values() if v.get("method") == "mad")

        st.divider()

        expander_title = f"⚠️ 异常值检测 — {total_outliers} 个异常值（{n_cols_out} 列）"
        if total_nans > 0:
            expander_title += f" | {total_nans} 个缺失值（{n_cols_nan} 列）"
        if n_cols_mad > 0:
            expander_title += f" | {n_cols_mad} 列使用 MAD 回退"

        with st.expander(expander_title, expanded=True):
            method_desc = f"**IQR 方法** (Q1 − {coeff}×IQR, Q3 + {coeff}×IQR)"
            if n_cols_mad > 0:
                method_desc += f"，{n_cols_mad} 列因 IQR=0 自动回退到 **MAD**（中位数绝对偏差）"
            st.markdown(method_desc)

            # ── Batch operation ──
            if total_outliers > 0 and len(outlier_info) > 1:
                st.markdown("**⚡ 批量操作**（对所有列同时执行）")
                bc1, bc2, bc3 = st.columns(3)
                with bc1:
                    if st.button("🔒 全部缩尾处理", use_container_width=True,
                                 help=f"将所有列的异常值截断到 [{coeff}×IQR 边界]，不删除任何行"):
                        for col, info in outlier_info.items():
                            if info.get("count", 0) > 0 and info["lower_bound"] is not None:
                                df[col] = df[col].clip(lower=info["lower_bound"], upper=info["upper_bound"])
                        _apply_fix(df)
                with bc2:
                    if st.button("📈 全部替换为中位数", use_container_width=True,
                                 help="将所有列的异常值替换为该列中位数"):
                        for col, info in outlier_info.items():
                            if info.get("count", 0) > 0:
                                mask = (df[col] < info["lower_bound"]) | (df[col] > info["upper_bound"])
                                if mask.any():
                                    df.loc[mask, col] = df[col].median()
                        _apply_fix(df)
                with bc3:
                    n_drop_cols = n_cols_out
                    if st.button(f"🗑️ 删除 {total_outliers} 个异常行", use_container_width=True,
                                 help=f"删除所有包含异常值的行（会影响 {n_drop_cols} 个列中的异常行）"):
                        all_outlier_indices = set()
                        for info in outlier_info.values():
                            all_outlier_indices.update(info.get("indices", []))
                        if all_outlier_indices:
                            df = df.drop(index=list(all_outlier_indices), errors="ignore")
                            df = df.reset_index(drop=True)
                            st.session_state.main_df = df
                            st.session_state._data_cleaned = True
                            st.session_state.outliers = detect_outliers(df, coefficient=coeff)
                            st.rerun()
                st.markdown("---")

            # ── Per-column detail ──
            for idx, (col, info) in enumerate(outlier_info.items()):
                nan_c = info.get("nan_count", 0)
                out_c = info.get("count", 0)
                method = info.get("method", "iqr")

                col_title = f"### 📌 `{col}`"
                if method == "mad":
                    col_title += " *(MAD 回退)*"
                st.markdown(col_title)

                # Build status line
                status_parts = []
                if out_c > 0:
                    status_parts.append(f"{out_c} 个异常值 · 边界 [{info['lower_bound']}, {info['upper_bound']}]")
                if nan_c > 0:
                    status_parts.append(f"{nan_c} 个缺失值 (NaN)")
                if not status_parts:
                    status_parts.append("无异常")
                st.caption(" | ".join(status_parts))

                # ── NaN rows ──
                if nan_c > 0:
                    nan_indices = df.index[df[col].isna()].tolist()
                    with st.expander(f"🔴 缺失值 ({nan_c} 行)", expanded=False):
                        st.markdown(f"列 `{col}` 中有 {nan_c} 个缺失值（NaN）。")
                        st.markdown(f"行号：{nan_indices[:20]}{'...' if len(nan_indices) > 20 else ''}")
                        cn1, cn2 = st.columns(2)
                        with cn1:
                            if st.button(f"📊 填充均值", key=f"nan_mean_{idx}"):
                                df[col] = df[col].fillna(df[col].mean())
                                st.session_state.main_df = df
                                st.session_state._data_cleaned = True
                                st.session_state.outliers = detect_outliers(df, coefficient=coeff)
                                st.rerun()
                        with cn2:
                            if st.button(f"📈 填充中位数", key=f"nan_med_{idx}"):
                                df[col] = df[col].fillna(df[col].median())
                                st.session_state.main_df = df
                                st.session_state._data_cleaned = True
                                st.session_state.outliers = detect_outliers(df, coefficient=coeff)
                                st.rerun()

                # ── Outlier rows ──
                if out_c > 0:
                    # Build outlier table with severity
                    sevs = info.get("severities", [])
                    dirs = info.get("directions", [])
                    rows_data = []
                    for i_out in range(len(info["indices"])):
                        row_data = {
                            "行号": info["indices"][i_out],
                            "异常值": info["values"][i_out],
                        }
                        if i_out < len(sevs):
                            sev = sevs[i_out]
                            dr = dirs[i_out] if i_out < len(dirs) else ""
                            emoji = "🔴" if sev > 3.0 else "🟡"
                            row_data["严重程度"] = f"{emoji} {sev}× {'↑ 偏高' if dr == 'high' else '↓ 偏低'}"
                        rows_data.append(row_data)

                    outlier_df = pd.DataFrame(rows_data)
                    st.dataframe(outlier_df, use_container_width=True, hide_index=True)

                    # Fix buttons
                    mask = (df[col] < info["lower_bound"]) | (df[col] > info["upper_bound"])
                    clean_vals = df.loc[~mask, col]
                    mean_val = clean_vals.mean() if len(clean_vals) > 0 else df[col].mean()
                    median_val = df[col].median()

                    c1, c2, c3, c4 = st.columns(4)
                    with c1:
                        if st.button("🔒 缩尾处理", key=f"win_{idx}", use_container_width=True,
                                     help=f"将异常值截断到 [{info['lower_bound']}, {info['upper_bound']}]"):
                            df[col] = df[col].clip(lower=info["lower_bound"], upper=info["upper_bound"])
                            st.session_state.main_df = df
                            st.session_state._data_cleaned = True
                            st.session_state.outliers = detect_outliers(df, coefficient=coeff)
                            st.rerun()
                    with c2:
                        if st.button(f"📊 替换为均值 ({mean_val:.2f})", key=f"mean_{idx}", use_container_width=True):
                            df.loc[mask, col] = mean_val
                            st.session_state.main_df = df
                            st.session_state._data_cleaned = True
                            st.session_state.outliers = detect_outliers(df, coefficient=coeff)
                            st.rerun()
                    with c3:
                        if st.button(f"📈 替换为中位数 ({median_val:.2f})", key=f"med_{idx}", use_container_width=True):
                            df.loc[mask, col] = median_val
                            st.session_state.main_df = df
                            st.session_state._data_cleaned = True
                            st.session_state.outliers = detect_outliers(df, coefficient=coeff)
                            st.rerun()
                    with c4:
                        drop_btn_label = f"🗑️ 删除 {info['count']} 行"
                        if st.button(drop_btn_label, key=f"drop_{idx}", use_container_width=True,
                                     help=f"删除列 {col} 中包含异常值的行。注意：这同时会删除这些行在其他列中的数据。"):
                            # Show cross-column impact before deleting
                            affected_indices = set(info["indices"])
                            other_cols_affected = []
                            for other_col, other_info in outlier_info.items():
                                if other_col != col:
                                    other_outliers = set(other_info.get("indices", []))
                                    overlap = len(affected_indices & other_outliers)
                                    if overlap > 0:
                                        other_cols_affected.append((other_col, overlap))
                            if other_cols_affected:
                                impact_msg = " | ".join([f"{c}: -{n} 异常" for c, n in other_cols_affected[:5]])
                                st.toast(f"删除 {len(affected_indices)} 行 → 同时影响: {impact_msg}", icon="ℹ️")

                            df = df.drop(index=list(affected_indices), errors="ignore")
                            df = df.reset_index(drop=True)
                            st.session_state.main_df = df
                            st.session_state._data_cleaned = True
                            st.session_state.outliers = detect_outliers(df, coefficient=coeff)
                            st.rerun()
                st.divider()

    # ── Data preview ──
    render_section_header("数据预览", "查看当前数据集并保存为后续页面使用的数据。")
    if total_outliers > 0 or total_nans > 0:
        c_info, c_preview = st.columns([3, 1])
        with c_info:
            parts = []
            if total_outliers > 0:
                parts.append(f"{total_outliers} 异常值")
            if total_nans > 0:
                parts.append(f"{total_nans} 缺失值")
            st.caption(" / ".join(parts))
    st.dataframe(df, use_container_width=True)
else:
    total_outliers = 0
    total_nans = 0

st.divider()
if df is None:
    st.info("请先上传数据，才能进行后续的可视化、处理和建模等操作（支持 CSV 和 Excel）。")
else:
    render_status_strip([
        ("当前数据集", f"{len(df)} 行 · {len(df.columns)} 列"),
        ("缺失值", f"{int(df.isna().sum().sum())} 个"),
        ("后端同步状态", f"session {str(st.session_state.get('session_id', ''))[:12]}" if st.session_state.get("session_id") else "尚未同步到后端"),
    ])
    if st.session_state.get("_data_cleaned", False):
        c_msg, c_btn = st.columns([3, 1])
        with c_msg:
            st.success("数据已修复。可导航到其他页面进行可视化、处理或建模。")
        with c_btn:
            if st.button("🔄 重新加载原始数据", use_container_width=True):
                st.session_state._data_cleaned = False
                st.session_state._source_file = ""
                st.session_state._source_file_hash = ""
                st.rerun()
    else:
        st.success("数据加载成功！可导航到其他页面进行可视化、处理或建模。")
