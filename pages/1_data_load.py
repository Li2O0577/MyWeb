import streamlit as st
import pandas as pd
from pages._prepare import data_uploader, render_sidebar, detect_outliers

st.set_page_config(page_title="数据加载", layout="wide", initial_sidebar_state="collapsed")
st.markdown("""<style>[data-testid="stSidebarNav"] {display: none;}</style>""", unsafe_allow_html=True)
render_sidebar("pages/1_data_load.py")
st.title("📊 数据加载")

df = data_uploader(warn_outliers=False, force_cached=st.session_state.get("_data_cleaned", False))

if df is not None:
    if len(df) == 0:
        st.warning("⚠️ 当前数据为空（所有行已被删除）。请重新上传数据。")
        st.stop()
    outlier_info = st.session_state.get("outliers", {})

    if outlier_info:
        total_outliers = sum(v["count"] for v in outlier_info.values())
        n_cols = len(outlier_info)
        st.divider()
        with st.expander(f"⚠️ 异常值检测 — {total_outliers} 个异常值分布在 {n_cols} 列", expanded=True):
            st.markdown(f"使用 **IQR 方法** (Q1 − 1.5×IQR, Q3 + 1.5×IQR) 检测。在 **{n_cols}** 个数值列中发现 **{total_outliers}** 个异常值。")

            for idx, (col, info) in enumerate(outlier_info.items()):
                col_key = f"col{idx}"
                st.markdown(f"### 📌 Column: `{col}`")
                st.caption(f"{info['count']} 个异常值 · 边界: [{info['lower_bound']}, {info['upper_bound']}]")
                outlier_df = pd.DataFrame({"行号": info["indices"], "异常值": info["values"]})
                st.dataframe(outlier_df, use_container_width=True)

                c1, c2, c3, c4 = st.columns(4)
                mask = (df[col] < info["lower_bound"]) | (df[col] > info["upper_bound"])
                clean_vals = df.loc[~mask, col]
                mean_val = clean_vals.mean() if len(clean_vals) > 0 else df[col].mean()
                median_val = df[col].median()

                with c1:
                    if st.button("🔒 缩尾处理（截断到边界）", key=f"win_{col_key}", use_container_width=True):
                        df[col] = df[col].clip(lower=info["lower_bound"], upper=info["upper_bound"])
                        st.session_state.main_df = df
                        st.session_state._data_cleaned = True
                        st.rerun()
                with c2:
                    if st.button(f"📊 替换为均值 ({mean_val:.2f})", key=f"mean_{col_key}", use_container_width=True):
                        df.loc[mask, col] = mean_val
                        st.session_state.main_df = df
                        st.session_state._data_cleaned = True
                        st.rerun()
                with c3:
                    if st.button(f"📈 替换为中位数 ({median_val:.2f})", key=f"med_{col_key}", use_container_width=True):
                        df.loc[mask, col] = median_val
                        st.session_state.main_df = df
                        st.session_state._data_cleaned = True
                        st.rerun()
                with c4:
                    if st.button(f"🗑️ 删除 {info['count']} 行", key=f"drop_{col_key}", use_container_width=True):
                        df = df.drop(index=df.index[mask])
                        st.session_state.main_df = df
                        st.session_state._data_cleaned = True
                        st.rerun()
                st.divider()

    st.subheader("📋 数据预览")
    st.dataframe(df, use_container_width=True)

st.divider()
if df is None:
    st.info("请先上传数据，才能进行后续的可视化、处理和建模等操作（支持 CSV 和 Excel）。")
else:
    if st.session_state.get("_data_cleaned", False):
        c_msg, c_btn = st.columns([3, 1])
        with c_msg:
            st.success("数据已修复。可导航到其他页面进行可视化、处理或建模。")
        with c_btn:
            if st.button("🔄 重新加载原始数据", use_container_width=True):
                st.session_state._data_cleaned = False
                st.session_state._source_file = ""
                st.rerun()
    else:
        st.success("数据加载成功！可导航到其他页面进行可视化、处理或建模。")
