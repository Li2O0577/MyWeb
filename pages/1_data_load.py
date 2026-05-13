import streamlit as st
import pandas as pd
from pages._prepare import data_uploader, render_sidebar, detect_outliers

# 1. 配置页面
st.set_page_config(page_title="Data Load", layout="wide", initial_sidebar_state="collapsed")
st.markdown("""
    <style>
        [data-testid="stSidebarNav"] {display: none;}
    </style>
""", unsafe_allow_html=True)
render_sidebar("pages/1_data_load.py")
st.title("📊 Data Load")

# 2. 数据上传（修复过的数据用持久标记跳过文件重读）
df = data_uploader(
    warn_outliers=False,
    force_cached=st.session_state.get("_data_cleaned", False),
)

# 3. 异常值详情与修复
if df is not None:
    outlier_info = st.session_state.get("outliers", {})

    if outlier_info:
        total_outliers = sum(v["count"] for v in outlier_info.values())
        n_cols = len(outlier_info)

        st.divider()
        with st.expander(
            f"⚠️ Outlier Detection — {total_outliers} outliers across {n_cols} column(s)",
            expanded=True,
        ):
            st.markdown(
                f"Detected using **IQR method** (Q1 − 1.5×IQR, Q3 + 1.5×IQR). "
                f"**{total_outliers}** outlier value(s) found in **{n_cols}** numeric column(s)."
            )

            for idx, (col, info) in enumerate(outlier_info.items()):
                col_key = f"col{idx}"

                st.markdown(f"### 📌 Column: `{col}`")
                st.caption(
                    f"{info['count']} outliers · "
                    f"Bounds: [{info['lower_bound']}, {info['upper_bound']}]"
                )

                # 异常行表格
                outlier_df = pd.DataFrame(
                    {"Row Index": info["indices"], "Value": info["values"]}
                )
                st.dataframe(outlier_df, use_container_width=True)

                # 修复按钮（均值用非异常值计算，避免被异常值污染）
                c1, c2, c3, c4 = st.columns(4)
                mask = (df[col] < info["lower_bound"]) | (
                    df[col] > info["upper_bound"]
                )
                clean_vals = df.loc[~mask, col]
                mean_val = clean_vals.mean() if len(clean_vals) > 0 else df[col].mean()
                median_val = df[col].median()

                with c1:
                    if st.button(
                        "🔒 Winsorize (clamp to bounds)",
                        key=f"win_{col_key}",
                        use_container_width=True,
                    ):
                        df[col] = df[col].clip(
                            lower=info["lower_bound"], upper=info["upper_bound"]
                        )
                        st.session_state.main_df = df
                        st.session_state._data_cleaned = True
                        st.rerun()

                with c2:
                    if st.button(
                        f"📊 Replace with Mean ({mean_val:.2f})",
                        key=f"mean_{col_key}",
                        use_container_width=True,
                    ):
                        df.loc[mask, col] = mean_val
                        st.session_state.main_df = df
                        st.session_state._data_cleaned = True
                        st.rerun()

                with c3:
                    if st.button(
                        f"📈 Replace with Median ({median_val:.2f})",
                        key=f"med_{col_key}",
                        use_container_width=True,
                    ):
                        df.loc[mask, col] = median_val
                        st.session_state.main_df = df
                        st.session_state._data_cleaned = True
                        st.rerun()

                with c4:
                    if st.button(
                        f"🗑️ Drop {info['count']} row(s)",
                        key=f"drop_{col_key}",
                        use_container_width=True,
                    ):
                        df = df.drop(index=df.index[mask])
                        st.session_state.main_df = df
                        st.session_state._data_cleaned = True
                        st.rerun()

                st.divider()

    # 4. 数据预览
    st.subheader("📋 Data Preview")
    st.dataframe(df, use_container_width=True)

# 5. 状态说明
st.divider()
if df is None:
    st.info("请先上传数据，才能进行后续的可视化、处理和建模等操作（支持 CSV 和 Excel）。")
else:
    st.success("数据加载成功！你现在可以导航到其他页面进行可视化、处理或建模。")
