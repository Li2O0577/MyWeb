"""Data processing page — local operations with optional backend sync."""
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder, OneHotEncoder
from sklearn.decomposition import PCA
from pages._prepare import render_sidebar, data_uploader

st.set_page_config(page_title="数据处理", layout="wide", initial_sidebar_state="collapsed")
st.markdown("""<style>[data-testid="stSidebarNav"] {display: none;}</style>""", unsafe_allow_html=True)
render_sidebar("pages/3_data_processing.py")
st.title("🧹 数据处理")

# ── Version counter — incremented on undo/reset so all column-dependent widgets get fresh keys ──
if "_processing_version" not in st.session_state:
    st.session_state._processing_version = 0
_ver = st.session_state._processing_version

def _k(name):
    """Return a versioned widget key."""
    return f"{name}_{_ver}"

# ── Load data ──
df_source = data_uploader(upload_to_backend=False)

if "original_df" not in st.session_state:
    st.session_state.original_df = None
if df_source is not None:
    current_source = st.session_state.get("_source_file_hash") or st.session_state.get("_source_file", None)
    if st.session_state.get("_3_snapshot_source") != current_source:
        st.session_state.original_df = df_source.copy()
        st.session_state._3_snapshot_source = current_source

if df_source is not None:
    if st.session_state.get("_3_working_source") != current_source or "_page3_working_df" not in st.session_state:
        st.session_state._page3_working_df = df_source.copy()
        st.session_state._3_working_source = current_source
    df = st.session_state._page3_working_df
else:
    df = None

if df is None:
    st.warning("请先在数据加载页面上传数据！")
    st.stop()

# Auto-detect column changes (PCA, encoding, undo, reset) and bump version
_col_sig = hash(tuple(df.columns))
if "_3_col_sig" not in st.session_state:
    st.session_state._3_col_sig = _col_sig
elif st.session_state._3_col_sig != _col_sig:
    st.session_state._3_col_sig = _col_sig
    st.session_state._processing_version += 1
    st.rerun()

# ═══════════════════════════════════════════════
# 1. 基础行列操作
# ═══════════════════════════════════════════════
with st.expander("1. 基础行列操作", expanded=True):
    st.subheader("行列筛选 / 增删 / 重命名")
    col1, col2 = st.columns(2)
    with col1:
        select_cols = st.multiselect("保留指定列", list(df.columns), default=list(df.columns), key=_k("select_cols"))
        df = df[select_cols]
        drop_cols = st.multiselect("删除指定列", list(df.columns), key=_k("drop_cols"))
        if drop_cols:
            df = df.drop(columns=drop_cols)
    with col2:
        if len(df.columns) == 0:
            st.warning("当前没有可操作的列。")
        else:
            rename_col = st.selectbox("选择要重命名的列", list(df.columns), key=_k("rename_col"))
            new_name = st.text_input("新列名", value=rename_col, key=_k("new_name"))
            if st.button("确认修改列名", key=_k("btn_rename")):
                df.rename(columns={rename_col: new_name}, inplace=True)
        if len(df) == 0:
            st.warning("当前没有可删除的行。")
        else:
            del_row = st.number_input("删除指定行号", min_value=0, max_value=len(df)-1, value=0, key=_k("del_row"))
            if st.button("删除该行", key=_k("btn_del_row")):
                df = df.drop(index=df.index[del_row])

if df.shape[1] == 0 or len(df) == 0:
    st.divider()
    st.warning("当前工作数据为空，请重置为原始数据或重新选择保留列。")
    st.dataframe(df, use_container_width=True)
    c_save, c_reset = st.columns(2)
    with c_save:
        if st.button("保存空数据状态", use_container_width=True, key=_k("btn_save_empty")):
            st.session_state.main_df = df.copy()
            st.session_state._data_cleaned = True
            st.success("已保存当前空数据状态。")
    with c_reset:
        if st.button("重置为原始数据", use_container_width=True, key=_k("btn_reset_empty")):
            if st.session_state.original_df is not None:
                st.session_state._page3_working_df = st.session_state.original_df.copy()
                st.session_state.main_df = st.session_state.original_df.copy()
                st.session_state._data_cleaned = True
                st.rerun()
            else:
                st.warning("暂无原始数据！请先上传数据后再重置。")
    st.session_state._page3_working_df = df.copy()
    st.stop()

# ═══════════════════════════════════════════════
# 2. 数据类型修改
# ═══════════════════════════════════════════════
with st.expander("2. 数据类型修改"):
    st.subheader("转换列的数据类型")
    type_col = st.selectbox("选择列", list(df.columns), key=_k("type_col"))
    type_choice = st.selectbox("目标类型", ["int", "float", "str", "datetime"], key=_k("type_choice"))
    if st.button("转换类型", key=_k("btn_type")):
        try:
            if type_choice == "datetime":
                df[type_col] = pd.to_datetime(df[type_col], errors="coerce")
            else:
                df[type_col] = df[type_col].astype(type_choice)
            st.success("转换成功！")
        except Exception:
            st.error("转换失败（数据不兼容）")

# ═══════════════════════════════════════════════
# 3. 统计指标计算
# ═══════════════════════════════════════════════
with st.expander("3. 统计指标计算"):
    st.subheader("数值列统计量")
    numeric_cols = list(df.select_dtypes(include=[np.number]).columns)
    if len(numeric_cols) == 0:
        st.warning("当前无数值列！")
    else:
        stat_col = st.selectbox("选择数值列", numeric_cols, key=_k("stat_col"))
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            st.metric("均值", round(df[stat_col].mean(), 2))
            st.metric("中位数", round(df[stat_col].median(), 2))
        with col_b:
            st.metric("方差", round(df[stat_col].var(), 2))
            st.metric("标准差", round(df[stat_col].std(), 2))
        with col_c:
            st.metric("最大值", df[stat_col].max())
            st.metric("最小值", df[stat_col].min())

# ═══════════════════════════════════════════════
# 4. 标准化 / 归一化
# ═══════════════════════════════════════════════
with st.expander("4. 标准化 / 归一化"):
    st.subheader("数值缩放")
    scale_cols = st.multiselect("选择数值列", list(df.select_dtypes(include=[np.number]).columns), key=_k("scale_cols"))
    method = st.radio("方法", ["StandardScaler (标准化)", "MinMaxScaler (归一化)"], key=_k("scale_method"))
    if st.button("执行缩放", key=_k("btn_scale")) and scale_cols:
        scaler = StandardScaler() if "Standard" in method else MinMaxScaler()
        df[scale_cols] = scaler.fit_transform(df[scale_cols])
        st.success("处理完成！")

# ═══════════════════════════════════════════════
# 5. 添加数据噪声
# ═══════════════════════════════════════════════
with st.expander("5. 添加数据噪声"):
    st.subheader("为数值列添加高斯噪声")
    numeric_cols = list(df.select_dtypes(include=[np.number]).columns)
    if len(numeric_cols) == 0:
        st.warning("当前无数值列！")
    else:
        noise_col = st.selectbox("选择列", numeric_cols, key=_k("noise_col"))
        noise_level = st.slider("噪声强度（标准差）", 0.01, 0.5, 0.1, key=_k("noise_level"))
        if st.button("添加噪声", key=_k("btn_noise")):
            noise = np.random.normal(0, noise_level, size=df[noise_col].shape)
            df[noise_col] = df[noise_col] + noise
            st.success("噪声添加完成！")

# ═══════════════════════════════════════════════
# 6. 类别特征编码
# ═══════════════════════════════════════════════
with st.expander("6. 类别特征编码"):
    st.subheader("文字 → 数字")
    cat_cols = list(df.select_dtypes(exclude=[np.number]).columns)
    if len(cat_cols) == 0:
        st.warning("当前无类别列！")
    else:
        cat_col = st.selectbox("选择类别列", cat_cols, key=_k("cat_col"))
        encode_method = st.radio("编码方式", ["标签编码", "独热编码"], key=_k("encode_method"))
        if st.button("执行编码", key=_k("btn_encode")):
            if encode_method == "标签编码":
                df[cat_col] = LabelEncoder().fit_transform(df[cat_col])
            else:
                ohe = OneHotEncoder(sparse_output=False, drop="first")
                new_cols = ohe.fit_transform(df[[cat_col]])
                feature_names = ohe.get_feature_names_out([cat_col])
                new_df = pd.DataFrame(new_cols, columns=feature_names, index=df.index)
                df = pd.concat([df.drop(columns=[cat_col]), new_df], axis=1)
            st.success("编码完成！")

# ═══════════════════════════════════════════════
# 7. 自定义计算列
# ═══════════════════════════════════════════════
with st.expander("7. 自定义计算列"):
    st.subheader("添加新列（公式计算）")
    numeric_cols = list(df.select_dtypes(include=[np.number]).columns)
    if len(numeric_cols) == 0:
        st.warning("当前无数值列！")
    else:
        calc_col = st.selectbox("参考列", numeric_cols, key=_k("calc_col"))
        calc_method = st.selectbox("计算方式", ["平方", "开方", "取对数", "+10", "*3"], key=_k("calc_method"))
        new_col_name = st.text_input("新列名", value="new_column", key=_k("calc_new_col"))
        if st.button("生成计算列", key=_k("btn_calc")):
            if calc_method == "平方": df[new_col_name] = df[calc_col] ** 2
            elif calc_method == "开方": df[new_col_name] = np.sqrt(df[calc_col])
            elif calc_method == "取对数": df[new_col_name] = np.log(df[calc_col] + 1e-6)
            elif calc_method == "+10": df[new_col_name] = df[calc_col] + 10
            elif calc_method == "*3": df[new_col_name] = df[calc_col] * 3
            st.success("新列已添加！")

# ═══════════════════════════════════════════════
# 8. PCA 降维
# ═══════════════════════════════════════════════
with st.expander("8. PCA 降维"):
    st.subheader("主成分分析降维")
    numeric_cols = list(df.select_dtypes(include=[np.number]).columns)
    pca_candidate_cols = [col for col in numeric_cols if not str(col).startswith("PCA_")]
    if len(pca_candidate_cols) < 2:
        st.warning("需要至少 2 个非 PCA 数值列才能降维")
    else:
        pca_cols = st.multiselect("用于降维的数值列", pca_candidate_cols, default=pca_candidate_cols, key=_k("pca_cols"))
        max_dim = len(pca_cols) if pca_cols else 1
        n_components = st.slider("降维维度", 1, max_dim, min(2, max_dim), key=_k("pca_dim"))
        if st.button("执行PCA", key=_k("btn_pca")) and len(pca_cols) >= n_components and len(pca_cols) >= 2:
            pca_old_cols = [col for col in df.columns if str(col).startswith("PCA_")]
            if pca_old_cols:
                df = df.drop(columns=pca_old_cols)
                st.info("已清理旧 PCA 列")
            pca = PCA(n_components=n_components)
            pca_result = pca.fit_transform(df[pca_cols])
            pca_df = pd.DataFrame(pca_result, columns=[f"PCA_{i+1}" for i in range(n_components)], index=df.index)
            df = pd.concat([df.drop(columns=pca_cols), pca_df], axis=1)
            st.success(f"降维至 {n_components} 维完成！")

# ═══════════════════════════════════════════════
# Preview + Actions
# ═══════════════════════════════════════════════
st.divider()
st.subheader("处理后数据预览")
st.dataframe(df, use_container_width=True)

st.divider()
st.subheader("操作控制")
ctrl1, ctrl2, ctrl3, ctrl4 = st.columns(4)

with ctrl1:
    if st.button("保存修改", use_container_width=True, key=_k("btn_save")):
        st.session_state.main_df = df.copy()
        st.session_state._data_cleaned = True
        from pages._api import sync_session_data, try_upload_backend
        sid = st.session_state.get("session_id")
        if sid:
            ok = sync_session_data(sid, df)
        else:
            import io
            csv_bytes = df.to_csv(index=False).encode('utf-8')
            ok = try_upload_backend(csv_bytes, "processed_data.csv")
        if ok:
            st.success("已保存！修改不会因切换页面而丢失。")
        else:
            st.warning("数据已保存到前端，但后端同步失败。ML 训练可能使用旧数据。")

with ctrl2:
    if st.button("重置为原始数据", use_container_width=True, key=_k("btn_reset")):
        if st.session_state.original_df is not None:
            st.session_state._page3_working_df = st.session_state.original_df.copy()
            st.session_state.main_df = st.session_state.original_df.copy()
            st.session_state._data_cleaned = True
            st.rerun()
        else:
            st.warning("暂无原始数据！请先上传数据后再重置。")

with ctrl3:
    if st.button("撤销操作", use_container_width=True, key=_k("btn_undo")):
        if "main_df" in st.session_state and st.session_state.main_df is not None:
            st.session_state._page3_working_df = st.session_state.main_df.copy()
            st.rerun()

with ctrl4:
    st.download_button(label="导出CSV", data=df.to_csv(index=False).encode("utf-8"), file_name="processed_data.csv", mime="text/csv", use_container_width=True, key=_k("btn_export"))

st.session_state._page3_working_df = df.copy()
