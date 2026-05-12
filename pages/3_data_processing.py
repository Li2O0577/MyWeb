import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder, OneHotEncoder
from sklearn.decomposition import PCA
from pages._prepare import render_sidebar, data_uploader

# 1. 配置页面，隐藏侧边栏导航 
st.set_page_config(page_title="Data processing", layout="wide", initial_sidebar_state="collapsed")
st.markdown(
"""
    <style>
        [data-testid="stSidebarNav"] {display: none;}
    </style>
""", unsafe_allow_html=True)
render_sidebar("pages/3_data_processing.py")
st.title("🧹 Data processing")

# 2. 数据上传与预处理组件
df = data_uploader()

# 初始化：保存原始数据（用于重置，避免None报错）
if "original_df" not in st.session_state:
    st.session_state.original_df = None  # 先预初始化，杜绝KeyError
# 每次上传新数据时，自动更新原始数据（确保重置的是最新上传的源数据）
if df is not None:
    if st.session_state.original_df is None or not df.equals(st.session_state.original_df):
        st.session_state.original_df = df.copy()

if df is None:
    st.warning("请先在 Data Load 页面上传数据！")
    st.stop()

# 3. 数据处理选项
with st.expander("1. 基础行列操作", expanded=True):
    st.subheader("行列筛选 / 增删 / 重命名")
    col1, col2 = st.columns(2)
    
    with col1:
        # 筛选列 + 唯一KEY
        select_cols = st.multiselect("保留指定列", df.columns, default=df.columns, key="select_cols_1")
        df = df[select_cols]
        # 删除列 + 唯一KEY
        drop_cols = st.multiselect("删除指定列", df.columns, key="drop_cols_1")
        if drop_cols:
            df = df.drop(columns=drop_cols)
    
    with col2:
        # 重命名列 + 唯一KEY
        rename_col = st.selectbox("选择要重命名的列", df.columns, key="rename_col_1")
        new_name = st.text_input("新列名", value=rename_col, key="new_name_1")
        if st.button("确认修改列名", key="btn_rename"):
            df.rename(columns={rename_col: new_name}, inplace=True)
        # 删除行 + 唯一KEY
        del_row = st.number_input("删除指定行号", min_value=0, max_value=len(df)-1, value=0, key="del_row_1")
        if st.button("删除该行", key="btn_del_row"):
            df = df.drop(index=del_row)

with st.expander("2. 数据类型修改"):
    st.subheader("转换列的数据类型")
    type_col = st.selectbox("选择列", df.columns, key="type_col_1")
    type_choice = st.selectbox("目标类型", ["int", "float", "str", "datetime"], key="type_choice_1")
    if st.button("转换类型", key="btn_type"):
        try:
            if type_choice == "datetime":
                df[type_col] = pd.to_datetime(df[type_col], errors="coerce")
            else:
                df[type_col] = df[type_col].astype(type_choice)
            st.success("转换成功！")
        except:
            st.error("转换失败（数据不兼容）")

with st.expander("3. 统计指标计算"):
    st.subheader("数值列统计量")
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) == 0:
        st.warning("当前无数值列！")
    else:
        stat_col = st.selectbox("选择数值列", numeric_cols, key="stat_col_1")
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

with st.expander("4. 标准化 / 归一化"):
    st.subheader("数值缩放")
    scale_cols = st.multiselect("选择数值列", df.select_dtypes(include=[np.number]).columns, key="scale_cols_1")
    method = st.radio("方法", ["StandardScaler (标准化)", "MinMaxScaler (归一化)"], key="scale_method_1")
    if st.button("执行缩放", key="btn_scale") and scale_cols:
        scaler = StandardScaler() if method.startswith("S") else MinMaxScaler()
        df[scale_cols] = scaler.fit_transform(df[scale_cols])
        st.success("处理完成！")

with st.expander("5. 添加数据噪声（模拟真实测量误差）"):
    st.subheader("为数值列添加高斯噪声")
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) == 0:
        st.warning("当前无数值列！")
    else:
        noise_col = st.selectbox("选择列", numeric_cols, key="noise_col_1")
        noise_level = st.slider("噪声强度（标准差）", 0.01, 0.5, 0.1, key="noise_level_1")
            
        if st.button("添加噪声", key="btn_noise"):
            noise = np.random.normal(0, noise_level, size=df[noise_col].shape)
            df[noise_col] = df[noise_col] + noise
            st.success("噪声添加完成！")

with st.expander("6. 类别特征编码"):
    st.subheader("文字 → 数字（建模必备）")
    cat_cols = df.select_dtypes(exclude=[np.number]).columns
    if len(cat_cols) == 0:
        st.warning("当前无类别列！")
    else:
        cat_col = st.selectbox("选择类别列", cat_cols, key="cat_col_1")
        encode_method = st.radio("编码方式", ["Label Encoding", "One-Hot Encoding"], key="encode_method_1")
        if st.button("执行编码", key="btn_encode"):
            if encode_method == "Label Encoding":
                df[cat_col] = LabelEncoder().fit_transform(df[cat_col])
            else:
                ohe = OneHotEncoder(sparse_output=False, drop="first")
                new_cols = ohe.fit_transform(df[[cat_col]])
                new_df = pd.DataFrame(new_cols, columns=[f"{cat_col}_{i}" for i in range(new_cols.shape[1])])
                df = pd.concat([df.drop(columns=[cat_col]), new_df], axis=1)
            st.success("编码完成！")

with st.expander("7. 自定义计算列"):
    st.subheader("添加新列（公式计算）")
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) == 0:
        st.warning("当前无数值列！")
    else:
        calc_col = st.selectbox("参考列", numeric_cols, key="calc_col_1")
        calc_method = st.selectbox("计算方式", ["平方", "开方", "取对数", "+10","*3"], key="calc_method_1")
        new_col_name = st.text_input("新列名", value="new_column", key="new_col_name_1")
        if st.button("生成计算列", key="btn_calc"):
            if calc_method == "平方":
                df[new_col_name] = df[calc_col] ** 2
            elif calc_method == "开方":
                df[new_col_name] = np.sqrt(df[calc_col])
            elif calc_method == "取对数":
                df[new_col_name] = np.log(df[calc_col] + 1e-6)
            elif calc_method == "+10":
                df[new_col_name] = df[calc_col] + 10
            elif calc_method == "*3":
                df[new_col_name] = df[calc_col] * 3
            st.success("新列已添加！")

with st.expander("8. PCA 降维"):
    st.subheader("主成分分析降维")
    st.markdown("""
    **功能说明**：将高维数值列压缩为低维主成分，每次执行会自动清理旧的PCA列，避免重复报错
    """)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    pca_candidate_cols = [col for col in numeric_cols if not str(col).startswith("PCA_")]
    
    if len(pca_candidate_cols) < 2:
        st.warning("需要至少 2 个非 PCA 数值列才能降维")
    else:
        pca_cols = st.multiselect(
            "用于降维的数值列", 
            pca_candidate_cols, 
            default=pca_candidate_cols, 
            key="pca_cols_1"
        )
        max_dim = len(pca_cols) if pca_cols else 1
        n_components = st.slider(
            "降维维度", 
            1, 
            max_dim, 
            min(2, max_dim), 
            key="pca_dim_1"
        )
        
        if st.button("执行PCA", key="btn_pca") and len(pca_cols) >= n_components and len(pca_cols)>=2:
            # 删掉旧 PCA 列
            pca_old_cols = [col for col in df.columns if str(col).startswith("PCA_")]
            if pca_old_cols:
                df = df.drop(columns=pca_old_cols)
                st.info("已清理旧 PCA 列")
            
            # 降维
            pca = PCA(n_components=n_components)
            pca_result = pca.fit_transform(df[pca_cols])
            pca_df = pd.DataFrame(pca_result, columns=[f"PCA_{i+1}" for i in range(n_components)], index=df.index)
            
            # 合并
            df = pd.concat([df.drop(columns=pca_cols), pca_df], axis=1)
            st.success(f"降维至 {n_components} 维完成！")

# 实时数据预览
st.divider()
st.subheader("📊 处理后数据预览")
st.dataframe(df, use_container_width=True)

# 保存/导出/重置
st.divider()
st.subheader("🔘 操作控制")
ctrl1, ctrl2, ctrl3, ctrl4 = st.columns(4)

with ctrl1:
    # 保存当前处理结果到全局变量
    if st.button("保存修改", use_container_width=True, key="btn_save"):
        st.session_state.main_df = df
        st.success("已保存！")

with ctrl2:
    # 重置为原始数据（加空值判断，彻底避免None报错）
    if st.button("重置为原始数据", use_container_width=True, key="btn_reset"):
        if st.session_state.original_df is not None:
            st.session_state.main_df = st.session_state.original_df.copy()
            st.rerun()
        else:
            st.warning("⚠️ 暂无原始数据！请先上传数据后再重置。")

with ctrl3:
    # 撤销（简易版：回到上一次保存）
    if st.button("撤销操作", use_container_width=True, key="btn_undo"):
        df = st.session_state.main_df.copy()
        st.rerun()

with ctrl4:
    # 导出CSV
    st.download_button(
        label="导出CSV",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name="processed_data.csv",
        mime="text/csv",
        use_container_width=True,
        key="btn_export"
    )

# 同步到全局数据
st.session_state.main_df = df