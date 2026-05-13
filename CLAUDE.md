# CLAUDE.md — Indeterminate Data Analysis Platform

基于 **Streamlit** 的多页数据科学 Web 应用，集成机器学习与深度学习模块。

## 技术栈

- **Web**: Streamlit
- **ML/DL**: PyTorch, scikit-learn
- **数据处理**: pandas, numpy
- **可视化**: matplotlib, seaborn, plotly
- **API**: requests (OpenAI 兼容)

## 项目结构

```
MyWeb/
├── main.py                    # 首页，9个功能按钮 + 侧边栏导航
├── pages/
│   ├── 1_data_load.py         # 上传 CSV/Excel，预览、IQR 异常值检测 + 修复
│   ├── 2_data_visualization.py # 散点图、折线图、直方图、箱线图、热力图
│   ├── 3_data_processing.py   # 空值处理、标准化/归一化、PCA 降维
│   ├── 4_regression.py        # PyTorch MLP 回归
│   ├── 5_classification.py    # PyTorch MLP 分类 (支持二分类/多分类)
│   ├── 6_diy_mlp.py           # 自定义 MLP 层数、神经元数、激活函数、学习率
│   ├── 7_decision_tree.py     # 决策树分类器 + 可视化 + 规则导出
│   ├── 8_k_means.py           # K-means 聚类 + 肘部法则
│   └── 9_llm_analysis.py      # LLM 分析，Smart（摘要）/ Direct（原始数据）双模式
└── .gitignore
```

## 数据流

- 各页面通过 `st.session_state` 共享数据（以 `_prepare.py` 为公共工具模块）
- `st.session_state.main_df` 存储当前数据，`st.session_state.outliers` 存储异常值检测结果
- 用户在 Data Load 页面上传数据后，其他页面可直接读取；异常值检测自动执行
- `_data_cleaned` 持久标记：Page 1 修复异常值后，切到其他页面再回来不会丢失修复

## 运行方式

```bash
streamlit run main.py
```

端口: `http://localhost:8501`

## 关键模块说明

- `_prepare.py` 是各页面的公共依赖，提供 `data_uploader()`, `render_sidebar()`, `detect_outliers()`
- `detect_outliers()` 使用 IQR 方法（Q1−1.5×IQR, Q3+1.5×IQR），仅对数值列检测
- `data_uploader()` 默认 `warn_outliers=True`（除 Page 1 外自动警告），支持 `force_cached` 参数

## 注意事项

- PyTorch 模块 (regression/classification/diy_mlp) 建议 GPU 加速，CPU 训练较慢
- LLM Analysis 支持所有兼容 OpenAI API 的服务（OpenAI、DeepSeek、通义千问、智谱等）
- LLM Analysis Smart 模式默认推荐，发送数据摘要而非原始数据，节省 token
