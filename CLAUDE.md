# CLAUDE.md — Indeterminate Data Analysis Platform

基于 **Streamlit** 的多页数据科学 Web 应用，集成机器学习与深度学习模块。

## 技术栈

- **Web**: Streamlit
- **ML/DL**: PyTorch, scikit-learn
- **数据处理**: pandas, numpy
- **可视化**: Plotly (交互式), matplotlib, seaborn
- **API**: requests (OpenAI 兼容)

## 项目结构

```
MyWeb/
├── main.py                    # 首页，9个功能按钮 + 侧边栏导航
├── pages/
│   ├── 1_data_load.py         # 上传 CSV/Excel，预览、IQR 异常值检测 + 修复
│   ├── 2_data_visualization.py # 12种交互式图表（Plotly），中文界面
│   ├── 3_data_processing.py   # 空值处理、标准化/归一化、PCA 降维
│   ├── 4_regression.py        # PyTorch MLP 回归（三层拆分、损失曲线、批量预测）
│   ├── 5_classification.py    # PyTorch MLP 分类（混淆矩阵、P/R/F1、批量预测）
│   ├── 6_diy_mlp.py           # 自定义 MLP（层构建器、双任务、自适应警告）
│   ├── 7_decision_tree.py     # 决策树分类器 + 可视化 + 规则导出
│   ├── 8_k_means.py           # K-means 聚类 + 肘部法则
│   └── 9_llm_analysis.py      # LLM 分析，Smart（摘要）/ Direct（原始数据）双模式
└── .gitignore
```

## 数据流

- 各页面通过 `st.session_state` 共享数据（以 `_prepare.py` 为公共工具模块）
- `st.session_state.main_df` 存储当前数据，`st.session_state.outliers` 存储异常值检测结果
- 用户在 Data Load 页面上传数据后，其他页面可直接读取
- `_data_cleaned` 持久标记：Page 1 修复异常值后，切到其他页面再回来不会丢失修复

## 运行方式

```bash
streamlit run main.py
```

端口: `http://localhost:8501`

## 关键模块说明

### 公共模块
- `_prepare.py` 是各页面的公共依赖，提供 `data_uploader()`, `render_sidebar()`, `detect_outliers()`
- `detect_outliers()` 使用 IQR 方法（Q1−1.5×IQR, Q3+1.5×IQR），仅对数值列检测
- `data_uploader()` 默认 `warn_outliers=True`（除 Page 1 外自动警告），支持 `force_cached` 参数

### 数据可视化 (Page 2)
- **全部使用 Plotly**：交互式缩放、悬停查看数值、一键导出 PNG
- **12 种图表**：散点图、折线图、柱状图、面积图、直方图、箱线图、小提琴图、二维密度热力图、饼图/环形图、成对关系图、相关性热力图、3D 散点图
- **聚合功能**：折线图/柱状图支持 Mean/Sum/Count/Median 等 groupby 聚合
- **散点图增强**：气泡大小、悬停信息、OLS/LOWESS 趋势线、边缘分布
- **布局**：左栏配置 + 右栏图表，全局色彩模板 + 尺寸设置
- **界面语言**：全中文

### 机器学习模块 (Page 4/5/6)
- **三层数据拆分**：训练集 64% / 验证集 16% / 测试集 20%，验证集仅用于早停，测试集仅用于最终评估
- **自适应 batch_size**：训练集小于所选 batch 时自动缩小
- **可调超参数**：学习率（0.01~0.0001）、最大训练轮数（20~500）、Batch Size（4~128）
- **训练可视化**：实时进度条 + 每轮 loss 显示 + 训练/验证损失曲线（Plotly）
- **设备安全**：切换 GPU/CPU 时自动迁移模型，避免张量设备不匹配
- **批量预测**：上传 CSV 批量预测并下载结果
- **模型路径**：reg_* / cls_* / diy_* 各自独立，不互相覆盖

#### 回归 (Page 4)
- PyTorch MLP 回归，输出连续值
- 指标：R² + MAE + RMSE
- `load_saved_model` 包含列名校验

#### 分类 (Page 5)
- 输出 raw logits，损失函数自带 Sigmoid/Softmax（修复了双重激活 bug）
- 损失函数：二分类 `BCEWithLogitsLoss` / 多分类 `CrossEntropyLoss`
- 指标：准确率 + 混淆矩阵（Plotly 热力图）+ Precision/Recall/F1 分类报告
- `reverse_label_map` 键统一为字符串，JSON 序列化安全

#### DIY MLP (Page 6)
- **层构建器**：自由添加/删除隐藏层，每层可配置神经元数、激活函数、BatchNorm、Dropout
- **激活函数**：ReLU / LeakyReLU / GELU / Tanh / Sigmoid / ELU / SELU / 无激活
- **双任务**：回归 + 分类，自动适配输出层与损失函数
- **自适应警告**：参数量 vs 样本数匹配度、梯度消失风险、纯线性退化警告
- **可选优化器**：Adam / AdamW / SGD / RMSprop

## 注意事项

- PyTorch 模块建议 GPU 加速，CPU 训练较慢
- LLM Analysis 支持所有兼容 OpenAI API 的服务（OpenAI、DeepSeek、通义千问、智谱等）
- LLM Analysis Smart 模式默认推荐，发送数据摘要而非原始数据，节省 token
- 分类/DIY 页面的 label_map 键统一转为字符串再 JSON 序列化，避免 numpy 类型问题
- 模型文件保存在 `models/` 目录，各页面独立命名不会互相覆盖
