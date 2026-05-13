# Indeterminate — Data Analysis Platform

基于 Streamlit 的数据分析与机器学习 Web 平台，集成回归、分类、聚类、决策树、自定义 MLP 以及 LLM 大模型分析功能。

## 功能概览

| 模块 | 说明 |
|------|------|
| 📊 Data Load | 上传 CSV/Excel 数据，预览、IQR 异常值检测与修复 |
| 📈 Data Visualization | 12 种 Plotly 交互式图表（散点图、折线图、小提琴图、3D 等） |
| 🧹 Data Processing | 数据清洗、特征缩放、PCA 降维等 |
| 🧠 Regression | PyTorch MLP 回归，三层拆分、损失曲线、批量预测 |
| 🔮 Classification | PyTorch MLP 分类，混淆矩阵、P/R/F1 报告、批量预测 |
| 🛠️ DIY MLP | 自定义 MLP 网络（层构建器、双任务、自适应参数警告） |
| 🌳 Decision Tree | 决策树分类器训练与规则导出 |
| 🧪 K-means | K-means 聚类分析 |
| 🤖 LLM Analysis | 调用大模型 API，支持 Smart（摘要分析）/ Direct（原始数据）双模式 |

## 环境要求

- Python 3.10+
- CUDA 可选（仅深度学习模块需要 GPU 加速）

## 安装与运行

### 1. 进入项目目录

```bash
cd MyWeb
```

### 2. 安装 Python 依赖

```bash
pip install streamlit pandas numpy scikit-learn matplotlib seaborn plotly requests
```

### 3. 安装 PyTorch（按显卡选择）

PyTorch 版本取决于你的显卡架构。请根据下表选择对应命令：

#### NVIDIA 显卡对照

| 显卡系列 | 架构 | 推荐 CUDA | PyTorch 最低版本 |
|----------|------|-----------|-----------------|
| RTX 5070 / 5080 / 5090 | Blackwell (sm_120) | CUDA 12.8 | torch ≥ 2.7 |
| RTX 5060 系列 | Blackwell (sm_120) | CUDA 12.8 | torch ≥ 2.7 |
| RTX 4060 / 4070 / 4080 / 4090 | Ada Lovelace (sm_89) | CUDA 12.1 | torch ≥ 2.0 |
| RTX 3060 / 3070 / 3080 / 3090 | Ampere (sm_80/86) | CUDA 11.8 | torch ≥ 1.10 |
| RTX 2060 / 2070 / 2080 | Turing (sm_75) | CUDA 11.8 | torch ≥ 1.10 |
| GTX 1650 / 1660 | Turing (sm_75) | CUDA 11.8 | torch ≥ 1.10 |
| GTX 1050 / 1060 / 1070 / 1080 | Pascal (sm_61) | CUDA 11.8 | torch ≥ 1.10 |
| AMD / Intel 显卡 / 无独显 | — | — | CPU 版本 |

#### RTX 5070 / 5080 / 5090 / 5060（Blackwell 架构）

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

#### RTX 4060 / 4070 / 4080 / 4090（Ada Lovelace）及其他 RTX 30/20 系列

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

#### GTX 16 / 10 系列及较旧 NVIDIA 显卡

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

#### CPU 运行（无 NVIDIA 显卡或 AMD/Intel 显卡）

```bash
pip install torch torchvision torchaudio
```

> **验证 GPU 是否可用**：安装完成后运行 `python -c "import torch; print(torch.cuda.is_available())"`，输出 `True` 即表示 GPU 可用。

### 4. 启动应用

```bash
streamlit run main.py
```

浏览器自动打开 `http://localhost:8501`。

## 使用提示

### 数据流
- 各页面通过 `📂 Data Input` 组件共享数据：在上一个页面上传后，切换到其他页面无需重复上传
- Data Load 页面内置 **IQR 异常值检测**，支持 Winsorize / 均值替换 / 中位数替换 / 删除行等修复方式
- 其他页面会自动提示异常值数量，引导用户前往 Data Load 页面处理

### 数据可视化
- **全部使用 Plotly**：图表支持缩放、平移、悬停查看数据点详情、一键导出 PNG
- **12 种图表类型**：散点图、折线图、柱状图、面积图、直方图、箱线图、小提琴图、二维密度热力图、饼图/环形图、成对关系图、相关性热力图、3D 散点图
- **散点图增强**：支持气泡大小映射、OLS/LOWESS 趋势线、边缘分布图
- **聚合功能**：柱状图/折线图支持按 X 轴 groupby 聚合（均值/求和/计数/中位数等）
- **自定义**：可调标题、7 种配色模板、图表宽高

### 机器学习模块
- **超参数可调**：学习率、训练轮数、Batch Size 均可在训练前调整
- **三层数据拆分**：训练集/验证集/测试集独立，防止数据泄漏
- **训练可视化**：实时进度条 + 训练/验证损失曲线
- **批量预测**：上传 CSV 文件批量预测并下载结果（Excel 可直接打开）
- **设备安全**：GPU/CPU 切换后自动迁移模型，不会因设备不匹配崩溃
- 分类任务显示混淆矩阵 + Precision/Recall/F1 报告
- DIY MLP 支持自由设计网络结构，实时显示参数量与过拟合/欠拟合警告
- 深度学习模块建议使用 GPU 加速，CPU 训练速度较慢
- LLM Analysis 的 Smart 模式只发送数据摘要（统计量+相关性），大幅节省 token；Direct 模式保留原始行为

---

Developed by Jiayang Li
