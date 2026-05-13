# Indeterminate — Data Analysis Platform

基于 Streamlit 的数据分析与机器学习 Web 平台，集成回归、分类、聚类、决策树、自定义 MLP 以及 LLM 大模型分析功能。

## 功能概览

| 模块 | 说明 |
|------|------|
| 📊 Data Load | 上传 CSV/Excel 数据，预览、IQR 异常值检测与修复 |
| 📈 Data Visualization | 使用 Matplotlib / Seaborn / Plotly 绘制图表 |
| 🧹 Data Processing | 数据清洗、特征缩放、PCA 降维等 |
| 🧠 Regression | 基于 PyTorch MLP 的回归分析 |
| 🔮 Classification | 基于 PyTorch MLP 的分类任务 |
| 🛠️ DIY MLP | 自定义 MLP 网络结构（层数、神经元、激活函数） |
| 🌳 Decision Tree | 决策树分类器训练与规则导出 |
| 🧪 K-means | K-means 聚类分析 |
| 🤖 LLM Analysis | 调用大模型 API，支持 Smart（摘要分析）/ Direct（原始数据）双模式 |

## 环境要求

- Python 3.10+
- CUDA 可选（仅深度学习模块需要 GPU 加速）

## 安装与运行

### 1. 克隆 / 下载项目

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

- 各页面通过 `📂 Data Input` 组件共享数据：在上一个页面上传后，切换到其他页面无需重复上传
- Data Load 页面内置 **IQR 异常值检测**，支持 Winsorize / 均值替换 / 中位数替换 / 删除行等修复方式
- 其他页面会自动提示异常值数量，引导用户前往 Data Load 页面处理
- LLM Analysis 的 Smart 模式只发送数据摘要（统计量+相关性），大幅节省 token；Direct 模式保留原始行为
- 深度学习模块建议使用 GPU 加速，CPU 训练速度较慢

---

Developed by Jiayang Li
