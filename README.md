# Indeterminate — Data Analysis Platform
# 现在已停止更新！作为老版本记录
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
| 🌳 Decision Tree | 决策树分类（Pipeline 预处理 + 规则可视化 + 模型持久化 + 交互预测） |
| 🧪 Clustering | K-means + DBSCAN 聚类分析 |
| 🤖 LLM Analysis | 大模型对话分析（智能摘要/原始数据双模式、聊天界面、多轮对话） |

## 环境要求

- Python 3.10+
- CUDA 可选（仅深度学习模块需要 GPU 加速）

## 安装与运行

### 一键配置（推荐）

项目提供了自动检测硬件并安装所有依赖的配置脚本：

```bash
cd MyWeb

# 方式一：双击 setup.bat（推荐，无需任何配置）

# 方式二：在 PowerShell 中手动运行
powershell -ExecutionPolicy Bypass -File setup.ps1
```

脚本会自动完成：
1. 检查 Python 是否安装
2. 安装基础依赖包（streamlit, pandas, numpy, scikit-learn, plotly 等）
3. 检测 NVIDIA GPU → 自动选择对应 CUDA 版本的 PyTorch，无 GPU 则安装 CPU 版
4. 逐一验证所有包安装成功，并显示 CUDA 是否可用

### 使用特定 Conda 环境

如果你使用 Conda 管理 Python 环境，先激活目标环境再运行脚本即可：

```bash
# 1. 激活 conda 环境
conda activate myenv

# 2. 进入项目目录
cd MyWeb

# 3. 在 PowerShell 中运行配置脚本
powershell -ExecutionPolicy Bypass -File setup.ps1
```

> **注意**：不要双击 `setup.bat`（它不会继承 conda 环境）。在已激活 conda 的终端中手动执行上述命令即可，脚本会自动使用当前环境中的 python 和 pip。

---

### 手动安装（备选）

如果你更希望手动控制每个安装步骤：

```bash
cd MyWeb

# 1. 基础依赖
pip install streamlit pandas numpy scikit-learn plotly requests openpyxl

# 2. PyTorch（根据显卡选择，见下表）
```

#### NVIDIA 显卡对照

| 显卡系列 | 架构 | 推荐 CUDA | PyTorch 安装命令 |
|----------|------|-----------|-----------------|
| RTX 5070 / 5080 / 5090 / 5060 | Blackwell (sm_120) | CUDA 12.8 | `pip install torch --index-url https://download.pytorch.org/whl/cu128` |
| RTX 4060 / 4070 / 4080 / 4090 | Ada Lovelace (sm_89) | CUDA 12.1 | `pip install torch --index-url https://download.pytorch.org/whl/cu121` |
| RTX 3060 / 3070 / 3080 / 3090 | Ampere (sm_80/86) | CUDA 12.1 | 同上 |
| RTX 2060 / 2070 / 2080 / GTX 16xx | Turing (sm_75) | CUDA 11.8 | `pip install torch --index-url https://download.pytorch.org/whl/cu118` |
| GTX 10xx | Pascal (sm_61) | CUDA 11.8 | 同上 |
| AMD / Intel 显卡 / 无独显 | — | — | `pip install torch`（CPU 版） |

> **验证 GPU 是否可用**：`python -c "import torch; print(torch.cuda.is_available())"` → `True` 即表示 GPU 可用。

### 启动应用

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

### 决策树
- **Pipeline 预处理流水线**：自动识别数值/分类特征，数值列标准化 + 分类列独热编码
- **可解释性**：文本规则导出 + 节点详细信息表（划分特征、阈值、熵/基尼值、样本数）
- **模型持久化**：训练后自动保存 `.pkl` + `.json` 配置，下次进入页面自动加载
- **交互式预测**：分类特征自动显示为下拉框，数值特征为输入框
- **可调参数**：最大深度 2-10（默认 3）、划分标准 entropy/gini（默认 entropy）

### LLM 大模型分析
- **双模式**：Smart（发送数据摘要——统计量、相关性、缺失值等）节省 Token；Direct（发送原始 CSV）完整分析
- **聊天界面**：暗色主题可滑动对话框，用户/AI 消息视觉区分
- **Skill 注入**：Smart 模式下向 LLM 注入平台全部 9 个模块的能力表，让 AI 精准推荐分析工具
- **可编辑系统提示词**：折叠面板中自由编辑，Smart/Direct 各有默认提示词
- **多轮对话**：对话历史自动保存，每次发送完整上下文
- **API 兼容**：支持所有 OpenAI 兼容接口（DeepSeek、通义千问、智谱等）
- **错误处理**：超时、连接失败、API 错误分别给出中文提示

---

Developed by Jiayang Li
