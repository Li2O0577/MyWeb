# Indeterminate — 数据分析平台 (Flask + Streamlit 前后端分离版)

基于 **Flask + Streamlit** 的多页数据科学 Web 应用，集成机器学习与深度学习模块。

## 技术栈

| 层 | 技术 |
|---|------|
| **前端** | Streamlit |
| **后端** | Flask (REST API) |
| **ML/DL** | PyTorch, scikit-learn |
| **数据处理** | pandas, numpy |
| **可视化** | Plotly (交互式) |
| **LLM** | OpenAI 兼容 API (DeepSeek, Qwen, 智谱等) |

## 架构

```
用户浏览器
    │
    ├── Streamlit (:8501) ── UI 渲染、Plotly 图表、用户交互
    │         │
    │         └── HTTP/JSON ── Flask (:5000) ── ML 训练、模型推理、数据解析
    │
    └── Flask SSE ── LLM 流式聊天
```

- **Streamlit** 负责全部 UI：页面布局、Plotly 图表、数据预览、参数控件
- **Flask** 负责全部计算：数据解析、模型训练、推理预测、LLM 代理
- 前后端通过 HTTP JSON 通信，Flask 无状态（session 数据 TTL 1 小时）

## 快速开始

### 1. 一键安装

```bash
# Windows: 双击 setup.bat
# 或手动:
powershell -ExecutionPolicy Bypass -File setup.ps1
```

脚本自动检测 GPU 并安装对应 PyTorch 版本（CUDA 11.8 / 12.1 / 12.8 / CPU）。

### 2. 启动

```bash
# 终端 1 — 启动 Flask 后端
cd backend
python app.py
# → http://localhost:5000

# 终端 2 — 启动 Streamlit 前端
cd ..
streamlit run main.py
# → http://localhost:8501
```

### 3. 使用

1. 打开 `http://localhost:8501`
2. 在「数据加载」页面上传 CSV/Excel
3. 在「数据可视化」页面探索数据
4. 在「数据处理」页面清洗和转换
5. 在 ML 页面（回归/分类/DIY MLP/决策树/聚类）训练模型并预测
6. 在「大模型分析」页面让 AI 推荐分析方向

## 项目结构

```
MyWeb1/
├── main.py                      # Streamlit 首页
├── requirements.txt             # 前端依赖
├── setup.bat / setup.ps1        # 一键环境配置
├── pages/                       # Streamlit 页面
│   ├── _api.py                  # Flask API 客户端（所有后端调用集中管理）
│   ├── _prepare.py              # 公共组件（数据上传、侧边栏导航、异常值检测）
│   ├── _mlp_common.py           # MLP UI 工具（设备选择器、损失曲线、输入校验）
│   ├── 1_data_load.py           # 数据加载 + IQR 异常值修复
│   ├── 2_data_visualization.py  # 12 种交互式图表（Plotly）
│   ├── 3_data_processing.py     # 数据处理：标准化/归一化/PCA/编码
│   ├── 4_regression.py          # 回归预测（PyTorch MLP）
│   ├── 5_classification.py      # 分类决策（PyTorch MLP）
│   ├── 6_diy_mlp.py             # 自定义 MLP（层构建器 + 双任务）
│   ├── 7_decision_tree.py       # 决策树（Pipeline + 规则可视化）
│   ├── 8_clustering.py          # 聚类分析（K-means + DBSCAN）
│   └── 9_llm_analysis.py        # 大模型分析（SSE 流式聊天）
├── backend/                     # Flask 后端
│   ├── app.py                   # Flask 入口（session 管理、blueprint 注册）
│   ├── requirements.txt         # 后端依赖
│   ├── routes/                  # API 路由层（参数校验 + 调用 service）
│   │   ├── data_routes.py       # /api/data/*
│   │   ├── regression_routes.py # /api/regression/*
│   │   ├── classification_routes.py
│   │   ├── diy_mlp_routes.py
│   │   ├── decision_tree_routes.py
│   │   ├── clustering_routes.py
│   │   └── llm_routes.py
│   ├── services/                # 业务逻辑层（从 Streamlit 抽出的核心计算）
│   │   ├── data_service.py
│   │   ├── regression_service.py
│   │   ├── classification_service.py
│   │   ├── diy_mlp_service.py
│   │   ├── decision_tree_service.py
│   │   ├── clustering_service.py
│   │   └── llm_service.py
│   └── models/                  # 训练好的模型文件（.gitignore 排除）
└── .gitignore
```

## API 概览

| 端点 | 方法 | 说明 |
|------|------|------|
| `/api/health` | GET | 健康检查 |
| `/api/data/upload` | POST | 上传 CSV/Excel，返回 session_id |
| `/api/data/<sid>/process` | POST | 数据处理操作 |
| `/api/data/<sid>/summary` | GET | 数据摘要（LLM 用） |
| `/api/regression/train` | POST | 训练回归模型 |
| `/api/regression/predict` | POST | 单条预测 |
| `/api/regression/batch_predict` | POST | 批量预测 |
| `/api/regression/clear` | POST | 清除模型 |
| `/api/classification/*` | POST | 同上模式 |
| `/api/diy_mlp/*` | POST | 同上模式 |
| `/api/decision_tree/*` | POST | 同上模式 |
| `/api/clustering/*` | POST | 同上模式 + `/elbow` |
| `/api/llm/chat` | POST | SSE 流式聊天 |

## 功能模块

### 数据加载 (Page 1)
- 上传 CSV/Excel，自动预览
- IQR 异常值检测（Q1−1.5×IQR, Q3+1.5×IQR）
- 异常值修复：缩尾处理 / 均值替换 / 中位数替换 / 删除行

### 数据可视化 (Page 2)
- 12 种交互式 Plotly 图表：散点图、折线图、柱状图、面积图、直方图、箱线图、小提琴图、二维密度热力图、饼图/环形图、成对关系图、相关性热力图、3D 散点图
- 支持聚合、趋势线、边缘分布、气泡大小、色彩模板

### 数据处理 (Page 3)
- 行列筛选/增减/重命名、类型转换
- StandardScaler / MinMaxScaler、PCA 降维
- 标签编码 / 独热编码、自定义计算列、高斯噪声

### ML 模块 (Page 4–8)
- **回归**：PyTorch MLP，R²/MAE/RMSE，损失曲线
- **分类**：PyTorch MLP，二分类/多分类自适应，混淆矩阵 + P/R/F1
- **DIY MLP**：自由设计网络结构，双任务支持，参数量/过拟合警告
- **决策树**：Pipeline 预处理，规则可视化，节点详情
- **聚类**：K-means + DBSCAN，肘部法则，轮廓系数，PCA 可视化

### 大模型分析 (Page 9)
- Smart 模式：发送数据摘要，AI 推荐分析方向
- Direct 模式：发送原始数据，AI 自定义分析
- SSE 流式响应，聊天界面

## 开发者

Jiayang Li
