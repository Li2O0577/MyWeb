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
    │         └── HTTP/JSON ── Flask (:5001) ── ML 训练、模型推理、数据解析
    │
    └── Flask SSE ── LLM 流式聊天
```

- **Streamlit** 负责 UI 和数据预处理：页面布局、Plotly 图表、数据加载、特征工程（缩放/编码/PCA）
- **Flask** 负责 ML 计算和会话管理：数据解析、模型训练、推理预测、LLM 代理
- 前后端通过 HTTP JSON 通信，Flask 无状态（session 数据 TTL 1 小时）

## 快速开始

### 1. 一键安装

```bash
# Windows: 双击 setup.bat
# 或手动:
powershell -ExecutionPolicy Bypass -File setup.ps1
```

脚本自动检测 GPU 并安装对应 PyTorch 版本（CUDA 11.8 / 12.1 / 12.8 / CPU）。

### 环境要求

| 依赖 | 版本 | 说明 |
|------|------|------|
| Python | 3.10+ | |
| Flask | **3.1.3** | 必须与此版本一致！高了低了都可能和 Werkzeug 不兼容 |
| Werkzeug | 3.1.x | Flask 3.1.3 的配套版本，自动安装 |
| flask-cors | 4.0+ | 跨域支持 |
| PyTorch | 2.0+ | 建议 GPU 版，CPU 版训练较慢 |
| pandas | 2.0+ | |
| scikit-learn | 1.3+ | |
| pyarrow | 14.0+ | Session 安全持久化 |
| Streamlit | 1.35+ | 前端（需 st.toast 支持） |

> **⚠️ 如果遇到 `ImportError: cannot import name 'url_quote' from 'werkzeug.urls'`**
> 
> 这是 Flask 版本太旧而 Werkzeug 太新导致的。执行以下修复：
> ```bash
> pip install --upgrade flask
> ```

### 依赖文件

- `requirements.txt` — 前端依赖（Streamlit + 可视化）
- `backend/requirements.txt` — 后端依赖（Flask + PyTorch + sklearn）
- `setup.ps1` — 一键安装脚本，自动检测 GPU

### 2. 一键启动

```bash
# Windows: 双击 start.bat
# 或手动:
powershell -ExecutionPolicy Bypass -File start.ps1
```

脚本自动检测 Python 解释器（标准 Python / Conda / uv），用户选择后自动在两个终端窗口启动 Flask 后端（:5001）和 Streamlit 前端（:8501）。

### 3. 手动启动

打开两个终端，均需先激活 Python 环境：

```bash
# 终端 1 — 启动 Flask 后端
conda activate 你的环境          # 或用你自己的环境名
cd backend
python app.py
# → http://localhost:5001

# 终端 2 — 启动 Streamlit 前端
conda activate 你的环境
cd ..
set INDETERMINATE_API_BASE=http://127.0.0.1:5001/api
streamlit run main.py
# → http://localhost:8501
```

可选配置：

| 环境变量 | 默认值 | 说明 |
|----------|--------|------|
| `FLASK_PORT` | `5001` | Flask 后端端口 |
| `FLASK_DEBUG` | `0` | 设为 `1` 开启 Flask debug 模式（有安全风险，仅开发用） |
| `INDETERMINATE_API_BASE` | `http://127.0.0.1:5001/api` | 前端 API 地址 |
| `LLM_API_KEY` | 无 | LLM 默认 API 密钥，设置后前端无需手动填写 |

### 4. 清理缓存

```bash
# Windows: 双击 clean.bat
# 或手动:
powershell -ExecutionPolicy Bypass -File clean.ps1          # 交互模式
powershell -ExecutionPolicy Bypass -File clean.ps1 -WhatIf  # 预览模式（不实际删除）
powershell -ExecutionPolicy Bypass -File clean.ps1 -Force   # 跳过确认直接删除
```

清理范围：`__pycache__` 目录、模型文件（.pth/.pkl 等）、模型版本目录、session 数据、日志、临时文件。**保留** `registry.json` 和代码文件。

### 5. 使用

1. 打开 `http://localhost:8501`
2. 在「数据加载」页面上传 CSV/Excel
3. 在「数据可视化」页面探索数据
4. 在「数据处理」页面清洗和转换
5. 在 ML 页面（回归/分类/DIY MLP/决策树/聚类）训练模型并预测
6. 在「大模型分析」页面与 AI 对话：Smart 模式推荐方向、Direct 模式深入分析、Agent 模式让 AI 自主执行训练和推理

## 项目结构

```
MyWeb1/
├── main.py                      # Streamlit 首页
├── requirements.txt             # 前端依赖
├── setup.bat / setup.ps1        # 一键环境配置（自动检测 GPU 安装 PyTorch）
├── start.bat / start.ps1        # 一键启动（自动检测 Python/Conda/uv 并启动前后端）
├── clean.bat / clean.ps1        # 清理缓存（__pycache__、模型文件、session 数据、日志等）
├── pages/                       # Streamlit 页面
│   ├── _api.py                  # Flask API 客户端（所有后端调用集中管理）
│   ├── _prepare.py              # 公共组件（数据上传、侧边栏导航，detect_outliers 从 backend 导入）
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
│   ├── session_store.py         # Session 持久化（内存 + Parquet/JSON 磁盘），支持重启恢复
│   ├── routes/                  # API 路由层（参数校验 + 调用 service）
│   │   ├── _versioning.py       # 共享版本管理路由工厂（5 模块共用）
│   │   ├── _helpers.py          # 列名适配工具
│   │   ├── _responses.py        # 统一错误响应格式
│   │   ├── data_routes.py       # /api/data/*
│   │   ├── regression_routes.py # /api/regression/*
│   │   ├── classification_routes.py
│   │   ├── diy_mlp_routes.py
│   │   ├── decision_tree_routes.py
│   │   ├── clustering_routes.py
│   │   └── llm_routes.py        # SSE 流式聊天
│   ├── services/                # 业务逻辑层（从 Streamlit 抽出的核心计算）
│   │   ├── data_service.py
│   │   ├── regression_service.py
│   │   ├── classification_service.py
│   │   ├── diy_mlp_service.py
│   │   ├── decision_tree_service.py
│   │   ├── clustering_service.py
│   │   ├── llm_service.py
│   │   └── training_validation.py  # 训练前友好校验
│   ├── models/                    # 模型版本存储 + 注册表
│   │   ├── registry.py            # 版本注册表管理器
│   │   └── {type}/{version_id}/   # 各版本目录（.gitignore 排除）
│   ├── sessions/                  # Session 持久化 (Parquet + JSON, .gitignore 排除)
└── .gitignore
```

## API 概览

错误响应统一为：

```json
{
  "ok": false,
  "error": {
    "code": "SESSION_EXPIRED",
    "message": "Session not found or expired",
    "detail": "Upload or sync the current dataset again before running this operation."
  }
}
```

| 端点 | 方法 | 说明 |
|------|------|------|
| `/api/health` | GET | 健康检查，返回活跃 session 数、最近 session 元信息、已保存模型 |
| `/api/data/upload` | POST | 上传 CSV/Excel，返回 session_id 和 session_meta |
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
| `/api/llm/agent` | POST | SSE Agent 模式（function calling 自主执行分析） |
| `/api/{type}/versions` | GET | 列出该类型所有模型版本 |
| `/api/{type}/version/<vid>` | GET | 获取版本详情 |
| `/api/{type}/activate` | POST | 切换激活版本 |
| `/api/{type}/version/<vid>` | DELETE | 删除版本及其文件 |

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
- 标签编码 / 独热编码、高斯噪声
- 自定义计算列：简单计算（一元/二元运算）+ 自定义表达式（`df.eval`）

### ML 模块 (Page 4–8)
- **回归**：PyTorch MLP，R²/MAE/RMSE，损失曲线
- **分类**：PyTorch MLP，二分类/多分类自适应，混淆矩阵 + P/R/F1
- **DIY MLP**：自由设计网络结构，双任务支持，参数量/过拟合警告
- **决策树**：Pipeline 预处理，规则可视化，节点详情
- **聚类**：K-means + DBSCAN，肘部法则，轮廓系数，PCA 可视化

### 大模型分析 (Page 9)
- **Smart 模式**：发送数据摘要（自动截断至 8000 字符），AI 推荐分析方向
- **Direct 模式**：发送原始数据，AI 自定义分析
- **Agent 模式**：AI 通过 function calling 自主调用平台工具
  - 支持 6 种工具：数据概览、列详情、回归训练、分类训练、聚类分析、相关性分析
  - AI 自动选择工具、执行分析、解读指标，实时展示调用进度
  - 最多 10 轮迭代，需要支持 function calling 的模型
- SSE 流式响应，聊天界面
- API Base 白名单（OpenAI / DeepSeek / 通义千问 / 智谱 / Kimi 等）+ `LLM_API_KEY` 环境密钥支持

## 安全设计

- **Debug 模式**：默认关闭，通过 `FLASK_DEBUG=1` 手动开启
- **LLM 代理**：API Base 域名白名单，防止 SSRF 攻击；Agent 模式下工具执行均在服务端完成，LLM 不直接访问数据文件
- **文件上传**：256MB 硬限制，防止内存耗尽
- **Session 存储**：Parquet + JSON 替代 pickle，消除反序列化代码执行风险
- **模型加载**：sklearn Pipeline / KMeans 加载时进行类型验证

## 开发者

Jiayang Li
