# Indeterminate — 数据分析平台 (Flask + Streamlit 前后端分离版)

基于 **Flask + Streamlit** 的多页数据科学 Web 应用，集成机器学习与深度学习模块。

## 技术栈

| 层 | 技术 |
|---|------|
| **前端** | Streamlit |
| **后端** | Flask (REST API) |
| **ML/DL** | PyTorch, scikit-learn |
| **数据处理** | pandas, numpy |
| **可视化** | Plotly (交互式), Matplotlib (后端图表生成) |
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
- ML/LLM/数据处理页面共用统一页头、顶部状态条、区块标题、工作台概况、稳定预测结果面板和风险提示样式，减少页面间体验差异

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
| `FLASK_DEBUG` | `0` | 设为 `1` 开启 Flask debug 模式（有安全风险，仅开发用；默认不向前端暴露 500 内部异常细节） |
| `INDETERMINATE_API_BASE` | `http://127.0.0.1:5001/api` | 前端 API 地址 |
| `LLM_API_KEY` | 无 | LLM 默认 API 密钥，设置后前端无需手动填写 |
| `LLM_ALLOW_LOCAL_API_BASE` | `0` | 设为 `1` 才允许 LLM API Base 使用 localhost/私网地址，仅建议本地开发使用 |

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
2. 在首页查看运行环境状态：Python、PyTorch、CUDA、Flask 后端和关键库是否可用
3. 在「数据加载」页面上传 CSV/Excel
4. 在「数据可视化」页面探索数据
5. 在「数据处理」页面清洗和转换
6. 在 ML 页面（回归/分类/DIY MLP/决策树/聚类）训练模型并预测
7. 在「大模型分析」页面与 AI 对话：Smart 模式推荐方向、Direct 模式深入分析、Agent 模式让 AI 自主执行训练和推理

### 6. 运行测试

```bash
python -m unittest discover -s tests
python -m compileall .\backend .\pages .\tests
```

测试覆盖训练前校验、Flask 上传/训练/预测冒烟流程，适合在修改 ML 或 API 逻辑后快速确认主链路没有断。

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
│   ├── _mlp_common.py           # ML UI 工具（工作台概况、预测结果面板、设备选择、损失曲线、输入校验）
│   ├── _ui_common.py            # 统一页头、状态条、区块标题、键值面板、预测面板和提示样式
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
│   │   ├── _safe_serialize.py     # 安全序列化：np.savez + RestrictedUnpickler 替代 pickle
│   │   ├── _sandbox.py            # 代码解释器沙箱：AST 验证 + 受限 exec() 环境
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
- IQR 异常值检测（可调 1.5×/3.0×IQR，IQR=0 时自动回退 MAD→STD）
- NaN 缺失值检测与填充（均值/中位数）
- 异常值修复：缩尾处理 / 均值替换 / 中位数替换 / 删除行
- **批量操作**：全部列一键缩尾/替换/删除
- **严重程度标注**：每个异常值显示超出边界的倍数和方向

### 数据可视化 (Page 2)
- 12 种交互式 Plotly 图表：散点图、折线图、柱状图、面积图、直方图、箱线图、小提琴图、二维密度热力图、饼图/环形图、成对关系图、相关性热力图、3D 散点图
- 支持聚合、趋势线、边缘分布、气泡大小、色彩模板

### 数据处理 (Page 3)
- 行列筛选/增减/重命名、类型转换
- StandardScaler / MinMaxScaler、PCA 降维
- 标签编码 / 独热编码、高斯噪声
- 自定义计算列：简单计算（一元/二元运算）+ 自定义表达式；表达式只允许数值列、数字、括号和基础算术运算，禁止函数调用和对象属性访问

### ML 模块 (Page 4–8)
- **回归**：PyTorch MLP，R²/MAE/RMSE，损失曲线
- **分类**：PyTorch MLP，二分类/多分类自适应，混淆矩阵 + P/R/F1
- **DIY MLP**：自由设计网络结构，双任务支持，参数量/过拟合警告
- **决策树**：Pipeline 预处理，规则可视化，节点详情
- **聚类**：K-means + DBSCAN，肘部法则，轮廓系数，PCA 可视化
- **统一体验**：各 ML 页面顶部统一展示当前数据集、后端同步状态、当前模型版本
- **训练工作台**：Page 4-8 统一展示“训练数据 / 训练准备 / 当前模型版本”三栏概况，减少页面间认知差异
- **稳定预测面板**：单条预测结果固定保留在页面内；回归显示预测值和当前模型指标，分类/DIY/决策树显示预测类别、置信度和类别概率表，聚类显示簇编号和簇说明
- **版本管理**：模型版本选择器会展示选中版本的数据集、目标/任务、指标、创建时间和推荐状态；切换或删除版本会清空当前页面旧预测结果，删除前需要勾选确认
- **运行环境提示**：首页显示 Python、PyTorch、CUDA、Flask 后端和关键库状态；未安装 PyTorch 时训练页仍可打开，但会提示训练依赖缺失
- **后端同步恢复**：后端短暂断连时前端保留已有 session 标识；只有后端明确返回 session 过期才清理状态，减少临时断连导致的数据状态丢失
- **风险提示**：小数据集、类别不均衡、类别过多、常量特征等风险使用统一提示样式
- **预测校验**：单条/批量预测会提前检查特征数量、空值、无穷值和非数值输入；批量预测 CSV 会额外检查空文件、重复列名、缺失特征列和单次行数上限；后端预测 API 也会兜底校验数值 payload，避免直接调用接口时暴露 sklearn/torch 技术异常
- **预测返回语义**：分类和 DIY MLP 的二分类 `prob` 表示预测类别置信度，并额外返回 `all_probs`/`label_names` 供前端概率表展示；决策树分类预测返回 `predict_proba` 概率表

### 大模型分析 (Page 9)
- **Smart 模式**：发送数据摘要（自动截断至 8000 字符），AI 推荐分析方向
- **Direct 模式**：发送原始数据，AI 自定义分析；为避免请求过大，最多发送 1000 行且 CSV 约 512KB
- **Agent 模式**：AI 通过 function calling 自主调用平台工具
  - 支持 8 种工具：数据概览、列详情、回归训练、分类训练、聚类分析、相关性分析、**图表生成**（散点图/直方图/热力图等 8 种）、**代码解释器**（子进程沙箱执行 Python 代码）
  - AI 自动选择工具、执行分析、绘制图表、编写代码，图表和代码输出**内嵌显示在对话中**；Agent 图表使用中等尺寸渲染，标题、坐标轴、图例和分类刻度会使用英文/ASCII 兜底，避免 matplotlib 中文字体缺失
  - 最多 10 轮迭代、12 次工具调用；工具参数会自动清洗和限幅
  - 代码解释器使用子进程隔离，30 秒超时，限制 10000 字符代码长度、最多 5000 行/80 列输入、最多返回 5 张图片且单张图片约 8MB 上限
  - 需要支持 function calling 的模型
- 现代化聊天 UI：原生 Streamlit 组件，Markdown 渲染，图片内嵌显示
- SSE 流式响应，实时打字机效果；常见 LLM/Agent 错误码会在前端转换为中文用户提示
- Agent 模式会先确认当前数据已同步到后端；LLM/Agent 非 200 响应和连接异常会转换为中文提示，避免暴露原始 requests 异常
- API Base 白名单（OpenAI / DeepSeek / 通义千问 / 智谱 / Kimi 等）+ `LLM_API_KEY` 环境密钥支持；localhost/私网地址默认禁用，可用 `LLM_ALLOW_LOCAL_API_BASE=1` 在本地开发时开启

## 安全设计

- **Debug 模式**：默认关闭，通过 `FLASK_DEBUG=1` 手动开启；默认 500 错误不向前端返回内部异常 detail
- **LLM 代理**：API Base 域名白名单，防止 SSRF 攻击；Agent 模式下工具执行均在服务端完成，LLM 不直接访问数据文件
- **本地 API Base 开关**：localhost/私网 LLM API Base 默认不允许，避免部署环境 SSRF；仅本地开发时设置 `LLM_ALLOW_LOCAL_API_BASE=1`
- **文件上传**：256MB 硬限制，防止内存耗尽
- **Session 存储**：Parquet + JSON 替代 pickle，旧 `.pkl` session 通过 `RestrictedUnpickler`（模块白名单）安全迁移后删除
- **模型序列化**：StandardScaler / KMeans 参数用 `np.savez` 保存（`.npz` 格式，零代码执行）；DecisionTree Pipeline 等复杂对象经 `RestrictedUnpickler` 加载，仅允许 sklearn/numpy/pandas/pyarrow 模块
- **LLM Agent 护栏**：限制消息长度、对话条数、工具调用次数和工具参数范围；Direct 模式限制原始数据大小；代码解释器使用**三层沙箱**（AST 预验证 → `exec()` 受限内置函数 → 子进程隔离 + 30s 超时），拦截 `import/eval/exec/open/__class__` 等攻击，并限制输入数据量、图片数量和图片大小；Agent 训练自动调用完整的训练前校验（缺失率/类别数/特征合法性），与路由层一致；代码执行失败时返回结构化中文错误

## 开发者

Jiayang Li
