# Indeterminate 数据科学工作台

这是一个已经迁移到 **React + Flask** 的数据科学 Web 应用。当前版本不再使用 Streamlit：前端在 `frontend/`，后端 API 在 `backend/`，生产模式下 Flask 可以直接提供 React 构建后的页面。

## 当前状态

已迁移并可用的主要功能：

- 首页 / 工作台：后端连接状态、最近 session、模型状态概览。
- 数据上传：支持 CSV / Excel，上传后创建后端 session。
- 数据预览：表格预览、字段类型、缺失值、异常值概览。
- 数据清洗：行列操作、重命名、类型转换、缺失值处理、异常值处理、标准化/归一化、类别编码、噪声、计算列、自定义表达式、PCA。
- 数据处理历史：后端 session 保存处理快照，支持撤销、重做、保存流水线、导入/导出流水线 JSON 和重复应用流水线。
- 数据可视化：使用 Plotly 的 React 图表工作台，支持散点、折线、面积、柱状、直方图、箱线、小提琴、密度热力图、饼图/环图、散点矩阵、相关热力图、3D 散点等。
- 模型页面：回归、分类、决策树、聚类、自定义 MLP 均已拆成独立工作区，并统一为“训练配置 / 结果分析 / 预测 / 批量预测”一类的页面结构。
- 模型结果：版本列表、激活、删除、指标面板；分类和决策树分类支持 classification report。
- 训练任务状态：训练接口会记录任务状态，前端模型页可查看最近训练任务、成功/失败、耗时和结果版本；回归、分类、决策树、聚类和 MLP 训练均支持 `?async=1` 后台任务。
- 本地账号：支持注册、登录、退出；未登录不能访问 `/api/*` 业务接口。
- 用户隔离：上传 session、数据处理历史、模型版本、模型激活状态和训练任务列表均按账号隔离。
- 任务队列：训练任务进入进程内队列，支持全局并发限制、单用户并发限制和排队状态。
- 任务取消：前端训练任务列表支持取消排队/运行中任务；排队任务会立即取消，运行中任务会标记为取消并在当前训练函数返回后结束。
- 基础资源保护：支持上传大小、数据集行列数、每账号 session 数量、每账号等待任务数量限制；同步训练请求也不能绕过并发限制。
- 认证保护：登录按“来源 IP + 用户名”统计连续失败，注册按来源 IP 限制频率，超限后返回 `Retry-After`。
- 预测能力：回归、分类、MLP、决策树支持单条和批量预测；K-Means 聚类支持批量预测，DBSCAN 只能查看已训练结果。
- LLM 分析：Direct Chat、Agent Chat、流式输出、图表/图片结果展示、API Key / Base / Model 配置面板。

仍需进一步增强的生产级多人能力：

- 当前账号系统是本地文件存储，适合单机/内网演示；正式部署建议迁移到数据库。
- 当前登录 session 存在内存中，服务重启后需要重新登录。
- 当前任务队列存在内存中，服务重启后任务状态会丢失。
- 运行中任务只能“协作式取消”：不能强制终止正在执行的训练线程，只能阻止结果落库并标记为已取消。
- 还没有角色权限、审计日志、账号禁用、数据库级限流、磁盘/CPU/GPU 配额和完整自动清理策略。

## 环境要求

本机推荐使用已有 conda 环境 `ml0`：

```powershell
conda activate ml0
```

需要：

- Python 3.10+
- Node.js 20+ 或当前 LTS
- npm
- conda 环境 `ml0`，或其他已安装同等 Python 依赖的环境

## 安装依赖

Python 后端依赖：

```powershell
conda run -n ml0 python -m pip install -r requirements.txt
```

React 前端依赖：

```powershell
cd frontend
npm install
```

PyTorch 如需 GPU 版本，请按本机 CUDA 情况单独安装。`backend/requirements.txt` 中保留了基础 `torch>=2.0`。

## 启动方式

### 开发模式

开发时推荐启动 Flask API + Vite dev server：

```powershell
.\dev-react.ps1
```

默认地址：

- Flask API: `http://127.0.0.1:5001`
- React Vite: `http://127.0.0.1:5173`

浏览器打开：

```text
http://127.0.0.1:5173
```

可选参数：

```powershell
.\dev-react.ps1 -CondaEnv ml0 -FlaskPort 5001 -ReactPort 5173
```

停止项目：关闭脚本启动的 Flask 和 React 两个窗口。

### 生产本地模式

先构建 React：

```powershell
.\build-prod.ps1
```

再启动 Flask，由 Flask 同时提供页面和 `/api/*`：

```powershell
.\start-prod.ps1
```

默认打开：

```text
http://127.0.0.1:5001
```

如果要在局域网内临时测试：

```powershell
.\start-prod.ps1 -HostName 0.0.0.0 -Port 5001
```

注意：绑定 `0.0.0.0` 只表示局域网设备可以访问，不代表已经具备账号、权限和用户隔离。

## 常用命令

前端构建：

```powershell
cd frontend
npm run build
```

构建时会看到 Plotly chunk 较大的提示，这是因为图表库按需拆成独立包加载；当前主应用包和图表包已经分离。

后端 API 测试：

```powershell
conda run -n ml0 python -m unittest discover -s tests
```

Python 语法检查：

```powershell
conda run -n ml0 python -m compileall backend tests
```

清理缓存和构建产物：

```powershell
.\clean.ps1
.\clean.ps1 -Force
```

默认只会展示将被清理的缓存和构建产物。确实要删除时加 `-Force`。

如果确认要同时清理上传 session 和训练模型运行数据：

```powershell
.\clean.ps1 -IncludeRuntimeData -Force
```

## 目录结构

```text
MyWeb1/
├── backend/
│   ├── app.py                    # Flask 入口，注册 API，并在生产模式提供 React 静态文件
│   ├── requirements.txt          # 后端 Python 依赖
│   ├── auth_store.py             # 本地账号和登录 cookie session
│   ├── session_store.py          # session 持久化和恢复
│   ├── task_store.py             # 后台任务队列、状态和取消
│   ├── routes/                   # Flask API 路由
│   ├── services/                 # 数据处理、可视化、模型、LLM 业务逻辑
│   ├── models/                   # 模型注册表和模型版本文件
│   ├── auth/                     # 本地账号数据，运行时生成，不提交
│   └── sessions/                 # 上传数据 session 文件
├── frontend/
│   ├── src/
│   │   ├── App.tsx
│   │   ├── components/           # React 工作区页面
│   │   ├── services/api.ts       # API 客户端
│   │   ├── styles.css
│   │   └── types.ts
│   ├── package.json
│   └── vite.config.ts
├── tests/                        # Flask API 和服务测试
├── dev-react.ps1 / .bat          # 开发模式启动 Flask + React
├── build-prod.ps1 / .bat         # 构建 React
├── start-prod.ps1 / .bat         # Flask 提供 React 构建产物
├── clean.ps1 / .bat              # 清理缓存和运行产物
├── requirements.txt              # Python 依赖入口
└── .env.example                  # 环境变量示例
```

## API 概览

基础接口：

- `POST /api/auth/register`
- `POST /api/auth/login`
- `POST /api/auth/logout`
- `GET /api/auth/me`
- `GET /api/health`
- `POST /api/data/upload`
- `DELETE /api/data/<sid>`
- `GET /api/data/<sid>/profile`
- `POST /api/data/<sid>/process`
- `POST /api/data/<sid>/visualize`

模型接口：

- `POST /api/regression/train`
- `POST /api/regression/predict`
- `POST /api/regression/batch_predict`
- `POST /api/classification/train`
- `POST /api/classification/predict`
- `POST /api/classification/batch_predict`
- `POST /api/diy_mlp/train`
- `POST /api/diy_mlp/predict`
- `POST /api/diy_mlp/batch_predict`
- `POST /api/decision_tree/train`
- `POST /api/decision_tree/predict`
- `POST /api/decision_tree/batch_predict`
- `POST /api/clustering/train`
- `POST /api/clustering/predict`
- `POST /api/clustering/batch_predict`

版本和 LLM：

- `GET /api/<model_type>/versions`
- `POST /api/<model_type>/activate`
- `DELETE /api/<model_type>/version/<version_id>`
- `GET /api/tasks`
- `GET /api/tasks/<task_id>`
- `POST /api/tasks/<task_id>/cancel`
- `POST /api/llm/chat`
- `POST /api/llm/agent`

统一错误响应格式：

```json
{
  "ok": false,
  "error": {
    "code": "SESSION_EXPIRED",
    "message": "Session not found or expired",
    "detail": "Upload or restore a session before running this operation."
  }
}
```

## 环境变量

可复制 `.env.example` 为 `.env` 自行配置。

| 变量 | 默认值 | 说明 |
| --- | --- | --- |
| `FLASK_HOST` | `127.0.0.1` | Flask 绑定地址 |
| `FLASK_PORT` | `5001` | Flask 端口 |
| `FLASK_DEBUG` | `0` | 设置为 `1` 开启后端调试 |
| `FRONTEND_DIST` | `frontend/dist` | React 构建产物目录 |
| `CORS_ORIGINS` | `http://127.0.0.1:5173,http://localhost:5173` | 开发模式允许访问 API 的前端来源 |
| `MYWEB1_GLOBAL_TASK_CONCURRENCY` | `2` | 全局最多同时运行的后台训练任务数 |
| `MYWEB1_USER_TASK_CONCURRENCY` | `1` | 单个账号最多同时运行的后台训练任务数 |
| `MYWEB1_MAX_PENDING_TASKS_PER_USER` | `5` | 单账号最多等待或运行中的训练任务数 |
| `MYWEB1_MAX_UPLOAD_MB` | `256` | 单次上传最大 MB 数 |
| `MYWEB1_MAX_DATASET_ROWS` | `500000` | 单个上传数据集最大行数 |
| `MYWEB1_MAX_DATASET_COLUMNS` | `1000` | 单个上传数据集最大列数 |
| `MYWEB1_MAX_SESSIONS_PER_USER` | `10` | 单账号最多保留的活跃数据 session 数 |
| `MYWEB1_LOGIN_MAX_FAILURES` | `5` | 登录失败窗口内允许的最大失败次数 |
| `MYWEB1_LOGIN_WINDOW_SECONDS` | `300` | 登录失败统计窗口，单位秒 |
| `MYWEB1_LOGIN_LOCK_SECONDS` | `900` | 超限后的临时锁定时间，单位秒 |
| `MYWEB1_REGISTER_MAX_ATTEMPTS` | `5` | 单个来源在窗口内最多注册尝试次数 |
| `MYWEB1_REGISTER_WINDOW_SECONDS` | `3600` | 注册频率统计窗口，单位秒 |
| `LLM_API_KEY` | 空 | 默认 LLM API Key |
| `LLM_API_BASE` | 空 | OpenAI 兼容 API Base |
| `LLM_MODEL` | 空 | 默认模型名 |
| `LLM_ALLOW_LOCAL_API_BASE` | `0` | 本地开发私有 LLM 地址开关 |

## 数据和模型文件

- `backend/sessions/` 存放上传数据 session，属于运行数据。
- `backend/auth/` 存放本地账号文件，属于运行数据，不应提交。
- `backend/models/registry.py` 是模型注册表代码，需要保留。
- `backend/models/registry.json` 和各模型子目录是训练产物，属于运行数据。
- `.gitignore` 已排除 session、模型文件、前端构建产物、缓存和本地环境文件。

## 当前验证结果

当前版本应至少通过：

```powershell
conda run -n ml0 python -m unittest discover -s tests
conda run -n ml0 python -m compileall backend tests
cd frontend
npm run build
```

测试里会打印两段预期异常日志，分别来自“不安全自定义表达式”和“非法热力图配置”的负向用例；只要最终显示 `OK` 就是通过。当前 smoke tests 覆盖了登录保护、登录限流、账号隔离、资源配额、任务取消、上传、处理历史、撤销/重做、流水线、可视化、模型、训练任务状态和 LLM 基础接口。

## 下一步建议

优先级建议：

1. 继续补齐旧 Streamlit 里还没搬完的高级交互细节，例如图表筛选、模型解释和报告生成。
2. 给长训练增加真实进度百分比和阶段日志，让任务状态面板更可解释。
3. 将账号、登录 session 和任务状态迁移为数据库表，避免服务重启后丢失。
4. 将当前基础配额扩展为磁盘、CPU、GPU 和模型存储配额，并增加审计日志和账号管理。
5. 正式部署时启用 HTTPS，并将登录 cookie 改为 `Secure`。
