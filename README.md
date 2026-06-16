# Indeterminate 数据科学工作台

这是一个已经迁移到 **React + Flask** 的数据科学 Web 应用。当前版本不再使用 Streamlit：前端在 `frontend/`，后端 API 在 `backend/`，生产模式下 Flask 可以直接提供 React 构建后的页面。

## 当前状态

已迁移并可用的主要功能：

- 首页 / 工作台：后端连接状态、最近 session、模型状态概览。
- 数据上传：支持 CSV / Excel，上传后创建后端 session。
- 数据预览：表格预览、字段类型、缺失值、异常值概览。
- 数据清洗：行列操作、重命名、类型转换、缺失值处理、异常值处理、标准化/归一化、类别编码、噪声、计算列、自定义表达式、PCA。
- 数据处理历史：后端 session 保存处理快照，支持撤销、重做、保存流水线和重复应用流水线。
- 数据可视化：使用 Plotly 的 React 图表工作台，支持散点、折线、面积、柱状、直方图、箱线、小提琴、密度热力图、饼图/环图、散点矩阵、相关热力图、3D 散点等。
- 模型页面：回归、分类、决策树、聚类、自定义 MLP 均已拆成独立工作区，并统一为“训练配置 / 结果分析 / 预测 / 批量预测”一类的页面结构。
- 模型结果：版本列表、激活、删除、指标面板；分类和决策树分类支持 classification report。
- 预测能力：回归、分类、MLP、决策树支持单条和批量预测；K-Means 聚类支持批量预测，DBSCAN 只能查看已训练结果。
- LLM 分析：Direct Chat、Agent Chat、流式输出、图表/图片结果展示、API Key / Base / Model 配置面板。

尚未完成的生产级多人能力：

- 没有账号、登录和权限控制。
- session 和模型版本还没有按用户隔离。
- 训练任务仍以同步请求为主，长训练后续应改为后台任务。
- 还没有任务取消、队列、资源配额、审计日志和自动清理策略。

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
│   ├── session_store.py          # session 持久化和恢复
│   ├── routes/                   # Flask API 路由
│   ├── services/                 # 数据处理、可视化、模型、LLM 业务逻辑
│   ├── models/                   # 模型注册表和模型版本文件
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

- `GET /api/health`
- `POST /api/data/upload`
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
| `LLM_API_KEY` | 空 | 默认 LLM API Key |
| `LLM_API_BASE` | 空 | OpenAI 兼容 API Base |
| `LLM_MODEL` | 空 | 默认模型名 |
| `LLM_ALLOW_LOCAL_API_BASE` | `0` | 本地开发私有 LLM 地址开关 |

## 数据和模型文件

- `backend/sessions/` 存放上传数据 session，属于运行数据。
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

测试里会打印两段预期异常日志，分别来自“不安全自定义表达式”和“非法热力图配置”的负向用例；只要最终显示 `OK` 就是通过。当前 smoke tests 覆盖了上传、处理历史、撤销/重做、流水线、可视化、模型和 LLM 基础接口。

## 下一步建议

优先级建议：

1. 继续补齐旧 Streamlit 里还没搬完的高级交互细节，例如图表筛选、模型解释和报告生成。
2. 将数据处理流水线扩展为可导入/导出的 JSON 文件。
3. 将长训练任务改为后台任务，并增加任务状态轮询。
4. 增强 session id，并引入用户 / 工作区隔离。
5. 加入认证、权限、资源限制、日志和资源清理策略。
