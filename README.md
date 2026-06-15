# Indeterminate 数据科学工作台

这是一个基于 **React + Flask** 的本地数据科学 Web 应用，用于完成数据上传、清洗、可视化、模型训练、预测和 LLM 辅助分析。前端主线是 `frontend/` 下的 React/Vite 应用，后端主线是 `backend/` 下的 Flask API。

## 当前状态

已经可用的 React 功能：

- 工作台首页：后端状态、最近 session、模型概览。
- 数据上传：CSV / Excel 上传，创建 Flask session。
- 数据预览：字段类型、缺失值、异常值概览。
- 数据处理：行列操作、重命名、类型转换、统计指标、缺失值处理、异常值处理、标准化/归一化、类别编码、添加噪声、计算列、自定义表达式、PCA。
- 数据可视化：React 图表工作台。
- 模型训练：决策树、聚类、回归、分类、自定义 MLP。
- 模型管理：版本列表、激活、删除、单条预测、批量预测。
- LLM 分析：Direct Chat、Agent Chat、流式输出、工具调用、图表/图片结果展示。
- 生产访问：React 构建后由 Flask 同端口提供页面和 `/api/*`。

还没有完成的生产级多人能力：

- 没有账号、登录、权限。
- session 和模型版本还没有按用户隔离。
- 训练任务仍是同步请求，长训练后续应改为后台任务。
- 没有资源配额、任务取消、队列、审计日志。

## 环境要求

本机推荐使用已有 conda 环境：

```powershell
conda activate ml0
```

需要：

- Python 3.10+
- Node.js 20+ 或当前 LTS
- npm
- conda 环境 `ml0`，或其他包含项目依赖的 Python 环境

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

PyTorch 如需 GPU 版本，请按本机 CUDA 情况单独安装。`backend/requirements.txt` 中保留了基础 `torch>=2.0` 依赖。

## 启动方式

### 开发模式

推荐开发时使用：

```powershell
.\dev-react.ps1
```

默认会启动：

- Flask API: `http://127.0.0.1:5001`
- React Vite: `http://127.0.0.1:5173`

打开：

```text
http://127.0.0.1:5173
```

可选参数：

```powershell
.\dev-react.ps1 -CondaEnv ml0 -FlaskPort 5001 -ReactPort 5173
```

停止项目：关闭脚本启动的两个服务窗口。

### 生产本地模式

先构建 React：

```powershell
.\build-prod.ps1
```

再启动 Flask，由 Flask 同时提供页面和 API：

```powershell
.\start-prod.ps1
```

打开：

```text
http://127.0.0.1:5001
```

局域网测试：

```powershell
.\start-prod.ps1 -HostName 0.0.0.0 -Port 5001
```

注意：绑定 `0.0.0.0` 只是让局域网设备可以访问，不等于已经具备认证和用户隔离。

## 常用命令

构建前端：

```powershell
cd frontend
npm run build
```

Python 测试：

```powershell
conda run -n ml0 python -m unittest discover -s tests
```

Python 语法检查：

```powershell
conda run -n ml0 python -m compileall backend tests
```

清理缓存和运行产物：

```powershell
.\clean.ps1 -WhatIf
.\clean.ps1 -Force
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
│   │   ├── components/           # 各 React 工作台页面
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

主要接口：

- `GET /api/health`
- `POST /api/data/upload`
- `GET /api/data/<sid>/profile`
- `POST /api/data/<sid>/process`
- `POST /api/data/<sid>/visualize`
- `POST /api/regression/train`
- `POST /api/regression/predict`
- `POST /api/regression/batch_predict`
- `POST /api/classification/*`
- `POST /api/diy_mlp/*`
- `POST /api/decision_tree/*`
- `POST /api/clustering/*`
- `POST /api/llm/chat`
- `POST /api/llm/agent`
- `GET /api/<model_type>/versions`
- `POST /api/<model_type>/activate`
- `DELETE /api/<model_type>/version/<version_id>`

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

- `backend/sessions/` 存放上传数据的 session。
- `backend/models/` 存放训练后的模型版本。
- 这些文件属于运行数据，不应随便提交。
- 删除模型前请确认不再需要对应版本。

## 下一步建议

优先级建议：

1. 继续补齐 React 页面里的旧功能细节。
2. 将长训练任务改为后台任务，并增加任务状态轮询。
3. 增强 session id，并引入用户/工作区隔离。
4. 加入认证、权限、资源限制和日志。
5. 决定是否将项目打包成可部署服务，使用 Nginx 或其他反向代理。
