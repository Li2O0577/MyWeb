"""LLM chat proxy with SSE streaming support."""
import json
import os
import re
import requests

# 允许的 API Base 域名白名单 — 防止 SSRF 攻击
_ALLOWED_HOSTS = [
    # OpenAI + Azure
    "api.openai.com",
    r".*\.openai\.azure\.com",
    # 国内主流 LLM 提供商
    "api.deepseek.com",
    "dashscope.aliyuncs.com",         # 阿里通义千问
    "open.bigmodel.cn",               # 智谱 GLM
    "api.moonshot.cn",                # Moonshot/Kimi
    "api.baichuan-ai.com",            # 百川
    "api.minimax.chat",               # MiniMax
    "api.zhipuai.cn",                 # 智谱 AI
    # 本地开发
    "localhost",
    "127.0.0.1",
    r"192\.168\..*",
    r"10\..*",
    r"172\.(1[6-9]|2[0-9]|3[0-1])\..*",
]

_MAX_MESSAGE_LENGTH = 32000  # max chars per message to avoid abuse
_MAX_AGENT_MESSAGES = 30
_MAX_AGENT_TOOL_CALLS = 12
_MAX_TOOL_RESULT_CHARS = 6000


def _validate_model_name(model):
    if not isinstance(model, str) or not model.strip():
        return False, "模型名称不能为空"
    if len(model) > 128:
        return False, "模型名称过长，请检查配置。"
    return True, None


def _validate_messages(messages):
    if not isinstance(messages, list) or len(messages) == 0:
        return False, "messages 不能为空"
    if len(messages) > _MAX_AGENT_MESSAGES:
        return False, f"对话轮数过多，请清空对话或缩短上下文（最多 {_MAX_AGENT_MESSAGES} 条消息）。"
    for i, msg in enumerate(messages):
        if not isinstance(msg, dict):
            return False, f"messages[{i}] 格式无效"
        role = msg.get("role")
        if role not in {"system", "user", "assistant", "tool"}:
            return False, f"messages[{i}] 的 role 无效"
        content = msg.get("content", "")
        if not isinstance(content, str):
            return False, f"messages[{i}] 内容必须是文本"
        if len(content) > _MAX_MESSAGE_LENGTH:
            return False, f"messages[{i}] 内容过长，上限 {_MAX_MESSAGE_LENGTH} 字符"
    return True, None


def _trim_messages(messages):
    return messages[-_MAX_AGENT_MESSAGES:] if len(messages) > _MAX_AGENT_MESSAGES else messages


def _safe_int(value, default, lower, upper):
    try:
        value = int(value)
    except (TypeError, ValueError):
        value = default
    return min(max(value, lower), upper)


def _safe_float(value, default, lower, upper):
    try:
        value = float(value)
    except (TypeError, ValueError):
        value = default
    return min(max(value, lower), upper)


def _validate_api_base(api_base):
    """Check api_base against the whitelist. Returns (ok, error_message)."""
    from urllib.parse import urlparse

    if not api_base or not isinstance(api_base, str):
        return False, "api_base 不能为空"
    if len(api_base) > 256:
        return False, "api_base 长度不能超过 256 字符"

    try:
        parsed = urlparse(api_base)
    except Exception:
        return False, f"无法解析 api_base: {api_base}"

    hostname = parsed.hostname or ""
    if not hostname:
        return False, f"api_base 缺少有效主机名: {api_base}"

    for pattern in _ALLOWED_HOSTS:
        if re.fullmatch(pattern, hostname):
            return True, None

    return False, f"不允许的 API 主机: {hostname}。请使用受支持的 LLM 提供商。"


def stream_chat(api_base, api_key, model, messages, temperature=0.7, timeout=180):
    """Generator that yields text chunks from an OpenAI-compatible chat API.

    If api_key is empty, falls back to LLM_API_KEY environment variable.
    """
    ok, err = _validate_api_base(api_base)
    if not ok:
        yield None, err
        return

    # 优先使用传入的 key，否则从环境变量读取
    effective_key = api_key or os.environ.get("LLM_API_KEY", "")
    if not effective_key:
        yield None, "未提供 API Key。请在设置中填写，或设置环境变量 LLM_API_KEY。"
        return

    ok, msg_err = _validate_messages(messages)
    if not ok:
        yield None, msg_err
        return
    ok, model_err = _validate_model_name(model)
    if not ok:
        yield None, model_err
        return
    messages = _trim_messages(messages)
    try:
        temperature = float(temperature)
    except (TypeError, ValueError):
        temperature = 0.7
    temperature = min(max(temperature, 0.0), 2.0)
    try:
        timeout = int(timeout)
    except (TypeError, ValueError):
        timeout = 180
    timeout = min(max(timeout, 30), 240)

    try:
        resp = requests.post(
            f"{api_base.rstrip('/')}/chat/completions",
            headers={
                "Authorization": f"Bearer {effective_key}",
                "Content-Type": "application/json"
            },
            json={
                "model": model,
                "messages": messages,
                "temperature": temperature,
                "stream": True
            },
            timeout=timeout,
            stream=True
        )
    except Exception as e:
        yield None, f"无法连接到 LLM API: {e}"
        return

    if resp.status_code != 200:
        yield None, f"大模型接口请求失败（HTTP {resp.status_code}）：{resp.text[:500]}"
        return

    for line in resp.iter_lines(decode_unicode=True):
        if not line or not line.startswith("data: "):
            continue
        data_str = line[6:]
        if data_str.strip() == "[DONE]":
            break
        try:
            delta = json.loads(data_str)["choices"][0]["delta"]
            if "content" in delta and delta["content"]:
                yield delta["content"], None
        except (json.JSONDecodeError, KeyError, IndexError):
            continue


# ═══════════════════════════════════════════════════════════════════════════════
# Agent Mode — Function Calling
# ═══════════════════════════════════════════════════════════════════════════════

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_data_overview",
            "description": "获取数据集的全局概览：行列数、所有列名、每列数据类型、缺失值统计、数值列的均值/标准差/最小/最大值、以及最强的5对相关性。这是分析任何数据的第一步。",
            "parameters": {"type": "object", "properties": {}, "required": []}
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_column_details",
            "description": "获取指定列的详细信息。数值列返回分布统计和直方图分箱数据；分类列返回唯一值数量、最高频类别及占比。",
            "parameters": {
                "type": "object",
                "properties": {
                    "columns": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "需要分析的列名列表，最多 10 列"
                    }
                },
                "required": ["columns"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "run_regression",
            "description": "训练一个 PyTorch MLP 回归模型来预测连续值目标列。返回测试集 R²、MAE、RMSE 等指标。需要指定目标列名称，目标列必须是数值类型。",
            "parameters": {
                "type": "object",
                "properties": {
                    "target_column": {
                        "type": "string",
                        "description": "要预测的目标列名称（必须是数值列）"
                    },
                    "epochs": {
                        "type": "integer",
                        "description": "训练轮数，默认 200，范围 50-500"
                    },
                    "learning_rate": {
                        "type": "number",
                        "description": "学习率，默认 0.001"
                    }
                },
                "required": ["target_column"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "run_classification",
            "description": "训练一个 PyTorch MLP 分类模型来预测离散标签列。返回准确率、混淆矩阵、每个类别的精确率和召回率。目标列可以是数值（自动识别类别数）或文本。",
            "parameters": {
                "type": "object",
                "properties": {
                    "target_column": {
                        "type": "string",
                        "description": "要预测的目标列名称（分类标签）"
                    },
                    "epochs": {
                        "type": "integer",
                        "description": "训练轮数，默认 200，范围 50-500"
                    },
                    "learning_rate": {
                        "type": "number",
                        "description": "学习率，默认 0.001"
                    }
                },
                "required": ["target_column"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "run_clustering",
            "description": "对数据执行聚类分析。支持 K-means（需指定簇数）和 DBSCAN（自动发现簇）。返回每个样本的簇标签、轮廓系数、PCA 降维坐标。自动对所有数值列进行标准化后聚类。",
            "parameters": {
                "type": "object",
                "properties": {
                    "algorithm": {
                        "type": "string",
                        "enum": ["kmeans", "dbscan"],
                        "description": "聚类算法：kmeans 需要指定簇数，dbscan 自动确定簇数"
                    },
                    "n_clusters": {
                        "type": "integer",
                        "description": "K-means 的簇数，默认 3，范围 2-10"
                    },
                    "eps": {
                        "type": "number",
                        "description": "DBSCAN 的邻域半径，默认 0.5。增大可获得更大的簇"
                    },
                    "min_samples": {
                        "type": "integer",
                        "description": "DBSCAN 的最小样本数，默认 5"
                    }
                },
                "required": ["algorithm"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "run_correlation_analysis",
            "description": "计算所有数值列之间的皮尔逊相关系数矩阵，返回绝对值最大的相关对。用于发现变量间的线性关系。",
            "parameters": {
                "type": "object",
                "properties": {
                    "top_n": {
                        "type": "integer",
                        "description": "返回前 N 对最相关的列对，默认 10"
                    }
                }
            }
        }
    }
]

_MAX_AGENT_ITERATIONS = 10  # 防止无限循环


def _sanitize_tool_args(tool_name, args, df):
    """Clamp LLM-provided tool args before executing local tools."""
    if not isinstance(args, dict):
        args = {}

    if tool_name == "get_column_details":
        columns = args.get("columns", [])
        if isinstance(columns, str):
            columns = [columns]
        if not isinstance(columns, list):
            columns = []
        return {"columns": [c for c in columns if c in df.columns][:10]}

    if tool_name in {"run_regression", "run_classification"}:
        target = args.get("target_column", "")
        return {
            "target_column": target if target in df.columns else "",
            "epochs": _safe_int(args.get("epochs", 120), 120, 50, 200),
            "learning_rate": _safe_float(args.get("learning_rate", 0.001), 0.001, 1e-5, 0.05),
        }

    if tool_name == "run_clustering":
        algorithm = args.get("algorithm", "kmeans")
        if algorithm not in {"kmeans", "dbscan"}:
            algorithm = "kmeans"
        return {
            "algorithm": algorithm,
            "n_clusters": _safe_int(args.get("n_clusters", 3), 3, 2, 10),
            "eps": _safe_float(args.get("eps", 0.5), 0.5, 0.05, 5.0),
            "min_samples": _safe_int(args.get("min_samples", 5), 5, 2, 50),
        }

    if tool_name == "run_correlation_analysis":
        return {"top_n": _safe_int(args.get("top_n", 10), 10, 3, 30)}

    return args


def _build_system_prompt(df):
    """根据数据集动态构建系统提示词。"""
    import numpy as np
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
    missing = int(df.isna().sum().sum())

    tool_descriptions = []
    for t in TOOLS:
        tinfo = t["function"]
        tool_descriptions.append(f"- **{tinfo['name']}**: {tinfo['description']}")

    return f"""你是本数据分析平台内置的 AI 分析师。你**有实际操作能力**——可以调用工具来执行分析，而不只是给建议。

## 当前数据集
- 行数: {len(df)}, 列数: {len(df.columns)}
- 数值列 ({len(numeric_cols)}): {', '.join(str(c) for c in numeric_cols[:30])}{'...' if len(numeric_cols) > 30 else ''}
- 分类/文本列 ({len(cat_cols)}): {', '.join(str(c) for c in cat_cols[:15])}{'...' if len(cat_cols) > 15 else ''}
- 缺失值总数: {missing}

## 可用工具
{chr(10).join(tool_descriptions)}

## 分析策略
1. 首先调用 `get_data_overview` 了解数据的统计特征和相关性
2. 根据数据特征和目标选择合适的分析方法：
   - 如果需要预测连续值 → `run_regression`
   - 如果需要预测分类标签 → `run_classification`
   - 如果探索数据结构和分组 → `run_clustering`
   - 如果发现强相关 → 用 `run_correlation_analysis` 深入分析
3. 综合所有工具返回的结果，给出具体的、有数据支撑的结论
4. 如果某个工具执行失败，分析原因并尝试调整参数

## 规则
- 必须实际调用工具来推进分析，**不要只给文字建议**
- 每次回复至少调用一个工具
- 分析完一个方向后，考虑是否需要从另一个角度继续
- 模型训练完成后，解释指标的含义（R² 越接近 1 越好，MAE/RMSE 越小越好等）
- 用中文回复，输出清晰有层次的 Markdown 分析报告"""


def _execute_tool(tool_name, args, df, session_id=""):
    """Execute a tool call and return the result string. All exceptions are caught."""
    import io
    import traceback
    import numpy as np
    import pandas as pd

    try:
        if tool_name == "get_data_overview":
            from services.data_service import build_data_summary
            return build_data_summary(df, max_chars=6000)

        elif tool_name == "get_column_details":
            columns = args.get("columns", [])[:10]
            buf = io.StringIO()
            for col in columns:
                if col not in df.columns:
                    buf.write(f"**{col}**: 列不存在\n\n")
                    continue
                s = df[col]
                buf.write(f"### {col} (dtype: {s.dtype})\n")
                buf.write(f"- 缺失: {s.isna().sum()}, 缺失率: {s.isna().mean():.1%}\n")
                if pd.api.types.is_numeric_dtype(s):
                    buf.write(f"- 均值: {s.mean():.2f}, 中位数: {s.median():.2f}\n")
                    buf.write(f"- 标准差: {s.std():.2f}, 最小: {s.min():.2f}, 最大: {s.max():.2f}\n")
                    buf.write(f"- Q25: {s.quantile(0.25):.2f}, Q75: {s.quantile(0.75):.2f}\n")
                    skew = s.skew()
                    buf.write(f"- 偏度: {skew:.2f} ({'右偏' if skew > 0.5 else '左偏' if skew < -0.5 else '近似对称'})\n")
                else:
                    vc = s.value_counts()
                    buf.write(f"- 唯一值数: {s.nunique()}\n")
                    top_n = min(5, len(vc))
                    for i in range(top_n):
                        buf.write(f"  - {vc.index[i]}: {vc.iloc[i]} ({vc.iloc[i]/len(s):.1%})\n")
                    if len(vc) > top_n:
                        buf.write(f"  ... 还有 {len(vc) - top_n} 个类别\n")
                buf.write("\n")
            return buf.getvalue()

        elif tool_name == "run_regression":
            target_col = args["target_column"]
            if target_col not in df.columns:
                return f"错误: 列 '{target_col}' 不存在。可用列: {', '.join(str(c) for c in df.columns)}"
            if not pd.api.types.is_numeric_dtype(df[target_col]):
                return f"错误: '{target_col}' 不是数值列，无法用于回归。请选择一个连续数值列作为目标。"

            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            feature_cols = [c for c in numeric_cols if c != target_col]
            if len(feature_cols) < 1:
                return f"错误: 没有可用的特征列（除目标列 '{target_col}' 外无其他数值列）。"

            from services.regression_service import train as regression_train
            epochs = min(max(args.get("epochs", 200), 50), 500)
            lr = args.get("learning_rate", 0.001)

            result, err = regression_train(
                df, target_col, feature_cols,
                hidden1=max(128, len(feature_cols)),
                hidden2=max(64, len(feature_cols) // 2),
                dropout_rate=0.2,
                learning_rate=lr, epochs=epochs, batch_size=32,
                device_str="cpu",
                dataset_name=getattr(df, 'attrs', {}).get('source_name', '') or "",
                session_id=session_id
            )
            if err:
                return f"回归训练失败: {err}"

            return json.dumps({
                "features_used": feature_cols,
                "target": target_col,
                "epochs_actual": min(epochs, len(result.get("train_losses", []))),
                "metrics": {
                    "R²": round(result["r2"], 4),
                    "MAE": round(result["mae"], 4),
                    "RMSE": round(result["rmse"], 4)
                },
                "interpretation": {
                    "R²": "决定系数，越接近 1 越好，表示模型解释了多大比例的目标变量方差",
                    "MAE": "平均绝对误差，预测值与真实值的平均差距",
                    "RMSE": "均方根误差，对大误差更敏感，始终 >= MAE"
                },
                "train_losses": result.get("train_losses", [])[-5:],
                "val_losses": result.get("val_losses", [])[-5:]
            }, ensure_ascii=False)

        elif tool_name == "run_classification":
            target_col = args["target_column"]
            if target_col not in df.columns:
                return f"错误: 列 '{target_col}' 不存在。可用列: {', '.join(str(c) for c in df.columns)}"

            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            feature_cols = [c for c in numeric_cols if c != target_col]
            if len(feature_cols) < 1:
                return f"错误: 没有可用的数值特征列。"

            from services.classification_service import train as classification_train
            epochs = min(max(args.get("epochs", 200), 50), 500)
            lr = args.get("learning_rate", 0.001)

            result, err = classification_train(
                df, target_col, feature_cols,
                hidden1=max(128, len(feature_cols)),
                hidden2=max(64, len(feature_cols) // 2),
                dropout_rate=0.2,
                learning_rate=lr, epochs=epochs, batch_size=32,
                device_str="cpu",
                dataset_name="", session_id=session_id
            )
            if err:
                return f"分类训练失败: {err}"

            cm = result["cm"]
            return json.dumps({
                "features_used": feature_cols,
                "target": target_col,
                "num_classes": result["n_classes"],
                "classes": result.get("label_names", []),
                "metrics": {
                    "accuracy": round(result["acc"], 4),
                },
                "confusion_matrix": cm,
                "train_losses": result.get("train_losses", [])[-5:],
                "val_losses": result.get("val_losses", [])[-5:]
            }, ensure_ascii=False)

        elif tool_name == "run_clustering":
            algorithm = args.get("algorithm", "kmeans")
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if len(numeric_cols) < 2:
                return f"错误: 需要至少 2 个数值列才能聚类，当前只有 {len(numeric_cols)} 个。"

            from services.clustering_service import train as clustering_train, elbow as clustering_elbow

            if algorithm == "kmeans":
                n_clusters = min(max(args.get("n_clusters", 3), 2), 10)
                params = {"n_clusters": n_clusters}

                # Also compute elbow
                try:
                    elbow_result = clustering_elbow(df, numeric_cols, min(10, len(df) // 10))
                except Exception:
                    elbow_result = None

                result, err = clustering_train(df, numeric_cols, "kmeans", params, "", session_id)
                if err:
                    return f"聚类失败: {err}"

                return json.dumps({
                    "algorithm": "kmeans",
                    "n_clusters": n_clusters,
                    "features": numeric_cols,
                    "cluster_sizes": result.get("cluster_counts", {}),
                    "silhouette_score": result.get("silhouette"),
                    "inertia": result.get("inertia"),
                    "elbow": {"ks": elbow_result["ks"], "inertias": elbow_result["inertias"]} if elbow_result else None,
                    "silhouette_guide": "轮廓系数范围 [-1, 1]，越接近 1 表示聚类质量越好"
                }, ensure_ascii=False)

            else:  # dbscan
                eps = args.get("eps", 0.5)
                min_samples = args.get("min_samples", 5)
                params = {"eps": eps, "min_samples": min_samples}
                result, err = clustering_train(df, numeric_cols, "dbscan", params, "", session_id)
                if err:
                    return f"DBSCAN 聚类失败: {err}"

                return json.dumps({
                    "algorithm": "dbscan",
                    "eps": eps, "min_samples": min_samples,
                    "features": numeric_cols,
                    "n_clusters_found": result.get("n_found", 0),
                    "cluster_sizes": result.get("cluster_counts", {}),
                    "silhouette_score": result.get("silhouette"),
                    "noise_points": result.get("cluster_counts", {}).get("-1", 0),
                    "silhouette_guide": "轮廓系数范围 [-1, 1]，越接近 1 越好。噪声点（簇 -1）已被排除在轮廓系数计算之外"
                }, ensure_ascii=False)

        elif tool_name == "run_correlation_analysis":
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if len(numeric_cols) < 2:
                return f"错误: 需要至少 2 个数值列才能计算相关性，当前只有 {len(numeric_cols)} 个。"

            top_n = args.get("top_n", 10)
            corr = df[numeric_cols].corr()
            # Only keep upper triangle, exclude diagonal
            mask = np.triu(np.ones(corr.shape), k=1).astype(bool)
            pairs_data = []
            for i in range(len(numeric_cols)):
                for j in range(i + 1, len(numeric_cols)):
                    pairs_data.append({
                        "列 A": numeric_cols[i],
                        "列 B": numeric_cols[j],
                        "相关系数": corr.iloc[i, j]
                    })
            pairs = pd.DataFrame(pairs_data)
            pairs["绝对值"] = pairs["相关系数"].abs()
            top = pairs.sort_values("绝对值", ascending=False).head(top_n)

            buf = io.StringIO()
            buf.write(f"### 相关性分析 (Top {top_n} 相关对)\n\n")
            for _, row in top.iterrows():
                r = row["相关系数"]
                if abs(r) > 0.7:
                    strength = "强" + ("正相关" if r > 0 else "负相关")
                elif abs(r) > 0.4:
                    strength = "中等" + ("正相关" if r > 0 else "负相关")
                else:
                    strength = "弱" + ("正相关" if r > 0 else "负相关")
                buf.write(f"- **{row['列 A']}** vs **{row['列 B']}**: r={r:.4f} ({strength})\n")
            return buf.getvalue()

        else:
            return f"未知工具: {tool_name}"

    except Exception:
        return f"工具执行出错: {traceback.format_exc()[-500:]}"


def _resolve_key(api_key):
    return api_key or os.environ.get("LLM_API_KEY", "")


def agent_chat(api_base, api_key, model, messages, session_id):
    """Agent mode generator — LLM can call tools autonomously.

    Yields SSE event dicts:
      {"status": "thinking"|"tool_call"|"tool_result", ...}
      {"chunk": "text", ...}
      {"done": True, ...}
    """
    import numpy as np
    from session_store import get_session

    # Validate
    ok, err = _validate_api_base(api_base)
    if not ok:
        yield {"error": {"code": "INVALID_API_BASE", "message": err}}
        return
    ok, model_err = _validate_model_name(model)
    if not ok:
        yield {"error": {"code": "INVALID_MODEL", "message": model_err}}
        return
    ok, msg_err = _validate_messages(messages)
    if not ok:
        yield {"error": {"code": "INVALID_MESSAGES", "message": msg_err}}
        return

    effective_key = _resolve_key(api_key)
    if not effective_key:
        yield {"error": {"code": "NO_API_KEY", "message": "未提供 API Key"}}
        return

    df = get_session(session_id)
    if df is None:
        yield {"error": {"code": "SESSION_EXPIRED", "message": "当前数据会话已失效，请重新上传或同步数据后再试。"}}
        return

    system_prompt = _build_system_prompt(df)
    messages = _trim_messages(messages)

    # Build full message list with system prompt
    full_messages = [{"role": "system", "content": system_prompt}]
    for msg in messages:
        full_messages.append(dict(msg))

    yield {"status": "thinking", "message": "正在分析数据..."}

    url = f"{api_base.rstrip('/')}/chat/completions"
    headers = {
        "Authorization": f"Bearer {effective_key}",
        "Content-Type": "application/json"
    }

    tool_results_context = []
    tool_call_count = 0

    for iteration in range(_MAX_AGENT_ITERATIONS):
        try:
            # Non-streaming call to check for tool calls
            resp = requests.post(url, headers=headers, json={
                "model": model,
                "messages": full_messages,
                "tools": TOOLS,
                "temperature": 0.7
            }, timeout=180)
        except Exception as e:
            yield {"error": {"code": "LLM_CONNECTION_FAILED", "message": f"无法连接到 LLM API: {e}"}}
            return

        if resp.status_code != 200:
            yield {"error": {"code": "LLM_API_ERROR", "message": f"大模型接口请求失败（HTTP {resp.status_code}）：{resp.text[:500]}"}}
            return

        try:
            body = resp.json()
            choice = body["choices"][0]
            msg = choice["message"]
        except (json.JSONDecodeError, KeyError, IndexError):
            yield {"error": {"code": "LLM_RESPONSE_PARSE", "message": "无法解析 LLM 响应"}}
            return

        # If LLM wants to call tools
        tool_calls = msg.get("tool_calls", [])
        if tool_calls:
            # Add assistant message (with tool_calls) to history
            full_messages.append(msg)

            for tc in tool_calls:
                if tool_call_count >= _MAX_AGENT_TOOL_CALLS:
                    yield {"error": {"code": "AGENT_TOOL_LIMIT", "message": f"本次 Agent 分析已达到 {_MAX_AGENT_TOOL_CALLS} 次工具调用上限，请缩小问题范围后继续。"}}
                    return
                func = tc["function"]
                tool_name = func["name"]
                try:
                    tool_args = json.loads(func.get("arguments", "{}"))
                except json.JSONDecodeError:
                    tool_args = {}
                tool_args = _sanitize_tool_args(tool_name, tool_args, df)
                tool_call_count += 1

                yield {"status": "tool_call", "tool": tool_name, "args": tool_args}

                result_str = _execute_tool(tool_name, tool_args, df, session_id)
                # Truncate very long results
                if len(result_str) > _MAX_TOOL_RESULT_CHARS:
                    result_str = result_str[:_MAX_TOOL_RESULT_CHARS] + "\n... (结果已截断)"

                yield {"status": "tool_result", "tool": tool_name, "result": result_str}

                full_messages.append({
                    "role": "tool",
                    "tool_call_id": tc["id"],
                    "content": result_str
                })
                tool_results_context.append({"tool": tool_name, "summary": result_str[:200]})

            continue  # back to LLM for next decision

        # No tool calls — LLM wants to respond with text
        # If there's content in the non-streaming response, use it
        content = msg.get("content", "")
        if content:
            yield {"chunk": content}
            yield {"done": True, "tools_used": len(tool_results_context)}
            return

        # If no content and no tool calls (shouldn't happen), prompt LLM again
        full_messages.append({"role": "user", "content": "请继续分析。调用一个工具来获取更多信息。"})
        # continue loop

    yield {"chunk": "\n\n> ⚠️ 已达到最大分析步数限制，分析可能不完整。请重新提问以继续。"}
    yield {"done": True, "tools_used": len(tool_results_context)}
