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
]

_LOCAL_ALLOWED_HOSTS = [
    # 本地/私网开发。默认禁用；设置 LLM_ALLOW_LOCAL_API_BASE=1 后启用。
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
_MAX_AGENT_TRAIN_EPOCHS = 120
_MAX_AGENT_TRAIN_ROWS = 5000
_MAX_CODE_INTERPRETER_ROWS = 5000
_MAX_CODE_INTERPRETER_COLS = 80
_MAX_CODE_INTERPRETER_CSV_BYTES = 5 * 1024 * 1024
_MAX_CODE_INTERPRETER_IMAGES = 5
_MAX_CODE_INTERPRETER_IMAGE_B64_CHARS = 8 * 1024 * 1024


def _validate_model_name(model):
    if not isinstance(model, str) or not model.strip():
        return False, "模型名称不能为空"
    if len(model) > 128:
        return False, "模型名称过长，请检查配置。"
    return True, None


def _validate_messages(messages):
    if not isinstance(messages, list) or len(messages) == 0:
        return False, "messages 不能为空"
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


def _prepare_messages(messages):
    """Trim long conversations before validating message shape/content."""
    if not isinstance(messages, list):
        return messages
    return _trim_messages(messages)


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

    if parsed.scheme not in {"http", "https"}:
        return False, "api_base 仅支持 http 或 https 协议"

    hostname = parsed.hostname or ""
    if not hostname:
        return False, f"api_base 缺少有效主机名: {api_base}"

    patterns = list(_ALLOWED_HOSTS)
    if os.environ.get("LLM_ALLOW_LOCAL_API_BASE", "0") == "1":
        patterns.extend(_LOCAL_ALLOWED_HOSTS)

    for pattern in patterns:
        if re.fullmatch(pattern, hostname):
            return True, None

    return False, f"不允许的 API 主机: {hostname}。请使用受支持的 LLM 提供商；如需本地/私网地址，请在开发环境设置 LLM_ALLOW_LOCAL_API_BASE=1。"


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

    messages = _prepare_messages(messages)
    ok, msg_err = _validate_messages(messages)
    if not ok:
        yield None, msg_err
        return
    ok, model_err = _validate_model_name(model)
    if not ok:
        yield None, model_err
        return
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
    },
    {
        "type": "function",
        "function": {
            "name": "generate_chart",
            "description": "创建数据可视化图表（matplotlib）并返回图片内嵌在对话中。支持散点图、折线图、柱状图、直方图、箱线图、相关性热力图、饼图、配对关系图。图表会自动显示在对话中供用户查看。图表标题必须使用英文，避免中文字体缺失。",
            "parameters": {
                "type": "object",
                "properties": {
                    "chart_type": {
                        "type": "string",
                        "enum": ["scatter", "line", "bar", "histogram", "box", "heatmap", "pie", "pairplot"],
                        "description": "图表类型：scatter=散点图, line=折线图, bar=柱状图, histogram=直方图, box=箱线图, heatmap=相关性热力图, pie=饼图, pairplot=成对关系图"
                    },
                    "x_column": {
                        "type": "string",
                        "description": "X轴列名。对于 heatmap/pairplot 可选，其他类型必填"
                    },
                    "y_column": {
                        "type": "string",
                        "description": "Y轴列名。histogram/pie/pairplot/heatmap 不需要"
                    },
                    "color_column": {
                        "type": "string",
                        "description": "颜色分组列名（可选），用于按该列的值着色不同数据点"
                    },
                    "title": {
                        "type": "string",
                        "description": "图表标题，必须使用英文；如果不确定请留空，由系统自动生成英文标题"
                    }
                },
                "required": ["chart_type"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "run_code_interpreter",
            "description": "编写并执行 Python 代码进行自定义数据分析和可视化。代码在隔离子进程中运行（30秒超时），支持 numpy/pandas/matplotlib/scipy/sklearn。数据集通过 df 变量自动注入，matplotlib 图表自动捕获并内嵌显示。适合做复杂的数据探索、统计检验、自定义可视化等。",
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "要执行的 Python 代码。df 变量已自动注入（当前数据集）。已自动导入：numpy(np), pandas(pd), matplotlib.pyplot(plt), scipy.stats(stats), sklearn。plt.show() 或 plt.savefig() 会自动捕获图表。print() 输出会显示在对话中。代码限制 10000 字符。"
                    },
                    "description": {
                        "type": "string",
                        "description": "简要描述这段代码要做什么（5-15字），用于日志和进度显示"
                    }
                },
                "required": ["code"]
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
            "epochs": _safe_int(args.get("epochs", 80), 80, 20, _MAX_AGENT_TRAIN_EPOCHS),
            "learning_rate": _safe_float(args.get("learning_rate", 0.001), 0.001, 1e-5, 0.05),
            "device": args.get("device", "auto"),
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

    if tool_name == "generate_chart":
        chart_type = args.get("chart_type", "scatter")
        if chart_type not in {"scatter", "line", "bar", "histogram", "box", "heatmap", "pie", "pairplot"}:
            chart_type = "scatter"
        x_col = args.get("x_column", "")
        y_col = args.get("y_column", "")
        color_col = args.get("color_column", "")
        return {
            "chart_type": chart_type,
            "x_column": x_col if x_col in df.columns else "",
            "y_column": y_col if y_col in df.columns else "",
            "color_column": color_col if color_col in df.columns else "",
            "title": str(args.get("title", ""))[:100]
        }

    if tool_name == "run_code_interpreter":
        code = args.get("code", "")
        if not isinstance(code, str) or not code.strip():
            return {"code": "", "description": ""}
        if len(code) > 10000:
            code = code[:10000]
        desc = str(args.get("description", ""))[:80]
        return {"code": code, "description": desc}

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

    return f"""你是本数据分析平台内置的 AI 分析师。你**有实际操作能力**——可以调用工具来执行分析、绘制图表、编写代码，而不只是给建议。

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
   - 如果需要**可视化数据分布/关系** → 优先使用 `generate_chart` 创建直观的图表（散点图、直方图、热力图等）
   - 如果需要**复杂的自定义分析**或现有工具无法满足需求 → 使用 `run_code_interpreter` 编写 Python 代码
   - 如果需要预测连续值 → `run_regression`
   - 如果需要预测分类标签 → `run_classification`
   - 如果探索数据结构和分组 → `run_clustering`
   - 如果发现强相关 → 用 `run_correlation_analysis` 深入分析
3. **重要：分析结论中配合可视化** — 在给出数据统计后，主动用 `generate_chart` 生成图表让用户直观感受数据特征
4. 综合所有工具返回的结果，给出具体的、有数据支撑的结论
5. 如果某个工具执行失败，分析原因并尝试调整参数

## 规则
- 必须实际调用工具来推进分析，**不要只给文字建议**
- 每次回复至少调用一个工具
- **主动画图** — 数据探索阶段至少生成 1-2 张图表（分布直方图、相关性热力图、散点图等）
- **图表标题必须使用英文** — matplotlib 环境可能缺少中文字体，调用 `generate_chart` 时 title 使用英文或留空
- 分析完一个方向后，考虑是否需要从另一个角度继续
- 模型训练完成后，解释指标的含义（R² 越接近 1 越好，MAE/RMSE 越小越好等）
- 用中文回复，输出清晰有层次的 Markdown 分析报告"""


def _make_result(text="", images=None):
    """Build a structured tool result."""
    return {"text": text, "images": images or []}


# ═══════════════════════════════════════════════════════════════════════════════
# Chart Generation
# ═══════════════════════════════════════════════════════════════════════════════

def _fig_to_base64(fig):
    """Convert a matplotlib figure to base64 PNG string."""
    import io
    import base64
    import matplotlib.pyplot as _plt
    _sanitize_matplotlib_text(fig)
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=90, bbox_inches='tight')
    _plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')


_AGENT_CHART_FIGSIZE = (5.8, 3.4)
_AGENT_SQUARE_FIGSIZE = (4.4, 4.4)
_AGENT_MAX_FIG_WIDTH = 6.2
_AGENT_MAX_FIG_HEIGHT = 4.8


def _ascii_chart_label(value, fallback):
    """Return an ASCII-only label for chart titles/captions."""
    text = str(value or "").strip()
    if text and text.isascii():
        return text[:60]
    return fallback


def _axis_label(value, fallback):
    """Return an ASCII-only axis/legend label."""
    return _ascii_chart_label(value, fallback)


def _category_labels(values, prefix="Category", limit=None):
    """Return ASCII-only category labels while preserving readable ASCII values."""
    labels = []
    for i, value in enumerate(list(values)[:limit] if limit else values):
        labels.append(_ascii_chart_label(value, f"{prefix} {i + 1}"))
    return labels


def _chart_title(proposed, fallback):
    """Use only English/ASCII chart titles to avoid missing CJK fonts."""
    return _ascii_chart_label(proposed, fallback)


def _bounded_figsize(width, height):
    return (min(width, _AGENT_MAX_FIG_WIDTH), min(height, _AGENT_MAX_FIG_HEIGHT))


def _sanitize_matplotlib_text(fig):
    """Replace non-ASCII chart text with English placeholders before export."""
    for ax_index, ax in enumerate(fig.get_axes(), start=1):
        if not ax.get_title().isascii():
            ax.set_title(f"Chart {ax_index}")
        if ax.get_xlabel() and not ax.get_xlabel().isascii():
            ax.set_xlabel(f"X Axis {ax_index}")
        if ax.get_ylabel() and not ax.get_ylabel().isascii():
            ax.set_ylabel(f"Y Axis {ax_index}")
        for i, tick in enumerate(ax.get_xticklabels(), start=1):
            if tick.get_text() and not tick.get_text().isascii():
                tick.set_text(f"Item {i}")
        for i, tick in enumerate(ax.get_yticklabels(), start=1):
            if tick.get_text() and not tick.get_text().isascii():
                tick.set_text(f"Item {i}")
        legend = ax.get_legend()
        if legend:
            if legend.get_title().get_text() and not legend.get_title().get_text().isascii():
                legend.get_title().set_text("Group")
            for i, text in enumerate(legend.get_texts(), start=1):
                if text.get_text() and not text.get_text().isascii():
                    text.set_text(f"Group {i}")


def _generate_chart(chart_type, x_column, y_column, color_column, title, df):
    """Generate a matplotlib chart and return a structured result with base64 image."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()

    try:
        if chart_type == "scatter":
            if not x_column or not y_column:
                return _make_result(text="散点图需要同时指定 x_column 和 y_column。")
            if x_column not in df.columns or y_column not in df.columns:
                return _make_result(text=f"列不存在。可用列: {', '.join(str(c) for c in df.columns)}")
            chart_title = _chart_title(
                title,
                f"Scatter Plot: {_ascii_chart_label(y_column, 'Y')} vs {_ascii_chart_label(x_column, 'X')}",
            )
            fig, ax = plt.subplots(figsize=_AGENT_CHART_FIGSIZE)
            if color_column and color_column in df.columns:
                unique_vals = df[color_column].dropna().unique()
                if len(unique_vals) > 20:
                    top_vals = df[color_column].value_counts().head(20).index
                    for label, display_label in zip(top_vals, _category_labels(top_vals, "Group")):
                        mask = df[color_column] == label
                        ax.scatter(df.loc[mask, x_column], df.loc[mask, y_column], alpha=0.6, label=display_label, s=20)
                    ax.legend(fontsize=7, title=f"{_axis_label(color_column, 'Group')} (top 20)")
                else:
                    for label, display_label in zip(unique_vals, _category_labels(unique_vals, "Group")):
                        mask = df[color_column] == label
                        ax.scatter(df.loc[mask, x_column], df.loc[mask, y_column], alpha=0.6, label=display_label, s=20)
                    ax.legend(fontsize=8, title=_axis_label(color_column, "Group"))
            else:
                ax.scatter(df[x_column], df[y_column], alpha=0.6, s=20)
            ax.set_xlabel(_axis_label(x_column, "X Axis")); ax.set_ylabel(_axis_label(y_column, "Y Axis"))
            ax.set_title(chart_title, fontsize=10)
            plt.tight_layout()
            b64 = _fig_to_base64(fig)
            return _make_result(
                text=f"已生成散点图：**{y_column}** vs **{x_column}**" + (f"，按 {color_column} 着色" if color_column else ""),
                images=[{"base64": b64, "title": chart_title, "alt": "scatter plot"}]
            )

        elif chart_type == "line":
            if not x_column or not y_column:
                return _make_result(text="折线图需要同时指定 x_column 和 y_column。")
            if x_column not in df.columns or y_column not in df.columns:
                return _make_result(text=f"列不存在。可用列: {', '.join(str(c) for c in df.columns)}")
            chart_title = _chart_title(
                title,
                f"Line Chart: {_ascii_chart_label(y_column, 'Y')} by {_ascii_chart_label(x_column, 'X')}",
            )
            fig, ax = plt.subplots(figsize=_AGENT_CHART_FIGSIZE)
            data = df[[x_column, y_column]].dropna().sort_values(x_column)
            ax.plot(data[x_column], data[y_column], linewidth=1.5)
            ax.set_xlabel(_axis_label(x_column, "X Axis")); ax.set_ylabel(_axis_label(y_column, "Y Axis"))
            ax.set_title(chart_title, fontsize=10)
            plt.tight_layout()
            b64 = _fig_to_base64(fig)
            return _make_result(
                text=f"已生成折线图：**{y_column}** vs **{x_column}**",
                images=[{"base64": b64, "title": chart_title, "alt": "line chart"}]
            )

        elif chart_type == "bar":
            if not x_column:
                return _make_result(text="柱状图需要指定 x_column。")
            if x_column not in df.columns:
                return _make_result(text=f"列 '{x_column}' 不存在。")
            chart_title = _chart_title(
                title,
                f"Bar Chart: {_ascii_chart_label(x_column, 'Category')}",
            )
            fig, ax = plt.subplots(figsize=_AGENT_CHART_FIGSIZE)
            if y_column and y_column in df.columns and pd.api.types.is_numeric_dtype(df[y_column]):
                # Group by x, aggregate y
                grouped = df.groupby(x_column)[y_column].mean().sort_values(ascending=False).head(30)
                ax.bar(range(len(grouped)), grouped.values)
                ax.set_xticks(range(len(grouped)))
                ax.set_xticklabels(_category_labels(grouped.index, "Category"), rotation=45, ha='right', fontsize=8)
                ax.set_ylabel(f"avg({_axis_label(y_column, 'Value')})")
            else:
                vc = df[x_column].value_counts().head(30)
                ax.bar(range(len(vc)), vc.values)
                ax.set_xticks(range(len(vc)))
                ax.set_xticklabels(_category_labels(vc.index, "Category"), rotation=45, ha='right', fontsize=8)
                ax.set_ylabel("Count")
            ax.set_title(chart_title, fontsize=10)
            plt.tight_layout()
            b64 = _fig_to_base64(fig)
            return _make_result(
                text=f"已生成柱状图：**{x_column}**",
                images=[{"base64": b64, "title": chart_title, "alt": "bar chart"}]
            )

        elif chart_type == "histogram":
            if not x_column:
                return _make_result(text="直方图需要指定 x_column。")
            if x_column not in df.columns:
                return _make_result(text=f"列 '{x_column}' 不存在。")
            s = df[x_column].dropna()
            if not pd.api.types.is_numeric_dtype(s):
                return _make_result(text=f"'{x_column}' 不是数值列，无法绘制直方图。")
            chart_title = _chart_title(
                title,
                f"Distribution: {_ascii_chart_label(x_column, 'Feature')}",
            )
            fig, ax = plt.subplots(figsize=_AGENT_CHART_FIGSIZE)
            ax.hist(s, bins=30, alpha=0.7, edgecolor='white')
            ax.set_xlabel(_axis_label(x_column, "Feature")); ax.set_ylabel("Frequency")
            ax.set_title(chart_title, fontsize=10)
            plt.tight_layout()
            b64 = _fig_to_base64(fig)
            return _make_result(
                text=f"已生成直方图：**{x_column}** 分布（均值={s.mean():.2f}, 中位数={s.median():.2f}）",
                images=[{"base64": b64, "title": chart_title, "alt": "histogram"}]
            )

        elif chart_type == "box":
            if not x_column:
                return _make_result(text="箱线图需要指定 x_column。")
            if x_column not in df.columns:
                return _make_result(text=f"列 '{x_column}' 不存在。")
            fig, ax = plt.subplots(figsize=_AGENT_CHART_FIGSIZE)
            if y_column and y_column in df.columns and pd.api.types.is_numeric_dtype(df[y_column]):
                chart_title = _chart_title(
                    title,
                    f"Box Plot: {_ascii_chart_label(y_column, 'Value')} by {_ascii_chart_label(x_column, 'Group')}",
                )
                # Box plot grouped by x
                groups = [df[df[x_column] == val][y_column].dropna().values for val in df[x_column].dropna().unique()[:20]]
                labels = _category_labels(df[x_column].dropna().unique()[:20], "Group")
                ax.boxplot(groups, labels=labels)
                ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
                ax.set_ylabel(_axis_label(y_column, "Value"))
                ax.set_title(chart_title, fontsize=10)
            else:
                chart_title = _chart_title(
                    title,
                    f"Box Plot: {_ascii_chart_label(x_column, 'Feature')}",
                )
                s = df[x_column].dropna()
                if not pd.api.types.is_numeric_dtype(s):
                    return _make_result(text=f"'{x_column}' 不是数值列，无法绘制箱线图。")
                ax.boxplot([s.values], labels=[_axis_label(x_column, "Feature")])
                ax.set_ylabel(_axis_label(x_column, "Feature"))
                ax.set_title(chart_title, fontsize=10)
            plt.tight_layout()
            b64 = _fig_to_base64(fig)
            return _make_result(
                text=f"已生成箱线图：**{x_column}**",
                images=[{"base64": b64, "title": chart_title, "alt": "box plot"}]
            )

        elif chart_type == "heatmap":
            if len(numeric_cols) < 2:
                return _make_result(text="需要至少 2 个数值列才能绘制热力图。")
            cols = numeric_cols[:20]
            corr = df[cols].corr()
            chart_title = _chart_title(title, "Correlation Heatmap")
            fig, ax = plt.subplots(figsize=_bounded_figsize(max(5.2, len(cols) * 0.38), max(3.8, len(cols) * 0.32)))
            im = ax.imshow(corr.values, cmap='coolwarm', vmin=-1, vmax=1, aspect='auto')
            ax.set_xticks(range(len(cols))); ax.set_yticks(range(len(cols)))
            ax.set_xticklabels([_axis_label(c, f"Feature {i + 1}") for i, c in enumerate(cols)], rotation=45, ha='right', fontsize=7)
            ax.set_yticklabels([_axis_label(c, f"Feature {i + 1}") for i, c in enumerate(cols)], fontsize=7)
            plt.colorbar(im, ax=ax, shrink=0.8)
            ax.set_title(chart_title, fontsize=10)
            plt.tight_layout()
            b64 = _fig_to_base64(fig)
            return _make_result(
                text=f"已生成相关性热力图（{len(cols)} 列）",
                images=[{"base64": b64, "title": chart_title, "alt": "heatmap"}]
            )

        elif chart_type == "pie":
            if not x_column:
                return _make_result(text="饼图需要指定 x_column。")
            if x_column not in df.columns:
                return _make_result(text=f"列 '{x_column}' 不存在。")
            vc = df[x_column].value_counts().head(10)
            chart_title = _chart_title(
                title,
                f"Pie Chart: {_ascii_chart_label(x_column, 'Category')}",
            )
            fig, ax = plt.subplots(figsize=_AGENT_SQUARE_FIGSIZE)
            wedges, texts, autotexts = ax.pie(vc.values, labels=_category_labels(vc.index, "Category"), autopct='%1.1f%%',
                                               textprops={'fontsize': 8})
            ax.set_title(chart_title, fontsize=10)
            plt.tight_layout()
            b64 = _fig_to_base64(fig)
            return _make_result(
                text=f"已生成饼图：**{x_column}**（共 {len(vc)} 个类别）",
                images=[{"base64": b64, "title": chart_title, "alt": "pie chart"}]
            )

        elif chart_type == "pairplot":
            # Select columns: use x_column, y_column, color_column hints if provided
            select_cols = []
            for c in [x_column, y_column, color_column]:
                if c and c in numeric_cols and c not in select_cols:
                    select_cols.append(c)
            # Fill up to 5 numeric columns
            for c in numeric_cols:
                if len(select_cols) >= 5:
                    break
                if c not in select_cols:
                    select_cols.append(c)
            select_cols = select_cols[:5]
            if len(select_cols) < 2:
                return _make_result(text="需要至少 2 个数值列才能绘制成对关系图。")

            n = len(select_cols)
            chart_title = _chart_title(title, "Pair Plot")
            fig, axes = plt.subplots(n, n, figsize=_bounded_figsize(n * 1.65, n * 1.45))
            for i in range(n):
                for j in range(n):
                    ax = axes[i][j] if n > 1 else axes
                    if i == j:
                        ax.hist(df[select_cols[i]].dropna(), bins=20, alpha=0.7)
                        ax.set_title(_ascii_chart_label(select_cols[i], f"Feature {i + 1}"), fontsize=7)
                    else:
                        ax.scatter(df[select_cols[j]], df[select_cols[i]], alpha=0.5, s=4)
                    if j == 0:
                        ax.set_ylabel(_axis_label(select_cols[i], f"Feature {i + 1}"), fontsize=7)
                    else:
                        ax.set_yticklabels([])
                    if i == n - 1:
                        ax.set_xlabel(_axis_label(select_cols[j], f"Feature {j + 1}"), fontsize=7)
                    else:
                        ax.set_xticklabels([])
            plt.suptitle(chart_title, fontsize=10)
            plt.tight_layout()
            b64 = _fig_to_base64(fig)
            return _make_result(
                text=f"已生成配对关系图（{len(select_cols)} 列：{', '.join(select_cols)}）",
                images=[{"base64": b64, "title": chart_title, "alt": "pair plot"}]
            )

        return _make_result(text=f"未知图表类型: {chart_type}")

    except Exception as e:
        import traceback
        return _make_result(text=f"图表生成失败: {e}\n{traceback.format_exc()[-300:]}")


# ═══════════════════════════════════════════════════════════════════════════════
# Code Interpreter (subprocess sandbox)
# ═══════════════════════════════════════════════════════════════════════════════

def _build_sandboxed_wrapper(code, csv_path):
    """Generate a wrapper script that runs *code* inside a restricted exec().

    Uses ``.replace()`` on sentinel markers (``__CODE_JSON__`` /
    ``__CSV_JSON__``) so that the template can contain literal curly braces
    without f-string escaping issues.
    """
    import json
    code_json = json.dumps(code)
    csv_json = json.dumps(csv_path)

    template = r'''\
import sys, os, io, json as _json, base64, math, random, collections, itertools, statistics, re
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import scipy.stats as stats
import sklearn

# ── Data ──
df = pd.read_csv(__CSV_JSON__)

# ── User code ──
USER_CODE = __CODE_JSON__

# ── Optional sandbox layers ──
_SANDBOX_ACTIVE = False
_backend_dir = os.environ.get("_SANDBOX_BACKEND_DIR")
if _backend_dir and _backend_dir not in sys.path:
    sys.path.insert(0, _backend_dir)
try:
    from services._sandbox import validate_code_ast, make_safe_globals, CodeValidationError
    _SANDBOX_ACTIVE = True
except ImportError:
    pass

if _SANDBOX_ACTIVE:
    try:
        validate_code_ast(USER_CODE)
    except CodeValidationError as e:
        print(f"Sandbox blocked: {e}", file=sys.stderr)
        sys.exit(1)
    except SyntaxError as e:
        print(f"Syntax error: {e}", file=sys.stderr)
        sys.exit(1)

# ── Figure capture ──
_captured_figures = []
_figure_warnings = []
_MAX_CAPTURED_FIGURES = 5
_MAX_FIGURE_B64_CHARS = 8 * 1024 * 1024
_MAX_FIGURE_WIDTH = 6.2
_MAX_FIGURE_HEIGHT = 4.8

def _ascii_text(value, fallback):
    text = str(value or "").strip()
    return text if text and text.isascii() else fallback

def _sanitize_matplotlib_text(fig):
    for ax_index, ax in enumerate(fig.get_axes(), start=1):
        ax.set_title(_ascii_text(ax.get_title(), f"Chart {ax_index}"))
        if ax.get_xlabel():
            ax.set_xlabel(_ascii_text(ax.get_xlabel(), f"X Axis {ax_index}"))
        if ax.get_ylabel():
            ax.set_ylabel(_ascii_text(ax.get_ylabel(), f"Y Axis {ax_index}"))
        for i, tick in enumerate(ax.get_xticklabels(), start=1):
            if tick.get_text() and not tick.get_text().isascii():
                tick.set_text(f"Item {i}")
        for i, tick in enumerate(ax.get_yticklabels(), start=1):
            if tick.get_text() and not tick.get_text().isascii():
                tick.set_text(f"Item {i}")
        legend = ax.get_legend()
        if legend:
            if legend.get_title().get_text():
                legend.get_title().set_text(_ascii_text(legend.get_title().get_text(), "Group"))
            for i, text in enumerate(legend.get_texts(), start=1):
                text.set_text(_ascii_text(text.get_text(), f"Group {i}"))

def _capture_figure(fig=None):
    fig = fig or plt.gcf()
    if fig.get_axes():
        if len(_captured_figures) >= _MAX_CAPTURED_FIGURES:
            _figure_warnings.append(f"已达到图片数量上限 {_MAX_CAPTURED_FIGURES} 张，后续图片已跳过。")
            plt.close(fig)
            return
        width, height = fig.get_size_inches()
        if width > _MAX_FIGURE_WIDTH or height > _MAX_FIGURE_HEIGHT:
            scale = min(_MAX_FIGURE_WIDTH / max(width, 0.1), _MAX_FIGURE_HEIGHT / max(height, 0.1))
            fig.set_size_inches(max(3.6, width * scale), max(2.4, height * scale), forward=True)
        _sanitize_matplotlib_text(fig)
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=90, bbox_inches="tight")
        buf.seek(0)
        encoded = base64.b64encode(buf.read()).decode("utf-8")
        if len(encoded) > _MAX_FIGURE_B64_CHARS:
            _figure_warnings.append("有一张图片过大，已跳过。")
        else:
            _captured_figures.append(encoded)
    plt.close(fig)

_original_show = plt.show
def _patched_show(*args, **kwargs):
    _capture_figure()
plt.show = _patched_show
plt.savefig = lambda *a, **kw: _capture_figure()

# ── Execute ──
if _SANDBOX_ACTIVE:
    _safe_globals = make_safe_globals()
    _safe_globals.update({
        "df": df,
        "np": np, "pd": pd, "plt": plt,
        "stats": stats, "sklearn": sklearn,
        "math": math, "random": random,
        "collections": collections, "itertools": itertools,
        "statistics": statistics, "re": re,
        "datetime": datetime, "timedelta": timedelta,
        "_capture_figure": _capture_figure,
    })
    try:
        exec(USER_CODE, _safe_globals)
    except Exception:
        import traceback
        traceback.print_exc(file=sys.stderr)
else:
    try:
        exec(USER_CODE, {
            "df": df, "np": np, "pd": pd, "plt": plt,
            "stats": stats, "sklearn": sklearn,
            "math": math, "random": random,
            "collections": collections, "itertools": itertools,
            "statistics": statistics, "re": re,
            "datetime": datetime, "timedelta": timedelta,
        })
    except Exception:
        import traceback
        traceback.print_exc(file=sys.stderr)

# ── Output figures ──
if _captured_figures:
    print("__FIGURES__:" + _json.dumps(_captured_figures))
if _figure_warnings:
    print("__FIGURE_WARNINGS__:" + _json.dumps(list(dict.fromkeys(_figure_warnings))))

# ── Cleanup ──
try:
    os.remove(__CSV_JSON__)
except Exception:
    pass
'''
    return template.replace('__CODE_JSON__', code_json).replace('__CSV_JSON__', csv_json)


def _execute_code_in_subprocess(code, df, timeout=30):
    """Execute user code in a subprocess sandbox. Returns structured result."""
    import subprocess, sys, tempfile, os, io, json, base64, ast

    # ── Layer 1: AST pre-validation (fast-fail, no subprocess) ──
    try:
        from services._sandbox import validate_code_ast as _sandbox_validate
        from services._sandbox import CodeValidationError as _SandboxCodeError
    except ImportError:
        _sandbox_validate = None
        _SandboxCodeError = None

    if _sandbox_validate is not None and _SandboxCodeError is not None:
        try:
            _sandbox_validate(code)
        except _SandboxCodeError as e:
            return {"text": f"代码安全检查未通过: {e}", "images": []}
        except SyntaxError as e:
            return {"text": f"代码语法错误: {e}", "images": []}

    if len(df.columns) > _MAX_CODE_INTERPRETER_COLS:
        return {
            "text": (
                f"代码解释器最多支持 {_MAX_CODE_INTERPRETER_COLS} 列，"
                f"当前数据有 {len(df.columns)} 列。请先在数据处理页筛选列，或让 Agent 指定更少的列。"
            ),
            "images": [],
        }

    run_df = df
    sampled = False
    if len(run_df) > _MAX_CODE_INTERPRETER_ROWS:
        run_df = run_df.sample(n=_MAX_CODE_INTERPRETER_ROWS, random_state=42)
        sampled = True

    csv_text = run_df.to_csv(index=False)
    csv_bytes = csv_text.encode("utf-8")
    if len(csv_bytes) > _MAX_CODE_INTERPRETER_CSV_BYTES:
        return {
            "text": (
                "代码解释器输入数据过大，已停止执行。"
                f"当前采样后 CSV 约 {len(csv_bytes) / 1024 / 1024:.1f} MB，"
                f"上限为 {_MAX_CODE_INTERPRETER_CSV_BYTES / 1024 / 1024:.0f} MB。"
                "请先筛选列或减少数据量。"
            ),
            "images": [],
        }

    csv_path = os.path.join(tempfile.gettempdir(), f"_llm_code_df_{os.urandom(4).hex()}.csv")
    with open(csv_path, "wb") as f:
        f.write(csv_bytes)

    wrapper_code = _build_sandboxed_wrapper(code, csv_path)

    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False, encoding='utf-8') as f:
        f.write(wrapper_code)
        script_path = f.name

    # Pass backend directory so the subprocess can import services._sandbox
    backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    env = os.environ.copy()
    env['_SANDBOX_BACKEND_DIR'] = backend_dir

    try:
        proc = subprocess.run(
            [sys.executable, script_path],
            capture_output=True, text=True, timeout=timeout,
            env=env,
        )

        stdout, stderr = proc.stdout, proc.stderr

        figures = []
        figure_warnings = []
        clean_lines = []
        for line in stdout.split('\n'):
            if line.startswith('__FIGURES__:'):
                try:
                    figures = json.loads(line[len('__FIGURES__:'):])
                except json.JSONDecodeError:
                    pass
            elif line.startswith('__FIGURE_WARNINGS__:'):
                try:
                    figure_warnings.extend(json.loads(line[len('__FIGURE_WARNINGS__:'):]))
                except json.JSONDecodeError:
                    pass
            else:
                clean_lines.append(line)
        clean_stdout = '\n'.join(clean_lines).strip()
        clean_stderr = stderr.strip()

        images = []
        for fig in figures:
            if len(images) >= _MAX_CODE_INTERPRETER_IMAGES:
                figure_warnings.append(f"已达到图片数量上限 {_MAX_CODE_INTERPRETER_IMAGES} 张，后续图片已跳过。")
                break
            if not isinstance(fig, str) or len(fig) > _MAX_CODE_INTERPRETER_IMAGE_B64_CHARS:
                figure_warnings.append("有一张图片过大或格式异常，已跳过。")
                continue
            image_index = len(images) + 1
            images.append({"base64": fig, "title": f"Code Chart {image_index}", "alt": f"Figure {image_index}"})

        text_parts = []
        if sampled:
            text_parts.append(
                f"提示：原始数据有 {len(df)} 行，代码解释器本次使用固定随机种子抽样的 {len(run_df)} 行执行。"
            )
        if clean_stdout:
            text_parts.append(clean_stdout[:4000])
        if clean_stderr:
            text_parts.append(f"\n\n**stderr:**\n```\n{clean_stderr[:2000]}\n```")
        if figure_warnings:
            text_parts.append("\n\n".join(f"提示：{w}" for w in dict.fromkeys(figure_warnings)))
        if not text_parts and not images:
            text_parts.append("代码执行完毕，无输出。")

        return {"text": '\n'.join(text_parts), "images": images}

    except subprocess.TimeoutExpired:
        return {"text": "代码执行超时（超过 30 秒），已自动终止。请优化代码或缩小数据范围。", "images": []}
    finally:
        try:
            os.unlink(script_path)
        except OSError:
            pass
        try:
            os.unlink(csv_path)
        except OSError:
            pass


def _execute_tool(tool_name, args, df, session_id=""):
    """Execute a tool call and return {"text": str, "images": [{base64,title,alt}]}. All exceptions are caught."""
    import io
    import traceback
    import numpy as np
    import pandas as pd

    try:
        if tool_name == "get_data_overview":
            from services.data_service import build_data_summary
            return _make_result(text=build_data_summary(df, max_chars=6000))

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
            return _make_result(text=buf.getvalue())

        elif tool_name == "run_regression":
            target_col = args["target_column"]
            if target_col not in df.columns:
                return _make_result(text=f"错误: 列 '{target_col}' 不存在。可用列: {', '.join(str(c) for c in df.columns)}")
            if not pd.api.types.is_numeric_dtype(df[target_col]):
                return _make_result(text=f"错误: '{target_col}' 不是数值列，无法用于回归。请选择一个连续数值列作为目标。")
            if len(df) > _MAX_AGENT_TRAIN_ROWS:
                return _make_result(
                    text=(
                        f"Agent 自动训练最多支持 {_MAX_AGENT_TRAIN_ROWS} 行，当前有 {len(df)} 行。"
                        "请先筛选/采样数据，或到回归页面手动训练。"
                    )
                )

            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            feature_cols = [c for c in numeric_cols if c != target_col]
            if len(feature_cols) < 1:
                return _make_result(text=f"错误: 没有可用的特征列（除目标列 '{target_col}' 外无其他数值列）。")

            # Full pre-training validation (same as route layer)
            from services.training_validation import validate_regression_training
            if val_err := validate_regression_training(df, target_col, feature_cols, batch_size=32):
                return _make_result(text=f"训练前校验未通过: {val_err['error']}")

            from services.regression_service import train as regression_train
            import torch as _torch
            device_str = args.get("device", "auto")
            if device_str == "auto":
                device_str = "cuda" if _torch.cuda.is_available() else "cpu"

            epochs = min(max(args.get("epochs", 80), 20), _MAX_AGENT_TRAIN_EPOCHS)
            lr = args.get("learning_rate", 0.001)

            result, err = regression_train(
                df, target_col, feature_cols,
                hidden1=max(128, len(feature_cols)),
                hidden2=max(64, len(feature_cols) // 2),
                dropout_rate=0.2,
                learning_rate=lr, epochs=epochs, batch_size=32,
                device_str=device_str,
                dataset_name="", session_id=session_id
            )
            if err:
                return _make_result(text=f"回归训练失败: {err}")

            return _make_result(text=json.dumps({
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
            }, ensure_ascii=False))

        elif tool_name == "run_classification":
            target_col = args["target_column"]
            if target_col not in df.columns:
                return _make_result(text=f"错误: 列 '{target_col}' 不存在。可用列: {', '.join(str(c) for c in df.columns)}")
            if len(df) > _MAX_AGENT_TRAIN_ROWS:
                return _make_result(
                    text=(
                        f"Agent 自动训练最多支持 {_MAX_AGENT_TRAIN_ROWS} 行，当前有 {len(df)} 行。"
                        "请先筛选/采样数据，或到分类页面手动训练。"
                    )
                )

            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            feature_cols = [c for c in numeric_cols if c != target_col]
            if len(feature_cols) < 1:
                return _make_result(text=f"错误: 没有可用的数值特征列。")

            # Full pre-training validation (same as route layer)
            from services.training_validation import validate_classification_training
            if val_err := validate_classification_training(df, target_col, feature_cols, batch_size=32):
                return _make_result(text=f"训练前校验未通过: {val_err['error']}")

            from services.classification_service import train as classification_train
            import torch as _torch
            device_str = args.get("device", "auto")
            if device_str == "auto":
                device_str = "cuda" if _torch.cuda.is_available() else "cpu"

            epochs = min(max(args.get("epochs", 80), 20), _MAX_AGENT_TRAIN_EPOCHS)
            lr = args.get("learning_rate", 0.001)

            result, err = classification_train(
                df, target_col, feature_cols,
                hidden1=max(128, len(feature_cols)),
                hidden2=max(64, len(feature_cols) // 2),
                dropout_rate=0.2,
                learning_rate=lr, epochs=epochs, batch_size=32,
                device_str=device_str,
                dataset_name="", session_id=session_id
            )
            if err:
                return _make_result(text=f"分类训练失败: {err}")

            cm = result["cm"]
            return _make_result(text=json.dumps({
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
            }, ensure_ascii=False))

        elif tool_name == "run_clustering":
            algorithm = args.get("algorithm", "kmeans")
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if len(numeric_cols) < 2:
                return _make_result(text=f"错误: 需要至少 2 个数值列才能聚类，当前只有 {len(numeric_cols)} 个。")

            from services.clustering_service import train as clustering_train, elbow as clustering_elbow

            if algorithm == "kmeans":
                n_clusters = min(max(args.get("n_clusters", 3), 2), 10)
                params = {"n_clusters": n_clusters}

                try:
                    elbow_result = clustering_elbow(df, numeric_cols, min(10, len(df) // 10))
                except Exception:
                    elbow_result = None

                result, err = clustering_train(df, numeric_cols, "kmeans", params, "", session_id)
                if err:
                    return _make_result(text=f"聚类失败: {err}")

                return _make_result(text=json.dumps({
                    "algorithm": "kmeans",
                    "n_clusters": n_clusters,
                    "features": numeric_cols,
                    "cluster_sizes": result.get("cluster_counts", {}),
                    "silhouette_score": result.get("silhouette"),
                    "inertia": result.get("inertia"),
                    "elbow": {"ks": elbow_result["ks"], "inertias": elbow_result["inertias"]} if elbow_result else None,
                    "silhouette_guide": "轮廓系数范围 [-1, 1]，越接近 1 表示聚类质量越好"
                }, ensure_ascii=False))

            else:  # dbscan
                eps = args.get("eps", 0.5)
                min_samples = args.get("min_samples", 5)
                params = {"eps": eps, "min_samples": min_samples}
                result, err = clustering_train(df, numeric_cols, "dbscan", params, "", session_id)
                if err:
                    return _make_result(text=f"DBSCAN 聚类失败: {err}")

                return _make_result(text=json.dumps({
                    "algorithm": "dbscan",
                    "eps": eps, "min_samples": min_samples,
                    "features": numeric_cols,
                    "n_clusters_found": result.get("n_found", 0),
                    "cluster_sizes": result.get("cluster_counts", {}),
                    "silhouette_score": result.get("silhouette"),
                    "noise_points": result.get("cluster_counts", {}).get("-1", 0),
                    "silhouette_guide": "轮廓系数范围 [-1, 1]，越接近 1 越好。噪声点（簇 -1）已被排除在轮廓系数计算之外"
                }, ensure_ascii=False))

        elif tool_name == "run_correlation_analysis":
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if len(numeric_cols) < 2:
                return _make_result(text=f"错误: 需要至少 2 个数值列才能计算相关性，当前只有 {len(numeric_cols)} 个。")

            top_n = args.get("top_n", 10)
            corr = df[numeric_cols].corr()
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
            return _make_result(text=buf.getvalue())

        elif tool_name == "generate_chart":
            return _generate_chart(
                chart_type=args.get("chart_type", "scatter"),
                x_column=args.get("x_column", ""),
                y_column=args.get("y_column", ""),
                color_column=args.get("color_column", ""),
                title=args.get("title", ""),
                df=df
            )

        elif tool_name == "run_code_interpreter":
            code = args.get("code", "")
            if not code.strip():
                return _make_result(text="错误: 代码不能为空。")
            return _execute_code_in_subprocess(code, df, timeout=30)

        else:
            return _make_result(text=f"未知工具: {tool_name}")

    except Exception:
        return _make_result(text=f"工具执行出错: {traceback.format_exc()[-500:]}")


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

                result = _execute_tool(tool_name, tool_args, df, session_id)
                result_text = result.get("text", "")
                result_images = result.get("images", [])

                # Yield image events before tool_result for inline display
                for img in result_images:
                    yield {"status": "image", "base64": img["base64"], "title": img.get("title", ""), "alt": img.get("alt", "")}

                # Truncate very long text results
                if len(result_text) > _MAX_TOOL_RESULT_CHARS:
                    result_text = result_text[:_MAX_TOOL_RESULT_CHARS] + "\n... (结果已截断)"

                yield {"status": "tool_result", "tool": tool_name, "result": result_text}

                full_messages.append({
                    "role": "tool",
                    "tool_call_id": tc["id"],
                    "content": result_text
                })
                tool_results_context.append({"tool": tool_name, "summary": result_text[:200]})

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
