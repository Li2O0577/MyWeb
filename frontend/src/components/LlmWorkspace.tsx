import { FormEvent, useMemo, useRef, useState } from "react";
import {
  Bot,
  BrainCircuit,
  Image as ImageIcon,
  KeyRound,
  MessageSquareText,
  PlugZap,
  Send,
  Settings2,
  Sparkles,
  Trash2,
  Wrench
} from "lucide-react";
import { streamLlmChat } from "../services/api";
import type { DataProfile, LlmConfig, LlmImage, LlmMode, LlmPayloadMessage, LlmStreamEvent } from "../types";

interface Props {
  profile: DataProfile | null;
}

interface ToolTrace {
  name: string;
  args?: Record<string, unknown>;
  result?: string;
  status: "running" | "done";
  images: LlmImage[];
}

interface ChatMessage {
  id: string;
  role: "user" | "assistant";
  mode: LlmMode;
  content: string;
  status?: string;
  images: LlmImage[];
  tools: ToolTrace[];
}

const CONFIG_STORAGE_KEY = "indeterminate.llmConfig";
const KEY_STORAGE_KEY = "indeterminate.llmApiKey";

const DEFAULT_CONFIG: LlmConfig = {
  api_base: "https://api.openai.com/v1",
  api_key: "",
  model: "gpt-4o"
};

const TOOL_LABELS: Record<string, string> = {
  get_data_overview: "数据概览",
  get_column_details: "列详情",
  run_regression: "回归分析",
  run_classification: "分类分析",
  run_clustering: "聚类分析",
  run_correlation_analysis: "相关性分析",
  generate_chart: "图表生成",
  run_code_interpreter: "代码执行"
};

function loadConfig(): LlmConfig {
  try {
    const stored = JSON.parse(localStorage.getItem(CONFIG_STORAGE_KEY) || "{}") as Partial<LlmConfig>;
    return {
      api_base: stored.api_base || DEFAULT_CONFIG.api_base,
      api_key: sessionStorage.getItem(KEY_STORAGE_KEY) || "",
      model: stored.model || DEFAULT_CONFIG.model
    };
  } catch {
    return DEFAULT_CONFIG;
  }
}

function saveConfig(config: LlmConfig) {
  localStorage.setItem(CONFIG_STORAGE_KEY, JSON.stringify({ api_base: config.api_base, model: config.model }));
  if (config.api_key) {
    sessionStorage.setItem(KEY_STORAGE_KEY, config.api_key);
  } else {
    sessionStorage.removeItem(KEY_STORAGE_KEY);
  }
}

function uid(prefix: string) {
  return `${prefix}_${Date.now()}_${Math.random().toString(16).slice(2)}`;
}

function imageSrc(image: LlmImage) {
  return `data:image/png;base64,${image.base64}`;
}

function shortJson(value: unknown) {
  const text = JSON.stringify(value ?? {}, null, 2);
  return text.length > 420 ? `${text.slice(0, 417)}...` : text;
}

function profileContext(profile: DataProfile | null) {
  if (!profile) return "";
  const columns = profile.column_profiles
    .slice(0, 16)
    .map((column) => `${column.name}: ${column.kind}, missing=${column.missing_count}, outliers=${column.outlier_count}`)
    .join("\n");
  return [
    "Data context:",
    `Dataset: ${profile.session_meta?.source_name || "current dataset"}`,
    `Rows: ${profile.n_rows}, Columns: ${profile.n_cols}`,
    `Numeric columns: ${profile.numeric_cols.join(", ") || "none"}`,
    `Categorical columns: ${profile.categorical_cols.join(", ") || "none"}`,
    `Missing cells: ${profile.missing_total}`,
    "Column profile:",
    columns,
    profile.summary ? `Backend summary:\n${profile.summary}` : ""
  ].filter(Boolean).join("\n");
}

function eventErrorMessage(event: LlmStreamEvent) {
  const error = event.error;
  if (!error) return "";
  const code = error.code || "ERROR";
  const message = error.detail || error.message || "请求没有完成。";
  return `${code}: ${message}`;
}

export default function LlmWorkspace({ profile }: Props) {
  const [mode, setMode] = useState<LlmMode>("direct");
  const [config, setConfig] = useState<LlmConfig>(() => loadConfig());
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState("");
  const [attachContext, setAttachContext] = useState(true);
  const [isStreaming, setIsStreaming] = useState(false);
  const [error, setError] = useState("");
  const abortRef = useRef<AbortController | null>(null);

  const canUseAgent = Boolean(profile?.session_id);
  const currentMessages = useMemo(() => messages.filter((message) => message.mode === mode), [messages, mode]);
  const contextText = useMemo(() => profileContext(profile), [profile]);

  const updateAssistant = (id: string, updater: (message: ChatMessage) => ChatMessage) => {
    setMessages((current) => current.map((message) => (message.id === id ? updater(message) : message)));
  };

  const handleEvent = (assistantId: string, event: LlmStreamEvent) => {
    if (event.error) {
      throw new Error(eventErrorMessage(event));
    }
    if (event.status === "thinking") {
      updateAssistant(assistantId, (message) => ({ ...message, status: event.message || "正在分析数据..." }));
      return;
    }
    if (event.status === "tool_call" && event.tool) {
      updateAssistant(assistantId, (message) => ({
        ...message,
        status: `正在执行：${TOOL_LABELS[event.tool || ""] || event.tool}`,
        tools: [
          ...message.tools,
          { name: event.tool || "tool", args: event.args, status: "running", images: [] }
        ]
      }));
      return;
    }
    if (event.status === "image" && event.base64) {
      const image = { base64: event.base64, title: event.title, alt: event.alt };
      updateAssistant(assistantId, (message) => {
        const tools = [...message.tools];
        const runningIndex = [...tools].reverse().findIndex((tool) => tool.status === "running");
        if (runningIndex >= 0) {
          const index = tools.length - 1 - runningIndex;
          tools[index] = { ...tools[index], images: [...tools[index].images, image] };
        }
        return { ...message, images: [...message.images, image], tools };
      });
      return;
    }
    if (event.status === "tool_result" && event.tool) {
      updateAssistant(assistantId, (message) => ({
        ...message,
        status: "工具执行完成，正在生成回答...",
        tools: message.tools.map((tool) =>
          tool.status === "running" && tool.name === event.tool
            ? { ...tool, status: "done", result: event.result || "" }
            : tool
        )
      }));
      return;
    }
    if (event.chunk) {
      updateAssistant(assistantId, (message) => ({
        ...message,
        status: "正在流式输出...",
        content: `${message.content}${event.chunk}`
      }));
      return;
    }
    if (event.done) {
      updateAssistant(assistantId, (message) => ({ ...message, status: "完成" }));
    }
  };

  const buildPayloadMessages = (userText: string): LlmPayloadMessage[] => {
    const history: LlmPayloadMessage[] = currentMessages
      .filter((message) => message.content.trim())
      .slice(-8)
      .map((message) => ({
        role: message.role,
        content: message.content
      }));
    const content =
      mode === "direct" && attachContext && contextText
        ? `${userText}\n\n---\n\n${contextText}`
        : userText;
    return [...history, { role: "user", content }];
  };

  const handleSubmit = async (event: FormEvent) => {
    event.preventDefault();
    const userText = input.trim();
    if (!userText || isStreaming) return;
    if (!config.model.trim()) {
      setError("请输入模型名称。");
      return;
    }
    if (mode === "agent" && !canUseAgent) {
      setError("Agent Chat 需要先上传或恢复一个后端 session。");
      return;
    }

    saveConfig(config);
    setError("");
    setInput("");
    setIsStreaming(true);

    const userMessage: ChatMessage = {
      id: uid("user"),
      role: "user",
      mode,
      content: userText,
      images: [],
      tools: []
    };
    const assistantId = uid("assistant");
    const assistantMessage: ChatMessage = {
      id: assistantId,
      role: "assistant",
      mode,
      content: "",
      status: mode === "agent" ? "正在启动 Agent..." : "正在连接模型...",
      images: [],
      tools: []
    };
    setMessages((current) => [...current, userMessage, assistantMessage]);

    const controller = new AbortController();
    abortRef.current = controller;
    try {
      const payload =
        mode === "agent"
          ? {
              session_id: profile?.session_id,
              api_base: config.api_base,
              api_key: config.api_key,
              model: config.model,
              messages: buildPayloadMessages(userText)
            }
          : {
              api_base: config.api_base,
              api_key: config.api_key,
              model: config.model,
              messages: buildPayloadMessages(userText)
            };
      await streamLlmChat(mode, payload, (streamEvent) => handleEvent(assistantId, streamEvent), controller.signal);
      updateAssistant(assistantId, (message) => ({ ...message, status: message.content || message.images.length ? "完成" : "空响应" }));
    } catch (err) {
      if (err instanceof DOMException && err.name === "AbortError") {
        updateAssistant(assistantId, (message) => ({ ...message, status: "已停止" }));
      } else {
        const message = err instanceof Error ? err.message : "LLM 请求失败";
        setError(message);
        updateAssistant(assistantId, (current) => ({ ...current, status: "失败", content: current.content || message }));
      }
    } finally {
      setIsStreaming(false);
      abortRef.current = null;
    }
  };

  const stopStreaming = () => {
    abortRef.current?.abort();
  };

  return (
    <section className="llm-section" aria-label="LLM 分析">
      <div className="section-heading">
        <MessageSquareText size={18} aria-hidden="true" />
        <div>
          <h2>LLM 分析</h2>
          <p>Direct Chat、Agent Chat、流式输出、工具调用和图表结果统一迁移到 React。</p>
        </div>
      </div>

      <div className="llm-shell">
        <aside className="llm-config" aria-label="LLM 配置">
          <div className="panel-title">
            <Settings2 size={17} aria-hidden="true" />
            <h3>连接配置</h3>
          </div>
          <label className="form-field">
            <span>API Base</span>
            <input
              value={config.api_base}
              onChange={(event) => setConfig({ ...config, api_base: event.target.value })}
              placeholder="https://api.openai.com/v1"
            />
          </label>
          <label className="form-field">
            <span>Model</span>
            <input value={config.model} onChange={(event) => setConfig({ ...config, model: event.target.value })} placeholder="gpt-4o" />
          </label>
          <label className="form-field">
            <span>API Key</span>
            <input
              type="password"
              value={config.api_key}
              onChange={(event) => setConfig({ ...config, api_key: event.target.value })}
              placeholder="留空则使用后端 LLM_API_KEY"
            />
          </label>
          <button className="button ghost full" type="button" onClick={() => saveConfig(config)}>
            <KeyRound size={15} aria-hidden="true" />
            保存配置
          </button>

          <div className="llm-status-card">
            <PlugZap size={16} aria-hidden="true" />
            <div>
              <strong>{profile ? "数据上下文可用" : "等待数据"}</strong>
              <span>{profile ? `${profile.n_rows.toLocaleString()} 行 · session ${profile.session_id}` : "上传或恢复 session 后 Agent 可用。"}</span>
            </div>
          </div>
        </aside>

        <div className="llm-chat-panel">
          <div className="llm-tabs" role="tablist" aria-label="LLM 模式">
            <button className={mode === "direct" ? "llm-tab active" : "llm-tab"} type="button" onClick={() => setMode("direct")}>
              <Sparkles size={15} aria-hidden="true" />
              Direct Chat
            </button>
            <button className={mode === "agent" ? "llm-tab active" : "llm-tab"} type="button" onClick={() => setMode("agent")}>
              <BrainCircuit size={15} aria-hidden="true" />
              Agent Chat
            </button>
          </div>

          <div className="llm-mode-note">
            {mode === "direct"
              ? "Direct Chat 直接与模型对话，可选择附带当前数据摘要。"
              : "Agent Chat 会把后端 session 交给 Agent，由模型自主调用数据概览、图表、聚类、回归等工具。"}
          </div>

          <div className="chat-log" aria-live="polite">
            {currentMessages.length ? (
              currentMessages.map((message) => (
                <article className={`chat-message ${message.role}`} key={message.id}>
                  <div className="chat-avatar" aria-hidden="true">
                    {message.role === "assistant" ? <Bot size={16} /> : "U"}
                  </div>
                  <div className="chat-bubble">
                    <div className="chat-meta">
                      <strong>{message.role === "assistant" ? "Assistant" : "You"}</strong>
                      {message.status ? <span>{message.status}</span> : null}
                    </div>
                    {message.content ? <p className="chat-text">{message.content}</p> : <div className="streaming-line">等待响应...</div>}
                    {message.images.length ? (
                      <div className="image-grid">
                        {message.images.map((image, index) => (
                          <figure className="llm-image" key={`${image.title}-${index}`}>
                            <img src={imageSrc(image)} alt={image.alt || image.title || "LLM chart"} />
                            <figcaption>
                              <ImageIcon size={14} aria-hidden="true" />
                              {image.title || "图表结果"}
                            </figcaption>
                          </figure>
                        ))}
                      </div>
                    ) : null}
                    {message.tools.length ? (
                      <div className="tool-trace-list">
                        {message.tools.map((tool, index) => (
                          <details className="tool-trace" key={`${tool.name}-${index}`}>
                            <summary>
                              <Wrench size={14} aria-hidden="true" />
                              <span>{TOOL_LABELS[tool.name] || tool.name}</span>
                              <small>{tool.status === "done" ? "完成" : "运行中"}</small>
                            </summary>
                            <pre>{shortJson(tool.args)}</pre>
                            {tool.result ? <p>{tool.result}</p> : null}
                          </details>
                        ))}
                      </div>
                    ) : null}
                  </div>
                </article>
              ))
            ) : (
              <div className="llm-empty">
                <Bot size={22} aria-hidden="true" />
                <strong>{mode === "agent" ? "让 Agent 分析当前数据" : "开始一次 Direct Chat"}</strong>
                <span>{mode === "agent" ? "例如：画图分析哪些因素影响 sale_price。" : "例如：根据当前数据摘要给我三个建模建议。"}</span>
              </div>
            )}
          </div>

          {error ? <div className="inline-error">{error}</div> : null}

          <form className="llm-composer" onSubmit={handleSubmit}>
            <textarea
              value={input}
              onChange={(event) => setInput(event.target.value)}
              placeholder={mode === "agent" ? "告诉 Agent 要做什么分析..." : "输入你的问题..."}
              rows={3}
            />
            <div className="composer-row">
              <label className="toggle-row">
                <input
                  type="checkbox"
                  checked={attachContext}
                  disabled={mode === "agent" || !profile}
                  onChange={(event) => setAttachContext(event.target.checked)}
                />
                <span>附带数据摘要</span>
              </label>
              <div className="composer-actions">
                <button className="button ghost" type="button" onClick={() => setMessages([])} disabled={isStreaming || !messages.length}>
                  <Trash2 size={15} aria-hidden="true" />
                  清空
                </button>
                {isStreaming ? (
                  <button className="button ghost" type="button" onClick={stopStreaming}>
                    停止
                  </button>
                ) : null}
                <button className="button primary" type="submit" disabled={isStreaming || !input.trim() || (mode === "agent" && !canUseAgent)}>
                  <Send size={15} aria-hidden="true" />
                  发送
                </button>
              </div>
            </div>
          </form>
        </div>
      </div>
    </section>
  );
}
