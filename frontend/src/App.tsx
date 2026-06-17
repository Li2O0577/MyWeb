import { ChangeEvent, useEffect, useRef, useState } from "react";
import {
  Activity,
  AlertTriangle,
  BarChart3,
  Brain,
  CheckCircle2,
  ChevronRight,
  Database,
  FileSpreadsheet,
  GitBranch,
  MessageSquareText,
  PanelLeftClose,
  PanelLeftOpen,
  Play,
  RefreshCcw,
  Settings2,
  Sparkles,
  Table2,
  Upload,
  Wand2
} from "lucide-react";
import {
  API_BASE,
  fetchDataProfile,
  fetchHealth,
  fetchMe,
  login,
  logout,
  register,
  uploadDataset
} from "./services/api";
import type { AuthUser, BackendStatus, ColumnProfile, DataProfile, HealthResponse, SavedModels, SessionMeta } from "./types";
import DataProcessingWorkspace from "./components/DataProcessingWorkspace";
import DataVisualization from "./components/DataVisualization";
import LlmWorkspace from "./components/LlmWorkspace";
import ModelWorkbench from "./components/ModelWorkbench";

const SESSION_STORAGE_KEY = "indeterminate.currentSessionId";
const SIDEBAR_STORAGE_KEY = "indeterminate.sidebarCollapsed";

type ViewId = "home" | "upload" | "preview" | "process" | "visualize" | "model" | "llm";

const viewIds = new Set<ViewId>(["home", "upload", "preview", "process", "visualize", "model", "llm"]);

const navItems = [
  { id: "home", label: "工作台", detail: "Overview", icon: Sparkles },
  { id: "upload", label: "数据加载", detail: "Upload", icon: Upload },
  { id: "preview", label: "数据预览", detail: "Preview", icon: Table2 },
  { id: "process", label: "数据处理", detail: "Prepare", icon: Settings2 },
  { id: "visualize", label: "数据可视化", detail: "Visualize", icon: BarChart3 },
  { id: "model", label: "模型训练", detail: "Train", icon: Brain },
  { id: "llm", label: "大模型分析", detail: "Agent", icon: MessageSquareText }
] satisfies Array<{ id: ViewId; label: string; detail: string; icon: typeof Sparkles }>;

const viewCopy: Record<ViewId, { eyebrow: string; title: string; description: string }> = {
  home: {
    eyebrow: "React data workspace",
    title: "工作台",
    description: "查看后端连接、当前数据、最近 session、模型状态和迁移进度。"
  },
  upload: {
    eyebrow: "Data entry",
    title: "数据加载",
    description: "上传 CSV 或 Excel，创建或恢复后端 session。"
  },
  preview: {
    eyebrow: "Data preview",
    title: "数据预览",
    description: "查看当前数据表、字段类型、缺失值和异常值概览。"
  },
  process: {
    eyebrow: "Data preparation",
    title: "数据处理",
    description: "迁移旧数据处理页：行列操作、类型转换、缺失值、缩放、编码、计算列和 PCA。"
  },
  visualize: {
    eyebrow: "Charts",
    title: "数据可视化",
    description: "生成散点、折线、柱状、面积、直方、箱线、小提琴、密度、饼图、成对关系、相关性和 3D 图表。"
  },
  model: {
    eyebrow: "Modeling",
    title: "模型训练",
    description: "训练决策树、聚类、回归、分类和自定义 MLP，并管理模型版本与预测。"
  },
  llm: {
    eyebrow: "LLM analysis",
    title: "大模型分析",
    description: "使用 Direct Chat 或 Agent Chat 进行流式数据分析、工具调用和图表生成。"
  }
};

function initialView(): ViewId {
  const raw = window.location.hash.replace("#", "") as ViewId;
  return viewIds.has(raw) ? raw : "home";
}

function viewNeedsData(view: ViewId) {
  return ["preview", "process", "visualize", "model", "llm"].includes(view);
}

function DataRequired({ onUpload }: { onUpload: () => void }) {
  return (
    <section className="empty-view" aria-label="需要数据">
      <Database size={24} aria-hidden="true" />
      <strong>这个界面需要先加载数据</strong>
      <span>上传文件或从工作台恢复最近 session 后即可使用。</span>
      <button className="button primary" type="button" onClick={onUpload}>
        <Upload size={15} aria-hidden="true" />
        去数据加载
      </button>
    </section>
  );
}

function AuthView({
  mode,
  username,
  password,
  error,
  loading,
  onModeChange,
  onUsernameChange,
  onPasswordChange,
  onSubmit
}: {
  mode: "login" | "register";
  username: string;
  password: string;
  error: string;
  loading: boolean;
  onModeChange: (mode: "login" | "register") => void;
  onUsernameChange: (value: string) => void;
  onPasswordChange: (value: string) => void;
  onSubmit: () => void;
}) {
  return (
    <main className="auth-shell">
      <section className="auth-panel" aria-label="账号登录">
        <div className="brand auth-brand">
          <div className="brand-mark" aria-hidden="true">I</div>
          <div className="brand-copy">
            <strong>Indeterminate</strong>
            <span>Data science workspace</span>
          </div>
        </div>
        <div className="auth-copy">
          <span>Account</span>
          <h1>{mode === "login" ? "登录工作台" : "创建本地账号"}</h1>
          <p>账号用于隔离数据 session、模型版本和训练任务，并为后续队列、配额和权限管理提供基础。</p>
        </div>
        <div className="auth-tabs" role="tablist" aria-label="账号模式">
          <button className={mode === "login" ? "active" : ""} type="button" onClick={() => onModeChange("login")}>登录</button>
          <button className={mode === "register" ? "active" : ""} type="button" onClick={() => onModeChange("register")}>注册</button>
        </div>
        <label className="form-field">
          <span>用户名</span>
          <input value={username} onChange={(event) => onUsernameChange(event.target.value)} autoComplete="username" />
        </label>
        <label className="form-field">
          <span>密码</span>
          <input
            value={password}
            onChange={(event) => onPasswordChange(event.target.value)}
            type="password"
            autoComplete={mode === "login" ? "current-password" : "new-password"}
            onKeyDown={(event) => {
              if (event.key === "Enter") onSubmit();
            }}
          />
        </label>
        {error ? <div className="inline-error">{error}</div> : null}
        <button className="button primary auth-submit" type="button" onClick={onSubmit} disabled={loading}>
          <Play size={15} aria-hidden="true" />
          {loading ? "处理中..." : mode === "login" ? "登录" : "创建账号"}
        </button>
      </section>
    </main>
  );
}

const workflowCards = [
  {
    label: "01 / Upload",
    title: "导入数据集",
    text: "CSV/Excel 上传已接入现有 Flask session，刷新后可恢复最近数据。"
  },
  {
    label: "02 / Explore",
    title: "检查字段和质量",
    text: "预览表、字段类型、缺失值和异常值概览已经迁移到 React。"
  },
  {
    label: "03 / Clean",
    title: "完整数据处理",
    text: "行列筛选、类型转换、缺失值处理、缩放、编码、计算列和 PCA 已迁移到独立页面。"
  },
  {
    label: "04 / Train",
    title: "训练和预测模型",
    text: "可视化、模型训练、版本列表、单条预测和批量预测已进入 React 工作台。"
  },
  {
    label: "05 / LLM",
    title: "大模型分析",
    text: "Direct Chat、Agent Chat、流式输出、工具调用和图表结果迁移到 React。"
  }
];

const modelLabels: Record<string, string> = {
  regression: "回归",
  classification: "分类",
  diy_mlp: "自定义 MLP",
  decision_tree: "决策树",
  clustering: "聚类"
};

function getBackendStatus(health: HealthResponse | null, error: string): BackendStatus {
  if (error) return "offline";
  if (!health) return "checking";
  if (health.status === "degraded") return "degraded";
  return "online";
}

function formatModels(health: HealthResponse | null) {
  if (!health?.saved_models) return "等待后端响应";
  const active = Object.entries(health.saved_models)
    .filter(([key, value]) => !key.endsWith("_versions") && value === true)
    .map(([key]) => modelLabels[key] ?? key);
  return active.length ? active.join(", ") : "暂无已激活模型";
}

function modelRows(savedModels: SavedModels | undefined) {
  return Object.entries(modelLabels).map(([key, label]) => {
    const active = savedModels?.[key] === true;
    const versions = Number(savedModels?.[`${key}_versions`] ?? 0);
    return { key, label, active, versions };
  });
}

function formatSession(session: SessionMeta) {
  const rows = typeof session.rows === "number" ? `${session.rows.toLocaleString()} 行` : "行数未知";
  const cols = typeof session.n_columns === "number" ? `${session.n_columns} 列` : "列数未知";
  return `${rows} · ${cols}`;
}

function countRiskColumns(profile: DataProfile | null) {
  if (!profile) return 0;
  return profile.column_profiles.filter((column) => column.missing_count > 0 || column.outlier_count > 0).length;
}

function totalOutliers(profile: DataProfile | null) {
  if (!profile) return 0;
  return profile.column_profiles.reduce((sum, column) => sum + column.outlier_count, 0);
}

function valueToText(value: unknown) {
  if (value === null || value === undefined) return "空";
  if (typeof value === "number") return Number.isFinite(value) ? String(value) : "非有限值";
  if (typeof value === "boolean") return value ? "true" : "false";
  if (typeof value === "object") return JSON.stringify(value);
  const text = String(value);
  return text.length > 80 ? `${text.slice(0, 77)}...` : text;
}

function kindLabel(kind: ColumnProfile["kind"]) {
  const labels = {
    numeric: "数值",
    categorical: "类别",
    datetime: "时间",
    boolean: "布尔"
  };
  return labels[kind] ?? kind;
}

function topMissingColumns(profile: DataProfile | null) {
  return [...(profile?.column_profiles ?? [])]
    .filter((column) => column.missing_count > 0)
    .sort((a, b) => b.missing_count - a.missing_count)
    .slice(0, 5);
}

function topOutlierColumns(profile: DataProfile | null) {
  return [...(profile?.column_profiles ?? [])]
    .filter((column) => column.outlier_count > 0)
    .sort((a, b) => b.outlier_count - a.outlier_count)
    .slice(0, 5);
}

function App() {
  const [user, setUser] = useState<AuthUser | null>(null);
  const [authLoading, setAuthLoading] = useState(true);
  const [authBusy, setAuthBusy] = useState(false);
  const [authMode, setAuthMode] = useState<"login" | "register">("login");
  const [authUsername, setAuthUsername] = useState("");
  const [authPassword, setAuthPassword] = useState("");
  const [authError, setAuthError] = useState("");
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [error, setError] = useState("");
  const [refreshToken, setRefreshToken] = useState(0);
  const [profile, setProfile] = useState<DataProfile | null>(null);
  const [uploadError, setUploadError] = useState("");
  const [uploading, setUploading] = useState(false);
  const [profileLoading, setProfileLoading] = useState(false);
  const [sidebarCollapsed, setSidebarCollapsed] = useState(() => localStorage.getItem(SIDEBAR_STORAGE_KEY) === "1");
  const [activeView, setActiveView] = useState<ViewId>(() => initialView());
  const fileInputRef = useRef<HTMLInputElement | null>(null);

  useEffect(() => {
    const controller = new AbortController();
    fetchMe(controller.signal)
      .then((nextUser) => setUser(nextUser))
      .catch(() => setUser(null))
      .finally(() => setAuthLoading(false));
    return () => controller.abort();
  }, []);

  useEffect(() => {
    const syncHashView = () => {
      const raw = window.location.hash.replace("#", "") as ViewId;
      setActiveView(viewIds.has(raw) ? raw : "home");
    };
    window.addEventListener("hashchange", syncHashView);
    return () => window.removeEventListener("hashchange", syncHashView);
  }, []);

  useEffect(() => {
    localStorage.setItem(SIDEBAR_STORAGE_KEY, sidebarCollapsed ? "1" : "0");
  }, [sidebarCollapsed]);

  useEffect(() => {
    if (!user) return;
    const controller = new AbortController();
    setError("");
    fetchHealth(controller.signal)
      .then((payload) => setHealth(payload))
      .catch((err: unknown) => {
        if (err instanceof DOMException && err.name === "AbortError") return;
        setHealth(null);
        setError(err instanceof Error ? err.message : "无法连接后端");
      });
    return () => controller.abort();
  }, [refreshToken, user]);

  useEffect(() => {
    const savedSessionId = localStorage.getItem(SESSION_STORAGE_KEY);
    if (!user || !savedSessionId || profile) return;

    const controller = new AbortController();
    setProfileLoading(true);
    fetchDataProfile(savedSessionId, controller.signal)
      .then((payload) => {
        setProfile(payload);
      })
      .catch(() => {
        localStorage.removeItem(SESSION_STORAGE_KEY);
      })
      .finally(() => setProfileLoading(false));
    return () => controller.abort();
  }, [profile, user]);

  const submitAuth = async () => {
    setAuthBusy(true);
    setAuthError("");
    try {
      const nextUser = authMode === "login"
        ? await login(authUsername, authPassword)
        : await register(authUsername, authPassword);
      setUser(nextUser);
      setAuthPassword("");
      setRefreshToken((value) => value + 1);
    } catch (err) {
      setAuthError(err instanceof Error ? err.message : "账号操作失败");
    } finally {
      setAuthBusy(false);
      setAuthLoading(false);
    }
  };

  const handleLogout = async () => {
    await logout();
    setUser(null);
    setHealth(null);
    setProfile(null);
    localStorage.removeItem(SESSION_STORAGE_KEY);
  };

  if (authLoading) {
    return <main className="auth-shell"><section className="auth-panel"><div className="empty-list">正在检查登录状态...</div></section></main>;
  }

  if (!user) {
    return (
      <AuthView
        mode={authMode}
        username={authUsername}
        password={authPassword}
        error={authError}
        loading={authBusy}
        onModeChange={setAuthMode}
        onUsernameChange={setAuthUsername}
        onPasswordChange={setAuthPassword}
        onSubmit={submitAuth}
      />
    );
  }

  const backendStatus = getBackendStatus(health, error);
  const statusText =
    backendStatus === "online" ? "Flask API 已连接" :
    backendStatus === "degraded" ? "Flask API 降级响应" :
    backendStatus === "offline" ? "后端未连接" :
    "正在检查后端";

  const recentSession = health?.recent_sessions?.[0];
  const riskColumns = countRiskColumns(profile);
  const outlierTotal = totalOutliers(profile);
  const previewRows = profile?.preview ?? [];
  const previewDisplayRows = previewRows;
  const previewColumns = profile?.columns.slice(0, 12) ?? [];
  const activeCopy = viewCopy[activeView];

  const navigateView = (view: ViewId) => {
    setActiveView(view);
    window.history.replaceState(null, "", `#${view}`);
    window.scrollTo({ top: 0, behavior: "smooth" });
  };

  const handleFileChange = async (event: ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;
    setUploading(true);
    setUploadError("");
    try {
      const payload = await uploadDataset(file);
      setProfile(payload);
      localStorage.setItem(SESSION_STORAGE_KEY, payload.session_id);
      setRefreshToken((value) => value + 1);
    } catch (err) {
      setUploadError(err instanceof Error ? err.message : "上传失败");
    } finally {
      setUploading(false);
      event.target.value = "";
    }
  };

  const restoreSession = async (sessionId?: string) => {
    if (!sessionId) return;
    setProfileLoading(true);
    setUploadError("");
    try {
      const payload = await fetchDataProfile(sessionId);
      setProfile(payload);
      localStorage.setItem(SESSION_STORAGE_KEY, payload.session_id);
      setRefreshToken((value) => value + 1);
    } catch (err) {
      setUploadError(err instanceof Error ? err.message : "恢复 session 失败");
    } finally {
      setProfileLoading(false);
    }
  };

  const handleProfileChange = (payload: DataProfile) => {
    setProfile(payload);
    localStorage.setItem(SESSION_STORAGE_KEY, payload.session_id);
    setRefreshToken((value) => value + 1);
  };

  return (
    <div className={sidebarCollapsed ? "app-shell sidebar-collapsed" : "app-shell"}>
      <aside className="sidebar" aria-label="主导航">
        <div className="brand">
          <div className="brand-mark" aria-hidden="true">
            I
          </div>
          <div className="brand-copy">
            <strong>Indeterminate</strong>
            <span>Data science workspace</span>
          </div>
          <button
            className="icon-button sidebar-toggle"
            type="button"
            aria-label={sidebarCollapsed ? "展开侧边栏" : "折叠侧边栏"}
            title={sidebarCollapsed ? "展开侧边栏" : "折叠侧边栏"}
            onClick={() => setSidebarCollapsed((value) => !value)}
          >
            {sidebarCollapsed ? <PanelLeftOpen size={16} /> : <PanelLeftClose size={16} />}
          </button>
        </div>

        <nav className="nav-list" aria-label="功能模块">
          <div className="nav-label">Workspace</div>
          {navItems.map((item) => {
            const Icon = item.icon;
            return (
              <button
                className={item.id === activeView ? "nav-item active" : "nav-item"}
                key={item.label}
                title={item.label}
                type="button"
                aria-current={item.id === activeView ? "page" : undefined}
                onClick={() => navigateView(item.id)}
              >
                <span className="nav-main">
                  <Icon size={16} aria-hidden="true" />
                  <span className="nav-text">{item.label}</span>
                </span>
                <small>{item.detail}</small>
              </button>
            );
          })}
        </nav>

        <div className="sidebar-note">
          <strong>Phase 7</strong>
          <span>数据处理和可视化页已迁移为独立 React 工作台，旧功能正在按页面补齐。</span>
        </div>
      </aside>

      <main className="main">
        <header className="topbar">
          <div className="connection">
            <span className={`status-dot ${backendStatus}`} aria-hidden="true" />
            <span>{statusText}</span>
            <code>{API_BASE}</code>
          </div>
          <div className="topbar-actions">
            <div className="user-chip" title={user.user_id}>{user.username}</div>
            <button className="button ghost" type="button" onClick={() => setRefreshToken((value) => value + 1)}>
              <RefreshCcw size={15} aria-hidden="true" />
              刷新状态
            </button>
            <button className="button primary" type="button" onClick={() => fileInputRef.current?.click()}>
              <Play size={15} aria-hidden="true" />
              上传数据
            </button>
            <button className="button ghost" type="button" onClick={handleLogout}>
              退出
            </button>
          </div>
        </header>

        <section
          className={["process", "visualize", "model", "llm"].includes(activeView) ? "workspace workspace-wide" : "workspace"}
          aria-labelledby="workspace-title"
        >
          <div className="intro">
            <div className="eyebrow">
              <Wand2 size={15} aria-hidden="true" />
              {activeCopy.eyebrow}
            </div>
            <h1 id="workspace-title">{activeCopy.title}</h1>
            <p>{activeCopy.description}</p>
          </div>

          {(activeView === "home" || activeView === "upload") ? (
          <section className="command-center" aria-label="工作台入口">
            <div className="upload-panel">
              <div className="upload-copy">
                <div className="panel-kicker">Data entry</div>
                <h2>上传 CSV 或 Excel，创建新的分析 session。</h2>
                <p>成功后会显示字段、预览、缺失值、异常值和最近 session 状态。session id 会保存在浏览器中用于刷新恢复。</p>
              </div>
              <input
                ref={fileInputRef}
                className="file-input"
                type="file"
                accept=".csv,.xlsx,.xls"
                onChange={handleFileChange}
              />
              <button
                className="upload-drop"
                type="button"
                onClick={() => fileInputRef.current?.click()}
                disabled={uploading || backendStatus === "offline"}
              >
                <FileSpreadsheet size={22} aria-hidden="true" />
                <strong>{uploading ? "正在上传..." : "选择数据文件"}</strong>
                <span>支持 .csv、.xlsx、.xls，最大限制沿用后端 256 MB。</span>
              </button>
              {uploadError ? <div className="inline-error">{uploadError}</div> : null}
              {backendStatus === "offline" ? (
                <div className="inline-warning">启动 Flask 后端后即可上传：cd backend && python app.py</div>
              ) : null}
            </div>

            <div className="snapshot-panel">
              <div className="snapshot-header">
                <span>Current snapshot</span>
                {profile ? <CheckCircle2 size={17} aria-hidden="true" /> : <Database size={17} aria-hidden="true" />}
              </div>
              {profile ? (
                <div className="uploaded-summary">
                  <strong>{profile.session_meta?.source_name ?? "已上传数据"}</strong>
                  <div className="snapshot-metrics">
                    <span>{profile.n_rows.toLocaleString()} 行</span>
                    <span>{profile.n_cols} 列</span>
                    <span>{profile.numeric_cols.length} 数值列</span>
                    <span>{riskColumns} 风险列</span>
                  </div>
                  <div className="column-strip">
                    {profile.columns.slice(0, 8).map((column) => (
                      <span key={column}>{column}</span>
                    ))}
                    {profile.columns.length > 8 ? <span>+{profile.columns.length - 8}</span> : null}
                  </div>
                </div>
              ) : (
                <div className="empty-state">
                  <strong>{profileLoading ? "正在恢复 session" : "等待数据"}</strong>
                  <span>{profileLoading ? "正在读取上次上传的数据概览。" : "上传后这里会显示数据集概览和字段摘要。"}</span>
                </div>
              )}
            </div>
          </section>
          ) : null}

          {activeView === "home" ? (
          <section className="status-grid" aria-label="项目状态">
            <div className="status-card">
              <span>后端状态</span>
              <strong>{statusText}</strong>
              <small>{error || "健康检查来自 /api/health"}</small>
            </div>
            <div className="status-card">
              <span>活跃会话</span>
              <strong>{health?.active_sessions ?? "-"}</strong>
              <small>{recentSession?.source_name ? `最近数据：${recentSession.source_name}` : "暂无最近数据"}</small>
            </div>
            <div className="status-card">
              <span>数据质量</span>
              <strong>{profile ? `${profile.missing_total} 缺失 / ${outlierTotal} 异常` : "等待数据"}</strong>
              <small>{profile ? `${riskColumns} 个字段需要关注` : "上传后自动计算"}</small>
            </div>
            <div className="status-card">
              <span>模型状态</span>
              <strong>{formatModels(health)}</strong>
              <small>已复用现有模型注册表返回值</small>
            </div>
          </section>
          ) : null}

          {viewNeedsData(activeView) && !profile ? <DataRequired onUpload={() => navigateView("upload")} /> : null}

          {profile && activeView === "preview" ? (
            <>
              <section className="data-grid single-view" aria-label="数据预览">
                <article className="data-panel preview-panel">
                  <div className="panel-title split">
                    <span>
                      <Table2 size={17} aria-hidden="true" />
                      <h2>数据预览</h2>
                    </span>
                    <small>
                      已显示 {previewDisplayRows.length} 行预览 / 全量 {profile.n_rows.toLocaleString()} 行 · 显示 {previewColumns.length} / {profile.n_cols} 列
                    </small>
                  </div>
                  <div className="table-wrap">
                    <table className="data-table">
                      <thead>
                        <tr>
                          {previewColumns.map((column) => (
                            <th key={column}>{column}</th>
                          ))}
                        </tr>
                      </thead>
                      <tbody>
                        {previewDisplayRows.map((row, rowIndex) => (
                          <tr key={rowIndex}>
                            {previewColumns.map((column) => (
                              <td key={column}>{valueToText(row[column])}</td>
                            ))}
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </article>
              </section>

              <section className="quality-grid" aria-label="字段类型和数据质量">
                <article className="data-panel">
                  <div className="panel-title">
                    <GitBranch size={17} aria-hidden="true" />
                    <h2>字段类型</h2>
                  </div>
                  <div className="field-list">
                    {profile.column_profiles.slice(0, 12).map((column) => (
                      <div className="field-row" key={column.name}>
                        <div>
                          <strong>{column.name}</strong>
                          <span>{column.dtype} · {column.unique_count} 唯一值</span>
                        </div>
                        <span className={`kind-badge ${column.kind}`}>{kindLabel(column.kind)}</span>
                      </div>
                    ))}
                  </div>
                </article>

                <article className="data-panel">
                  <div className="panel-title">
                    <AlertTriangle size={17} aria-hidden="true" />
                    <h2>缺失值 / 异常值</h2>
                  </div>
                  <div className="quality-columns">
                    <div>
                      <h3>缺失值</h3>
                      {topMissingColumns(profile).length ? (
                        topMissingColumns(profile).map((column) => (
                          <div className="quality-row" key={column.name}>
                            <span>{column.name}</span>
                            <strong>{column.missing_count} · {column.missing_pct}%</strong>
                          </div>
                        ))
                      ) : (
                        <div className="empty-list">未发现缺失值。</div>
                      )}
                    </div>
                    <div>
                      <h3>异常值</h3>
                      {topOutlierColumns(profile).length ? (
                        topOutlierColumns(profile).map((column) => (
                          <div className="quality-row" key={column.name}>
                            <span>{column.name}</span>
                            <strong>{column.outlier_count}</strong>
                          </div>
                        ))
                      ) : (
                        <div className="empty-list">未发现明显异常值。</div>
                      )}
                    </div>
                  </div>
                </article>
              </section>
            </>
          ) : null}

          {profile && activeView === "process" ? <DataProcessingWorkspace profile={profile} onProfileChange={handleProfileChange} /> : null}

          {profile && activeView === "visualize" ? <DataVisualization profile={profile} /> : null}
          {profile && activeView === "model" ? <ModelWorkbench profile={profile} /> : null}
          {profile && activeView === "llm" ? <LlmWorkspace profile={profile} /> : null}

          {activeView === "home" ? (
          <>
          <section className="overview-grid" aria-label="最近状态和模型概览">
            <article className="overview-panel">
              <div className="panel-title">
                <Database size={17} aria-hidden="true" />
                <h2>最近 session</h2>
              </div>
              <div className="session-list">
                {health?.recent_sessions?.length ? (
                  health.recent_sessions.slice(0, 4).map((session) => (
                    <button
                      className="session-row"
                      key={session.session_id ?? session.updated_at_iso}
                      type="button"
                      disabled={!session.session_id || profileLoading}
                      onClick={() => restoreSession(session.session_id)}
                    >
                      <div>
                        <strong>{session.source_name ?? "unknown"}</strong>
                        <span>{formatSession(session)}</span>
                      </div>
                      <code>{session.session_id ?? "-"}</code>
                    </button>
                  ))
                ) : (
                  <div className="empty-list">暂无后端 session。上传数据后会出现在这里。</div>
                )}
              </div>
            </article>

            <article className="overview-panel">
              <div className="panel-title">
                <Brain size={17} aria-hidden="true" />
                <h2>模型概览</h2>
              </div>
              <div className="model-list">
                {modelRows(health?.saved_models).map((model) => (
                  <div className="model-row" key={model.key}>
                    <div>
                      <strong>{model.label}</strong>
                      <span>{model.versions} 个版本</span>
                    </div>
                    <span className={model.active ? "model-badge active" : "model-badge"}>
                      {model.active ? "Active" : "Empty"}
                    </span>
                  </div>
                ))}
              </div>
            </article>
          </section>

          <section className="cards" aria-label="迁移流程">
            {workflowCards.map((card) => (
              <article className="card" key={card.label}>
                <span>{card.label}</span>
                <h2>{card.title}</h2>
                <p>{card.text}</p>
                <ChevronRight size={16} aria-hidden="true" />
              </article>
            ))}
          </section>

          <section className="baseline-panel" aria-label="基线检查">
            <div>
              <Activity size={18} aria-hidden="true" />
              <strong>重构基线</strong>
            </div>
            <p>
              旧前端、后端 API 和模型服务暂未删除。React 现在拥有第一条真实数据链路：
              上传、预览、字段质量、session 恢复、完整数据处理、可视化、模型训练预测和 LLM 分析。
            </p>
          </section>
          </>
          ) : null}
        </section>
      </main>
    </div>
  );
}

export default App;
