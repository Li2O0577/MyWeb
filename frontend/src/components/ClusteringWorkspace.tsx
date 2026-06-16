import { ChangeEvent, useEffect, useMemo, useState } from "react";
import {
  AlertTriangle,
  BarChart3,
  Brain,
  CircleDot,
  Download,
  GitPullRequest,
  Play,
  RefreshCcw,
  Send,
  Trash2,
  Upload
} from "lucide-react";
import {
  activateModelVersion,
  batchPredictModel,
  clearModel,
  deleteModelVersion,
  fetchClusteringElbow,
  fetchModelStatus,
  fetchModelVersionDetail,
  fetchModelVersions,
  predictModel,
  trainModel
} from "../services/api";
import type { ApiJson, DataProfile, ModelVersion } from "../types";

interface Props {
  profile: DataProfile;
}

type Algorithm = "kmeans" | "dbscan";
type ClusterTab = "train" | "result" | "predict";

interface PcaPayload {
  x?: number[];
  y?: number[];
  ev1?: number;
  ev2?: number;
}

interface ClusterResult extends ApiJson {
  version_id?: string;
  algorithm?: Algorithm;
  labels?: number[];
  cluster_counts?: Record<string, number>;
  n_found?: number;
  silhouette?: number | null;
  inertia?: number | null;
  pca?: PcaPayload;
  metrics?: Record<string, unknown>;
  params?: {
    algorithm?: Algorithm;
    params?: Record<string, unknown>;
  };
  features?: string[];
  has_model?: boolean;
  session_id?: string;
}

interface ElbowResult extends ApiJson {
  ks?: number[];
  inertias?: number[];
  warning?: string;
}

interface ClusterBatchRow {
  source: Record<string, string>;
  cluster?: number;
}

const CLUSTER_COLORS = ["#a4512a", "#326b7b", "#7c6a34", "#6f5b9a", "#3f7b55", "#a23b52", "#506b9a", "#8b6f47"];

function formatNumber(value: unknown, digits = 4) {
  const n = Number(value);
  return Number.isFinite(n) ? n.toFixed(digits) : "-";
}

function valueToText(value: unknown) {
  if (value === null || value === undefined) return "";
  return String(value);
}

function defaultFeatures(profile: DataProfile) {
  return profile.numeric_cols.slice(0, Math.min(6, profile.numeric_cols.length));
}

function normalizeClusterResult(payload: ApiJson | null): ClusterResult | null {
  if (!payload) return null;
  const metrics = (payload.metrics ?? {}) as Record<string, unknown>;
  const params = (payload.params ?? {}) as ClusterResult["params"];
  const rawAlgorithm = payload.algorithm || params?.algorithm;
  const algorithm: Algorithm = rawAlgorithm === "dbscan" ? "dbscan" : "kmeans";
  return {
    ...payload,
    algorithm,
    n_found: payload.n_found as number | undefined ?? metrics.n_clusters as number | undefined,
    silhouette: payload.silhouette as number | null | undefined ?? metrics.silhouette as number | null | undefined,
    inertia: payload.inertia as number | null | undefined ?? metrics.inertia as number | null | undefined
  } as ClusterResult;
}

function clusterColor(label: number | string) {
  const n = Number(label);
  if (n === -1) return "#8a8580";
  const index = Number.isFinite(n) ? Math.abs(n) % CLUSTER_COLORS.length : 0;
  return CLUSTER_COLORS[index];
}

function cleanRowsEstimate(profile: DataProfile, featureCols: string[]) {
  if (!featureCols.length) return 0;
  const maxMissing = Math.max(...featureCols.map((column) => profile.missing_counts[column] ?? 0), 0);
  return Math.max(0, profile.n_rows - maxMissing);
}

function compatibleFeatures(profile: DataProfile, features: unknown) {
  return Array.isArray(features) && features.length > 0 && features.every((column) => profile.numeric_cols.includes(String(column)));
}

function ClusterScatter({ result }: { result: ClusterResult | null }) {
  const x = result?.pca?.x ?? [];
  const y = result?.pca?.y ?? [];
  const labels = result?.labels ?? [];
  if (!x.length || !y.length) {
    return <div className="empty-list">训练后显示 PCA 二维聚类图。</div>;
  }

  const width = 680;
  const height = 380;
  const padding = 42;
  const minX = Math.min(...x);
  const maxX = Math.max(...x);
  const minY = Math.min(...y);
  const maxY = Math.max(...y);
  const scaleX = (value: number) => padding + ((value - minX) / Math.max(maxX - minX, 1e-9)) * (width - padding * 2);
  const scaleY = (value: number) => height - padding - ((value - minY) / Math.max(maxY - minY, 1e-9)) * (height - padding * 2);
  const step = Math.max(1, Math.ceil(x.length / 700));

  return (
    <div className="cluster-scatter-wrap">
      <svg className="cluster-scatter" viewBox={`0 0 ${width} ${height}`} role="img" aria-label="PCA 聚类散点图">
        <line x1={padding} y1={height - padding} x2={width - padding} y2={height - padding} />
        <line x1={padding} y1={padding} x2={padding} y2={height - padding} />
        {x.map((value, index) => {
          if (index % step !== 0) return null;
          const label = labels[index] ?? 0;
          return (
            <circle
              cx={scaleX(value)}
              cy={scaleY(y[index] ?? 0)}
              fill={clusterColor(label)}
              key={`${value}-${index}`}
              r={label === -1 ? 3.2 : 4.2}
            />
          );
        })}
        <text x={width - padding} y={height - 12}>PC1 {formatNumber((result?.pca?.ev1 ?? 0) * 100, 1)}%</text>
        <text x={10} y={padding - 16}>PC2 {formatNumber((result?.pca?.ev2 ?? 0) * 100, 1)}%</text>
      </svg>
    </div>
  );
}

function ElbowChart({ result }: { result: ElbowResult | null }) {
  const ks = result?.ks ?? [];
  const inertias = result?.inertias ?? [];
  if (!ks.length || !inertias.length) return <div className="empty-list">点击“计算肘部法则”后显示 K 与 inertia 曲线。</div>;
  const width = 620;
  const height = 240;
  const padding = 34;
  const maxInertia = Math.max(...inertias);
  const minInertia = Math.min(...inertias);
  const scaleX = (index: number) => padding + (index / Math.max(ks.length - 1, 1)) * (width - padding * 2);
  const scaleY = (value: number) => height - padding - ((value - minInertia) / Math.max(maxInertia - minInertia, 1e-9)) * (height - padding * 2);
  const d = inertias.map((value, index) => `${index === 0 ? "M" : "L"} ${scaleX(index)} ${scaleY(value)}`).join(" ");

  return (
    <div className="elbow-chart-wrap">
      <svg className="elbow-chart" viewBox={`0 0 ${width} ${height}`} role="img" aria-label="肘部法则曲线">
        <line x1={padding} y1={height - padding} x2={width - padding} y2={height - padding} />
        <line x1={padding} y1={padding} x2={padding} y2={height - padding} />
        <path d={d} />
        {inertias.map((value, index) => (
          <g key={ks[index]}>
            <circle cx={scaleX(index)} cy={scaleY(value)} r="4" />
            <text x={scaleX(index)} y={height - 10}>{ks[index]}</text>
          </g>
        ))}
      </svg>
      {result?.warning ? <div className="inline-warning">{result.warning}</div> : null}
    </div>
  );
}

function ClusterCounts({ counts }: { counts?: Record<string, number> }) {
  const entries = Object.entries(counts ?? {}).sort(([a], [b]) => Number(a) - Number(b));
  if (!entries.length) return <div className="empty-list">训练后显示每个簇的样本数量。</div>;
  const total = entries.reduce((sum, [, value]) => sum + Number(value), 0);
  return (
    <div className="cluster-counts">
      {entries.map(([label, count]) => (
        <div className="cluster-count-row" key={label}>
          <span><i style={{ backgroundColor: clusterColor(label) }} />簇 {label}</span>
          <div><b style={{ width: `${Math.max(2, (Number(count) / Math.max(total, 1)) * 100)}%` }} /></div>
          <strong>{count}</strong>
        </div>
      ))}
    </div>
  );
}

function clusterExplanation(cluster: unknown, result: ClusterResult | null) {
  const key = String(cluster);
  const count = result?.cluster_counts?.[key];
  if (key === "-1") {
    return count === undefined
      ? "该样本被判为噪声点。"
      : `该样本被判为噪声点，训练集中噪声点有 ${count.toLocaleString()} 条。`;
  }
  if (count !== undefined) {
    return `该样本被分配到簇 ${key}，训练集中该簇包含 ${count.toLocaleString()} 条样本。`;
  }
  return `该样本被分配到簇 ${key}。簇编号只表示分组，不代表好坏或大小顺序。`;
}

function parseCsv(text: string) {
  const rows: string[][] = [];
  let cell = "";
  let row: string[] = [];
  let quoted = false;
  for (let i = 0; i < text.length; i += 1) {
    const char = text[i];
    const next = text[i + 1];
    if (char === '"' && quoted && next === '"') {
      cell += '"';
      i += 1;
    } else if (char === '"') {
      quoted = !quoted;
    } else if (char === "," && !quoted) {
      row.push(cell);
      cell = "";
    } else if ((char === "\n" || char === "\r") && !quoted) {
      if (char === "\r" && next === "\n") i += 1;
      row.push(cell);
      if (row.some((item) => item.trim() !== "")) rows.push(row);
      row = [];
      cell = "";
    } else {
      cell += char;
    }
  }
  row.push(cell);
  if (row.some((item) => item.trim() !== "")) rows.push(row);
  if (!rows.length) return [];
  const headers = rows[0].map((item) => item.trim());
  return rows.slice(1).map((values) => {
    const record: Record<string, string> = {};
    headers.forEach((header, index) => {
      record[header] = values[index] ?? "";
    });
    return record;
  });
}

function toBatchCsv(rows: ClusterBatchRow[]) {
  const headers = rows.length ? [...Object.keys(rows[0].source), "cluster"] : ["cluster"];
  const escape = (value: unknown) => {
    const text = valueToText(value);
    return /[",\n\r]/.test(text) ? `"${text.replace(/"/g, '""')}"` : text;
  };
  return [
    headers.map(escape).join(","),
    ...rows.map((row) => headers.map((header) => escape(header === "cluster" ? row.cluster : row.source[header])).join(","))
  ].join("\n");
}

export default function ClusteringWorkspace({ profile }: Props) {
  const [algorithm, setAlgorithm] = useState<Algorithm>("kmeans");
  const [featureCols, setFeatureCols] = useState<string[]>(defaultFeatures(profile));
  const [nClusters, setNClusters] = useState(Math.min(3, Math.max(2, profile.n_rows)));
  const [eps, setEps] = useState(0.5);
  const [minSamples, setMinSamples] = useState(5);
  const [maxK, setMaxK] = useState(Math.min(10, Math.max(2, profile.n_rows - 1)));
  const [trainResult, setTrainResult] = useState<ClusterResult | null>(null);
  const [elbowResult, setElbowResult] = useState<ElbowResult | null>(null);
  const [predictResult, setPredictResult] = useState<ApiJson | null>(null);
  const [versions, setVersions] = useState<ModelVersion[]>([]);
  const [activeVersion, setActiveVersion] = useState<string | null | undefined>(null);
  const [selectedVersion, setSelectedVersion] = useState("");
  const [inputs, setInputs] = useState<Record<string, string>>({});
  const [batchRows, setBatchRows] = useState<ClusterBatchRow[]>([]);
  const [loading, setLoading] = useState(false);
  const [versionsLoading, setVersionsLoading] = useState(false);
  const [batchLoading, setBatchLoading] = useState(false);
  const [error, setError] = useState("");
  const [batchError, setBatchError] = useState("");
  const [activeTab, setActiveTab] = useState<ClusterTab>("train");

  const cleanRows = cleanRowsEstimate(profile, featureCols);
  const maxClusterCount = Math.max(2, Math.min(15, cleanRows || profile.n_rows));
  const canTrain = Boolean(profile.session_id && featureCols.length && cleanRows >= 10);
  const predictionEnabled = Boolean(trainResult?.version_id) && trainResult?.algorithm === "kmeans";

  useEffect(() => {
    setFeatureCols(defaultFeatures(profile));
    setNClusters(Math.min(3, Math.max(2, profile.n_rows)));
    setMaxK(Math.min(10, Math.max(2, profile.n_rows - 1)));
    setTrainResult(null);
    setElbowResult(null);
    setPredictResult(null);
    setBatchRows([]);
  }, [profile.session_id]);

  useEffect(() => {
    const firstRow = profile.preview[0] ?? {};
    const next: Record<string, string> = {};
    for (const column of featureCols) next[column] = valueToText(firstRow[column]);
    setInputs(next);
  }, [profile.session_id, featureCols]);

  const featureProfiles = useMemo(
    () => featureCols.map((column) => profile.column_profiles.find((item) => item.name === column)).filter(Boolean),
    [featureCols, profile.column_profiles]
  );

  const loadClusterState = async () => {
    setVersionsLoading(true);
    setError("");
    try {
      const [versionPayload, statusPayload] = await Promise.all([
        fetchModelVersions("clustering"),
        fetchModelStatus("clustering")
      ]);
      setVersions(versionPayload.versions ?? []);
      setActiveVersion(versionPayload.active);
      const activeMeta = (versionPayload.versions ?? []).find((version) => version.version_id === versionPayload.active);
      const activeCompatible = compatibleFeatures(profile, activeMeta?.features);
      setSelectedVersion(activeCompatible ? versionPayload.active ?? "" : "");
      const status = normalizeClusterResult(statusPayload);
      if (status?.has_model && compatibleFeatures(profile, status.features)) {
        setTrainResult(status);
        if (status.algorithm === "kmeans" || status.algorithm === "dbscan") setAlgorithm(status.algorithm);
        if (status.features?.length) setFeatureCols(status.features);
        const rawParams = status.params?.params ?? {};
        if (typeof rawParams.n_clusters === "number") setNClusters(rawParams.n_clusters);
        if (typeof rawParams.eps === "number") setEps(rawParams.eps);
        if (typeof rawParams.min_samples === "number") setMinSamples(rawParams.min_samples);
      } else {
        setTrainResult(null);
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : "读取聚类状态失败");
    } finally {
      setVersionsLoading(false);
    }
  };

  useEffect(() => {
    void loadClusterState();
  }, [profile.session_id]);

  const toggleFeature = (column: string) => {
    setFeatureCols((current) =>
      current.includes(column) ? current.filter((item) => item !== column) : [...current, column]
    );
  };

  const handleTrain = async () => {
    if (!canTrain) return;
    setLoading(true);
    setError("");
    setPredictResult(null);
    setBatchRows([]);
    try {
      const payload = await trainModel("clustering", {
        session_id: profile.session_id,
        feature_cols: featureCols,
        algorithm,
        params: algorithm === "kmeans" ? { n_clusters: nClusters } : { eps, min_samples: minSamples }
      });
      const normalized = normalizeClusterResult(payload);
      setTrainResult(normalized);
      setActiveTab("result");
      if (normalized?.version_id) {
        setSelectedVersion(normalized.version_id);
        setActiveVersion(normalized.version_id);
      }
      await loadClusterState();
    } catch (err) {
      setError(err instanceof Error ? err.message : "聚类训练失败");
    } finally {
      setLoading(false);
    }
  };

  const handleElbow = async () => {
    if (!profile.session_id || !featureCols.length) return;
    setLoading(true);
    setError("");
    try {
      const payload = await fetchClusteringElbow({
        session_id: profile.session_id,
        feature_cols: featureCols,
        max_k: maxK
      });
      setElbowResult(payload as ElbowResult);
      setActiveTab("result");
    } catch (err) {
      setError(err instanceof Error ? err.message : "肘部法则计算失败");
    } finally {
      setLoading(false);
    }
  };

  const activateVersion = async (versionId: string) => {
    setError("");
    try {
      await activateModelVersion("clustering", versionId);
      const detail = await fetchModelVersionDetail("clustering", versionId);
      const normalized = normalizeClusterResult({ ...detail, has_model: true });
      setTrainResult(normalized);
      setSelectedVersion(versionId);
      if (normalized?.algorithm) setAlgorithm(normalized.algorithm);
      if (normalized?.features?.length) setFeatureCols(normalized.features);
      await loadClusterState();
    } catch (err) {
      setError(err instanceof Error ? err.message : "切换聚类版本失败");
    }
  };

  const removeVersion = async (versionId: string) => {
    setError("");
    try {
      await deleteModelVersion("clustering", versionId);
      if (selectedVersion === versionId) setSelectedVersion("");
      await loadClusterState();
    } catch (err) {
      setError(err instanceof Error ? err.message : "删除版本失败");
    }
  };

  const clearActive = async () => {
    setError("");
    try {
      await clearModel("clustering");
      setTrainResult(null);
      setPredictResult(null);
      setBatchRows([]);
      await loadClusterState();
    } catch (err) {
      setError(err instanceof Error ? err.message : "清除模型失败");
    }
  };

  const handleBatchFile = async (event: ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    event.target.value = "";
    if (!file) return;
    setBatchError("");
    setBatchRows([]);
    try {
      const parsed = parseCsv(await file.text());
      if (!parsed.length) {
        setBatchError("CSV 文件为空或无法解析。");
        return;
      }
      if (parsed.length > 10000) {
        setBatchError(`单次最多支持 10000 行，当前为 ${parsed.length} 行。`);
        return;
      }
      const missing = featureCols.filter((column) => !(column in parsed[0]));
      if (missing.length) {
        setBatchError(`CSV 缺少模型需要的特征列：${missing.join("、")}`);
        return;
      }
      setBatchRows(parsed.map((row) => ({ source: row })));
    } catch (err) {
      setBatchError(err instanceof Error ? err.message : "CSV 读取失败");
    }
  };

  const handleBatchPredict = async () => {
    if (!batchRows.length || !predictionEnabled) return;
    setBatchLoading(true);
    setBatchError("");
    try {
      const rows = batchRows.map((row) => featureCols.map((column) => Number(row.source[column])));
      if (rows.some((row) => row.some((value) => !Number.isFinite(value)))) {
        setBatchError("批量数据包含非数值、空值或无穷值，请先清洗后再预测。");
        return;
      }
      const payload = await batchPredictModel("clustering", {
        rows,
        version_id: selectedVersion || undefined
      });
      const clusters = (payload.clusters ?? []) as number[];
      setBatchRows((current) => current.map((row, index) => ({ ...row, cluster: Number(clusters[index]) })));
    } catch (err) {
      setBatchError(err instanceof Error ? err.message : "聚类批量预测失败");
    } finally {
      setBatchLoading(false);
    }
  };

  const downloadBatchResult = () => {
    const csv = toBatchCsv(batchRows);
    const blob = new Blob([`\ufeff${csv}`], { type: "text/csv;charset=utf-8" });
    const url = URL.createObjectURL(blob);
    const anchor = document.createElement("a");
    anchor.href = url;
    anchor.download = "clustering_predictions.csv";
    anchor.click();
    URL.revokeObjectURL(url);
  };

  const handlePredict = async () => {
    if (!featureCols.length || !predictionEnabled) return;
    setLoading(true);
    setError("");
    setPredictResult(null);
    try {
      const features = featureCols.map((column) => Number(inputs[column] ?? 0));
      const payload = await predictModel("clustering", {
        features,
        version_id: selectedVersion || undefined
      });
      setPredictResult(payload);
    } catch (err) {
      setError(err instanceof Error ? err.message : "聚类预测失败");
    } finally {
      setLoading(false);
    }
  };

  return (
    <section className="clustering-workspace" aria-label="聚类工作台">
      <div className="dt-toolbar">
        <div>
          <span>当前数据</span>
          <strong>{profile.n_rows.toLocaleString()} 行 · {profile.n_cols} 列</strong>
        </div>
        <div>
          <span>可用数值列</span>
          <strong>{profile.numeric_cols.length} 列</strong>
        </div>
        <div>
          <span>有效行估计</span>
          <strong>{cleanRows.toLocaleString()} 行</strong>
        </div>
      </div>

      {error ? <div className="inline-error">{error}</div> : null}
      {!profile.numeric_cols.length ? <div className="inline-warning">聚类训练需要至少一个数值列。</div> : null}
      {cleanRows < 10 && featureCols.length ? <div className="inline-warning">聚类训练至少需要 10 行无缺失的特征数据。</div> : null}
      {activeVersion && !trainResult ? (
        <div className="inline-warning">当前激活的聚类模型来自其他字段结构。版本已保留在列表中，当前数据请重新训练。</div>
      ) : null}

      <div className="workflow-tabbar three" role="tablist" aria-label="聚类功能分区">
        <button className={activeTab === "train" ? "workflow-tab active" : "workflow-tab"} type="button" role="tab" aria-selected={activeTab === "train"} onClick={() => setActiveTab("train")}>聚类配置</button>
        <button className={activeTab === "result" ? "workflow-tab active" : "workflow-tab"} type="button" role="tab" aria-selected={activeTab === "result"} onClick={() => setActiveTab("result")}>结果可视化</button>
        <button className={activeTab === "predict" ? "workflow-tab active" : "workflow-tab"} type="button" role="tab" aria-selected={activeTab === "predict"} onClick={() => setActiveTab("predict")}>模型预测</button>
      </div>

      {activeTab === "train" || activeTab === "result" ? (
      <div className="dt-layout cluster-layout single-pane">
        {activeTab === "train" ? (
        <aside className="dt-config cluster-config" aria-label="聚类训练配置">
          <div className="panel-title split">
            <span>
              <CircleDot size={17} aria-hidden="true" />
              <h2>聚类配置</h2>
            </span>
            <small>{algorithm === "kmeans" ? "K-means" : "DBSCAN"}</small>
          </div>

          <div className="segmented">
            <button className={algorithm === "kmeans" ? "selected" : ""} type="button" onClick={() => setAlgorithm("kmeans")}>K-means</button>
            <button className={algorithm === "dbscan" ? "selected" : ""} type="button" onClick={() => setAlgorithm("dbscan")}>DBSCAN</button>
          </div>

          <div className="form-field">
            <span>特征列</span>
            <div className="column-select compact tall">
              {profile.numeric_cols.map((column) => (
                <button
                  className={featureCols.includes(column) ? "column-choice selected" : "column-choice"}
                  key={column}
                  type="button"
                  onClick={() => toggleFeature(column)}
                >
                  {column}
                </button>
              ))}
            </div>
          </div>

          {algorithm === "kmeans" ? (
            <div className="operation-columns two">
              <label className="form-field">
                <span>聚类数量 K：{nClusters}</span>
                <input min={2} max={maxClusterCount} type="range" value={nClusters} onChange={(event) => setNClusters(Number(event.target.value))} />
              </label>
              <label className="form-field">
                <span>肘部最大 K</span>
                <input min={2} max={Math.max(2, Math.min(15, profile.n_rows - 1))} type="number" value={maxK} onChange={(event) => setMaxK(Number(event.target.value))} />
              </label>
            </div>
          ) : (
            <div className="operation-columns two">
              <label className="form-field">
                <span>eps</span>
                <input min={0.05} step={0.05} type="number" value={eps} onChange={(event) => setEps(Number(event.target.value))} />
              </label>
              <label className="form-field">
                <span>min_samples</span>
                <input min={2} step={1} type="number" value={minSamples} onChange={(event) => setMinSamples(Number(event.target.value))} />
              </label>
            </div>
          )}

          <div className="dt-feature-summary">
            <span>已选特征 {featureCols.length}</span>
            <span>缺失影响 {profile.n_rows - cleanRows}</span>
            <span>{algorithm === "kmeans" ? `K=${nClusters}` : `eps=${eps}`}</span>
          </div>

          <div className="prediction-actions">
            <button className="button primary" type="button" disabled={!canTrain || loading} onClick={handleTrain}>
              <Play size={15} aria-hidden="true" />
              {loading ? "处理中..." : "开始聚类"}
            </button>
            {algorithm === "kmeans" ? (
              <button className="button ghost" type="button" disabled={!featureCols.length || loading} onClick={handleElbow}>
                <BarChart3 size={15} aria-hidden="true" />
                计算肘部法则
              </button>
            ) : null}
            <button className="button danger" type="button" onClick={clearActive} disabled={loading}>
              <Trash2 size={15} aria-hidden="true" />
              清除当前模型
            </button>
          </div>

          <div className="cluster-feature-notes">
            {featureProfiles.map((column) => (
              <span key={column?.name}>{column?.name}: 缺失 {column?.missing_count ?? 0} · 异常 {column?.outlier_count ?? 0}</span>
            ))}
          </div>
        </aside>
        ) : null}

        {activeTab === "result" ? (
        <section className="dt-main" aria-label="聚类结果">
          <article className="dt-panel">
            <div className="panel-title split">
              <span>
                <GitPullRequest size={17} aria-hidden="true" />
                <h2>聚类结果</h2>
              </span>
              <small>{trainResult?.version_id ? `version ${trainResult.version_id.slice(0, 12)}` : "等待训练"}</small>
            </div>
            <div className="metric-grid">
              <div className="mini-metric"><span>算法</span><strong>{trainResult?.algorithm === "dbscan" ? "DBSCAN" : "K-means"}</strong></div>
              <div className="mini-metric"><span>簇数量</span><strong>{trainResult?.n_found ?? "-"}</strong></div>
              <div className="mini-metric"><span>轮廓系数</span><strong>{formatNumber(trainResult?.silhouette)}</strong></div>
              <div className="mini-metric"><span>Inertia</span><strong>{formatNumber(trainResult?.inertia, 2)}</strong></div>
            </div>
            <ClusterScatter result={trainResult} />
          </article>

          <article className="dt-panel">
            <div className="panel-title split">
              <span>
                <BarChart3 size={17} aria-hidden="true" />
                <h2>簇规模</h2>
              </span>
              <small>{trainResult?.cluster_counts?.["-1"] ? "包含噪声点" : "训练样本分布"}</small>
            </div>
            <ClusterCounts counts={trainResult?.cluster_counts} />
          </article>

          <article className="dt-panel">
            <div className="panel-title split">
              <span>
                <BarChart3 size={17} aria-hidden="true" />
                <h2>肘部法则</h2>
              </span>
              <small>K-means 辅助选 K</small>
            </div>
            <ElbowChart result={elbowResult} />
          </article>
        </section>
        ) : null}
      </div>
      ) : null}

      {activeTab === "predict" ? (
      <>
        <div className="dt-bottom-grid">
          <article className="dt-panel">
          <div className="panel-title split">
            <span>
              <RefreshCcw size={17} aria-hidden="true" />
              <h2>模型版本</h2>
            </span>
            <button className="tiny-button" type="button" onClick={loadClusterState} disabled={versionsLoading}>刷新</button>
          </div>
          <div className="version-list tall">
            {versions.length ? versions.map((version) => (
              <div className="version-row" key={version.version_id}>
                <div>
                  <strong>{version.version_id}</strong>
                  <span>{version.dataset_name || "unknown"} · {String(version.params?.algorithm ?? "clustering")} · {version.created_at || "-"}</span>
                </div>
                <div className="version-actions">
                  <button className={activeVersion === version.version_id ? "model-badge active" : "model-badge clickable"} type="button" onClick={() => activateVersion(version.version_id)}>
                    {activeVersion === version.version_id ? "Active" : "激活"}
                  </button>
                  <button className="icon-button danger-icon" type="button" aria-label={`删除版本 ${version.version_id}`} title="删除版本" onClick={() => removeVersion(version.version_id)}>
                    <Trash2 size={14} aria-hidden="true" />
                  </button>
                </div>
              </div>
            )) : <div className="empty-list">暂无聚类版本。</div>}
          </div>
          </article>

          <article className="dt-panel">
          <div className="panel-title split">
            <span>
              <Send size={17} aria-hidden="true" />
              <h2>单条预测</h2>
            </span>
            <small>{trainResult ? (predictionEnabled ? "K-means 可用" : "DBSCAN 不支持 predict") : "等待 K-means 模型"}</small>
          </div>
          <div className="dt-predict-grid">
            {featureCols.slice(0, 12).map((column) => (
              <label className="form-field" key={column}>
                <span>{column}</span>
                <input value={inputs[column] ?? ""} onChange={(event) => setInputs({ ...inputs, [column]: event.target.value })} />
              </label>
            ))}
          </div>
          <label className="form-field">
            <span>预测版本</span>
            <select value={selectedVersion} onChange={(event) => setSelectedVersion(event.target.value)}>
              <option value="">当前激活版本</option>
              {versions.map((version) => <option key={version.version_id} value={version.version_id}>{version.version_id}</option>)}
            </select>
          </label>
          <button className="button primary full" type="button" onClick={handlePredict} disabled={!featureCols.length || loading || !predictionEnabled}>
            <Brain size={15} aria-hidden="true" />
            执行聚类预测
          </button>
          {trainResult && !predictionEnabled ? (
            <div className="inline-warning">
              <AlertTriangle size={15} aria-hidden="true" />
              DBSCAN 没有稳定的单点 predict 语义。如需新样本归簇，请切换到 K-means。
            </div>
          ) : null}
          {predictResult ? (
            <div className="prediction-result">
              <strong>预测簇：{String(predictResult.cluster)}</strong>
              <span>{clusterExplanation(predictResult.cluster, trainResult)}</span>
              <span>簇编号只表示分组，不代表大小或好坏顺序。</span>
            </div>
          ) : null}
          </article>
        </div>

        <article className="dt-panel cluster-batch-panel">
          <div className="panel-title split">
            <span>
              <Upload size={17} aria-hidden="true" />
              <h2>批量预测</h2>
            </span>
            <small>{predictionEnabled ? "CSV · 最多 10000 行" : "仅 K-means 支持"}</small>
          </div>
          <div className="batch-toolbar">
            <label className="button ghost file-button">
              <Upload size={15} aria-hidden="true" />
              上传批量 CSV
              <input type="file" accept=".csv" onChange={handleBatchFile} />
            </label>
            <button className="button primary" type="button" disabled={!batchRows.length || batchLoading || !predictionEnabled} onClick={handleBatchPredict}>
              {batchLoading ? "预测中..." : "执行批量预测"}
            </button>
            <button className="button ghost" type="button" disabled={!batchRows.some((row) => row.cluster !== undefined)} onClick={downloadBatchResult}>
              <Download size={15} aria-hidden="true" />
              下载结果
            </button>
          </div>
          {batchError ? <div className="inline-error">{batchError}</div> : null}
          {!predictionEnabled ? (
            <div className="inline-warning">
              <AlertTriangle size={15} aria-hidden="true" />
              DBSCAN 不支持稳定批量 predict；请切换或训练 K-means 版本。
            </div>
          ) : null}
          {batchRows.length ? (
            <div className="table-wrap compact-table">
              <table className="data-table">
                <thead>
                  <tr>
                    {featureCols.slice(0, 8).map((column) => <th key={column}>{column}</th>)}
                    <th>预测簇</th>
                  </tr>
                </thead>
                <tbody>
                  {batchRows.slice(0, 12).map((row, index) => (
                    <tr key={index}>
                      {featureCols.slice(0, 8).map((column) => <td key={column}>{row.source[column]}</td>)}
                      <td>{row.cluster ?? "-"}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          ) : <div className="empty-list">上传包含当前特征列的 CSV 后可批量预测。</div>}
        </article>
      </>
      ) : null}
    </section>
  );
}
