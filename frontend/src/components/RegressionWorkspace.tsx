import { ChangeEvent, useEffect, useMemo, useState } from "react";
import {
  AlertTriangle,
  Brain,
  CheckCircle2,
  Download,
  LineChart,
  Play,
  RefreshCcw,
  Send,
  SlidersHorizontal,
  Trash2,
  Upload
} from "lucide-react";
import {
  activateModelVersion,
  batchPredictModel,
  clearModel,
  deleteModelVersion,
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

interface RegressionResult extends ApiJson {
  version_id?: string;
  r2?: number;
  mae?: number;
  rmse?: number;
  train_losses?: number[];
  val_losses?: number[];
  metrics?: Record<string, unknown>;
  params?: Record<string, unknown>;
  features?: string[];
  target?: string;
  has_model?: boolean;
}

interface BatchRow {
  source: Record<string, string>;
  prediction?: number;
}

const LEARNING_RATES = [0.01, 0.005, 0.001, 0.0005, 0.0001];
const BATCH_SIZES = [4, 8, 16, 32, 64, 128];

function formatNumber(value: unknown, digits = 4) {
  const n = Number(value);
  return Number.isFinite(n) ? n.toFixed(digits) : "-";
}

function valueToText(value: unknown) {
  if (value === null || value === undefined) return "";
  return String(value);
}

function defaultTarget(profile: DataProfile) {
  return profile.numeric_cols[profile.numeric_cols.length - 1] ?? "";
}

function defaultFeatures(profile: DataProfile, target: string) {
  return profile.numeric_cols.filter((column) => column !== target).slice(0, Math.min(8, Math.max(0, profile.numeric_cols.length - 1)));
}

function cleanRowsEstimate(profile: DataProfile, columns: string[]) {
  if (!columns.length) return 0;
  const maxMissing = Math.max(...columns.map((column) => profile.missing_counts[column] ?? 0), 0);
  return Math.max(0, profile.n_rows - maxMissing);
}

function compatibleRegression(profile: DataProfile, payload: { features?: unknown; target?: unknown } | null | undefined) {
  const features = payload?.features;
  const target = String(payload?.target ?? "");
  return (
    Boolean(target && profile.numeric_cols.includes(target)) &&
    Array.isArray(features) &&
    features.length > 0 &&
    features.every((column) => profile.numeric_cols.includes(String(column)) && String(column) !== target)
  );
}

function normalizeRegressionResult(payload: ApiJson | null): RegressionResult | null {
  if (!payload) return null;
  const metrics = (payload.metrics ?? {}) as Record<string, unknown>;
  return {
    ...payload,
    r2: payload.r2 as number | undefined ?? metrics.r2 as number | undefined,
    mae: payload.mae as number | undefined ?? metrics.mae as number | undefined,
    rmse: payload.rmse as number | undefined ?? metrics.rmse as number | undefined
  } as RegressionResult;
}

function architectureFor(sampleCount: number, featureCount: number) {
  if (sampleCount < 500) {
    return { h1: Math.max(8, featureCount), h2: Math.max(4, Math.floor(featureCount / 2)), dropout: 0.1, label: "轻量网络" };
  }
  if (sampleCount < 5000) {
    return { h1: featureCount * 2, h2: featureCount, dropout: 0.2, label: "标准网络" };
  }
  return { h1: featureCount * 3, h2: featureCount * 2, dropout: 0.3, label: "深层网络" };
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

function toCsv(rows: BatchRow[], target: string) {
  const headers = rows.length ? [...Object.keys(rows[0].source), `prediction_${target}`] : [`prediction_${target}`];
  const escape = (value: unknown) => {
    const text = valueToText(value);
    return /[",\n\r]/.test(text) ? `"${text.replace(/"/g, '""')}"` : text;
  };
  return [
    headers.map(escape).join(","),
    ...rows.map((row) => headers.map((header) => escape(header === `prediction_${target}` ? row.prediction : row.source[header])).join(","))
  ].join("\n");
}

function LossChart({ train, val }: { train?: number[]; val?: number[] }) {
  const trainLosses = train ?? [];
  const valLosses = val ?? [];
  const count = Math.max(trainLosses.length, valLosses.length);
  if (!count) return <div className="empty-list">训练后显示训练 / 验证损失曲线。</div>;

  const width = 680;
  const height = 260;
  const padding = 36;
  const values = [...trainLosses, ...valLosses].filter((value) => Number.isFinite(Number(value)));
  const minY = Math.min(...values);
  const maxY = Math.max(...values);
  const scaleX = (index: number) => padding + (index / Math.max(count - 1, 1)) * (width - padding * 2);
  const scaleY = (value: number) => height - padding - ((value - minY) / Math.max(maxY - minY, 1e-9)) * (height - padding * 2);
  const pathFor = (losses: number[]) => losses.map((value, index) => `${index === 0 ? "M" : "L"} ${scaleX(index)} ${scaleY(value)}`).join(" ");

  return (
    <div className="loss-chart-wrap">
      <svg className="loss-chart" viewBox={`0 0 ${width} ${height}`} role="img" aria-label="训练和验证损失曲线">
        <line x1={padding} y1={height - padding} x2={width - padding} y2={height - padding} />
        <line x1={padding} y1={padding} x2={padding} y2={height - padding} />
        {trainLosses.length ? <path className="train-line" d={pathFor(trainLosses)} /> : null}
        {valLosses.length ? <path className="val-line" d={pathFor(valLosses)} /> : null}
        <text x={padding} y={22}>MSE loss</text>
        <text x={width - 122} y={22}>epochs {count}</text>
      </svg>
      <div className="chart-legend">
        <span><i className="train-dot" />训练损失</span>
        <span><i className="val-dot" />验证损失</span>
      </div>
    </div>
  );
}

export default function RegressionWorkspace({ profile }: Props) {
  const [targetCol, setTargetCol] = useState(defaultTarget(profile));
  const [featureCols, setFeatureCols] = useState<string[]>(defaultFeatures(profile, targetCol));
  const [learningRate, setLearningRate] = useState(0.001);
  const [epochs, setEpochs] = useState(100);
  const [batchSize, setBatchSize] = useState(8);
  const [device, setDevice] = useState<"cpu" | "cuda">("cpu");
  const [trainResult, setTrainResult] = useState<RegressionResult | null>(null);
  const [predictResult, setPredictResult] = useState<ApiJson | null>(null);
  const [batchRows, setBatchRows] = useState<BatchRow[]>([]);
  const [versions, setVersions] = useState<ModelVersion[]>([]);
  const [activeVersion, setActiveVersion] = useState<string | null | undefined>(null);
  const [selectedVersion, setSelectedVersion] = useState("");
  const [inputs, setInputs] = useState<Record<string, string>>({});
  const [loading, setLoading] = useState(false);
  const [versionsLoading, setVersionsLoading] = useState(false);
  const [batchLoading, setBatchLoading] = useState(false);
  const [error, setError] = useState("");
  const [batchError, setBatchError] = useState("");

  const targetProfile = profile.column_profiles.find((column) => column.name === targetCol);
  const cleanRows = cleanRowsEstimate(profile, [...featureCols, targetCol].filter(Boolean));
  const architecture = architectureFor(cleanRows || profile.n_rows, featureCols.length);
  const constantFeatures = useMemo(
    () => featureCols.filter((column) => {
      const columnProfile = profile.column_profiles.find((item) => item.name === column);
      return (columnProfile?.unique_count ?? 2) <= 1;
    }),
    [featureCols, profile.column_profiles]
  );
  const canTrain = Boolean(profile.session_id && targetCol && featureCols.length && cleanRows >= 10 && !constantFeatures.length);

  useEffect(() => {
    const nextTarget = defaultTarget(profile);
    setTargetCol(nextTarget);
    setFeatureCols(defaultFeatures(profile, nextTarget));
    setTrainResult(null);
    setPredictResult(null);
    setBatchRows([]);
  }, [profile.session_id]);

  useEffect(() => {
    const firstRow = profile.preview[0] ?? {};
    const next: Record<string, string> = {};
    for (const column of featureCols) next[column] = valueToText(firstRow[column]);
    setInputs(next);
  }, [profile.session_id, featureCols]);

  const loadRegressionState = async () => {
    setVersionsLoading(true);
    setError("");
    try {
      const [versionPayload, statusPayload] = await Promise.all([
        fetchModelVersions("regression"),
        fetchModelStatus("regression")
      ]);
      setVersions(versionPayload.versions ?? []);
      setActiveVersion(versionPayload.active);
      const activeMeta = (versionPayload.versions ?? []).find((version) => version.version_id === versionPayload.active);
      const activeCompatible = compatibleRegression(profile, activeMeta);
      setSelectedVersion(activeCompatible ? versionPayload.active ?? "" : "");
      const status = normalizeRegressionResult(statusPayload);
      if (status?.has_model && compatibleRegression(profile, status)) {
        setTrainResult(status);
        if (status.target) setTargetCol(status.target);
        if (status.features?.length) setFeatureCols(status.features);
        if (typeof status.params?.learning_rate === "number") setLearningRate(status.params.learning_rate);
        if (typeof status.params?.epochs === "number") setEpochs(status.params.epochs);
        if (typeof status.params?.batch_size === "number") setBatchSize(status.params.batch_size);
      } else {
        setTrainResult(null);
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : "读取回归状态失败");
    } finally {
      setVersionsLoading(false);
    }
  };

  useEffect(() => {
    void loadRegressionState();
  }, [profile.session_id]);

  const availableFeatures = profile.numeric_cols.filter((column) => column !== targetCol);

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
      const payload = await trainModel("regression", {
        session_id: profile.session_id,
        target_col: targetCol,
        feature_cols: featureCols,
        learning_rate: learningRate,
        epochs,
        batch_size: batchSize,
        device
      });
      const normalized = normalizeRegressionResult(payload);
      setTrainResult(normalized);
      if (normalized?.version_id) {
        setSelectedVersion(normalized.version_id);
        setActiveVersion(normalized.version_id);
      }
      await loadRegressionState();
    } catch (err) {
      setError(err instanceof Error ? err.message : "回归训练失败");
    } finally {
      setLoading(false);
    }
  };

  const activateVersion = async (versionId: string) => {
    setError("");
    try {
      await activateModelVersion("regression", versionId);
      const detail = await fetchModelVersionDetail("regression", versionId);
      const normalized = normalizeRegressionResult({ ...detail, has_model: true });
      if (compatibleRegression(profile, normalized)) {
        setTrainResult(normalized);
        setSelectedVersion(versionId);
        if (normalized?.target) setTargetCol(normalized.target);
        if (normalized?.features?.length) setFeatureCols(normalized.features);
      } else {
        setTrainResult(null);
        setSelectedVersion("");
      }
      setPredictResult(null);
      setBatchRows([]);
      await loadRegressionState();
    } catch (err) {
      setError(err instanceof Error ? err.message : "切换回归版本失败");
    }
  };

  const removeVersion = async (versionId: string) => {
    setError("");
    try {
      await deleteModelVersion("regression", versionId);
      if (selectedVersion === versionId) setSelectedVersion("");
      await loadRegressionState();
    } catch (err) {
      setError(err instanceof Error ? err.message : "删除版本失败");
    }
  };

  const clearActive = async () => {
    setError("");
    try {
      await clearModel("regression");
      setTrainResult(null);
      setPredictResult(null);
      setBatchRows([]);
      await loadRegressionState();
    } catch (err) {
      setError(err instanceof Error ? err.message : "清除模型失败");
    }
  };

  const handlePredict = async () => {
    if (!featureCols.length || !trainResult?.version_id) return;
    setLoading(true);
    setError("");
    setPredictResult(null);
    try {
      const features = featureCols.map((column) => Number(inputs[column] ?? 0));
      const result = await predictModel("regression", {
        features,
        device,
        version_id: selectedVersion || undefined
      });
      setPredictResult(result);
    } catch (err) {
      setError(err instanceof Error ? err.message : "回归预测失败");
    } finally {
      setLoading(false);
    }
  };

  const handleBatchFile = async (event: ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    event.target.value = "";
    if (!file) return;
    setBatchError("");
    setBatchRows([]);
    try {
      const text = await file.text();
      const parsed = parseCsv(text);
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
      const rows = parsed.map((row) => ({ source: row }));
      setBatchRows(rows);
    } catch (err) {
      setBatchError(err instanceof Error ? err.message : "CSV 读取失败");
    }
  };

  const handleBatchPredict = async () => {
    if (!batchRows.length || !trainResult?.version_id) return;
    setBatchLoading(true);
    setBatchError("");
    try {
      const rows = batchRows.map((row) => featureCols.map((column) => Number(row.source[column])));
      if (rows.some((row) => row.some((value) => !Number.isFinite(value)))) {
        setBatchError("批量数据包含非数值、空值或无穷值，请先清洗后再预测。");
        return;
      }
      const result = await batchPredictModel("regression", {
        rows,
        device,
        version_id: selectedVersion || undefined
      });
      const predictions = (result.predictions ?? []) as number[];
      setBatchRows((current) => current.map((row, index) => ({ ...row, prediction: Number(predictions[index]) })));
    } catch (err) {
      setBatchError(err instanceof Error ? err.message : "批量预测失败");
    } finally {
      setBatchLoading(false);
    }
  };

  const downloadBatchResult = () => {
    const csv = toCsv(batchRows, targetCol);
    const blob = new Blob([`\ufeff${csv}`], { type: "text/csv;charset=utf-8" });
    const url = URL.createObjectURL(blob);
    const anchor = document.createElement("a");
    anchor.href = url;
    anchor.download = "regression_predictions.csv";
    anchor.click();
    URL.revokeObjectURL(url);
  };

  return (
    <section className="regression-workspace" aria-label="回归工作台">
      <div className="dt-toolbar">
        <div>
          <span>当前数据</span>
          <strong>{profile.n_rows.toLocaleString()} 行 · {profile.n_cols} 列</strong>
        </div>
        <div>
          <span>数值字段</span>
          <strong>{profile.numeric_cols.length} 列</strong>
        </div>
        <div>
          <span>有效行估计</span>
          <strong>{cleanRows.toLocaleString()} 行</strong>
        </div>
      </div>

      {error ? <div className="inline-error">{error}</div> : null}
      {activeVersion && !trainResult ? (
        <div className="inline-warning">当前激活的回归模型与当前数据字段不兼容。版本仍保留在列表中，当前数据请重新训练。</div>
      ) : null}
      {cleanRows < 10 && featureCols.length ? <div className="inline-warning">回归训练至少需要 10 行无缺失的数值数据。</div> : null}
      {constantFeatures.length ? <div className="inline-warning">常量特征会影响训练：{constantFeatures.join("、")}。建议先到数据处理页移除。</div> : null}

      <div className="dt-layout regression-layout">
        <aside className="dt-config regression-config" aria-label="回归训练配置">
          <div className="panel-title split">
            <span>
              <SlidersHorizontal size={17} aria-hidden="true" />
              <h2>训练配置</h2>
            </span>
            <small>{architecture.label}</small>
          </div>

          <label className="form-field">
            <span>预测目标列</span>
            <select value={targetCol} onChange={(event) => {
              setTargetCol(event.target.value);
              setFeatureCols(defaultFeatures(profile, event.target.value));
              setTrainResult(null);
              setPredictResult(null);
            }}>
              {profile.numeric_cols.map((column) => <option key={column}>{column}</option>)}
            </select>
          </label>

          <div className="form-field">
            <span>特征列</span>
            <div className="column-select compact tall">
              {availableFeatures.map((column) => (
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

          <div className="operation-columns three">
            <label className="form-field">
              <span>学习率</span>
              <select value={learningRate} onChange={(event) => setLearningRate(Number(event.target.value))}>
                {LEARNING_RATES.map((rate) => <option key={rate} value={rate}>{rate}</option>)}
              </select>
            </label>
            <label className="form-field">
              <span>最大轮数：{epochs}</span>
              <input min={20} max={500} step={20} type="range" value={epochs} onChange={(event) => setEpochs(Number(event.target.value))} />
            </label>
            <label className="form-field">
              <span>Batch</span>
              <select value={batchSize} onChange={(event) => setBatchSize(Number(event.target.value))}>
                {BATCH_SIZES.map((size) => <option key={size} value={size}>{size}</option>)}
              </select>
            </label>
          </div>

          <div className="segmented">
            <button className={device === "cpu" ? "selected" : ""} type="button" onClick={() => setDevice("cpu")}>CPU</button>
            <button className={device === "cuda" ? "selected" : ""} type="button" onClick={() => setDevice("cuda")}>CUDA</button>
          </div>

          <div className="dt-feature-summary">
            <span>特征 {featureCols.length}</span>
            <span>目标唯一值 {targetProfile?.unique_count ?? "-"}</span>
            <span>{architecture.h1}/{architecture.h2} · dropout {architecture.dropout}</span>
          </div>

          <div className="prediction-actions">
            <button className="button primary" type="button" disabled={!canTrain || loading} onClick={handleTrain}>
              <Play size={15} aria-hidden="true" />
              {loading ? "训练中..." : "开始训练"}
            </button>
            <button className="button danger" type="button" disabled={loading} onClick={clearActive}>
              <Trash2 size={15} aria-hidden="true" />
              清除当前模型
            </button>
          </div>
        </aside>

        <section className="dt-main" aria-label="回归训练结果">
          <article className="dt-panel">
            <div className="panel-title split">
              <span>
                <CheckCircle2 size={17} aria-hidden="true" />
                <h2>训练结果</h2>
              </span>
              <small>{trainResult?.version_id ? `version ${trainResult.version_id.slice(0, 12)}` : "等待训练"}</small>
            </div>
            <div className="metric-grid">
              <div className="mini-metric"><span>R²</span><strong>{formatNumber(trainResult?.r2)}</strong></div>
              <div className="mini-metric"><span>MAE</span><strong>{formatNumber(trainResult?.mae)}</strong></div>
              <div className="mini-metric"><span>RMSE</span><strong>{formatNumber(trainResult?.rmse)}</strong></div>
              <div className="mini-metric"><span>训练轮次</span><strong>{trainResult?.train_losses?.length ?? "-"}</strong></div>
            </div>
          </article>

          <article className="dt-panel">
            <div className="panel-title split">
              <span>
                <LineChart size={17} aria-hidden="true" />
                <h2>损失曲线</h2>
              </span>
              <small>MSE</small>
            </div>
            <LossChart train={trainResult?.train_losses} val={trainResult?.val_losses} />
          </article>

          <article className="dt-panel">
            <div className="panel-title split">
              <span>
                <AlertTriangle size={17} aria-hidden="true" />
                <h2>训练风险</h2>
              </span>
              <small>数据质量</small>
            </div>
            <div className="cluster-feature-notes">
              {featureCols.map((column) => {
                const columnProfile = profile.column_profiles.find((item) => item.name === column);
                return <span key={column}>{column}: 缺失 {columnProfile?.missing_count ?? 0} · 异常 {columnProfile?.outlier_count ?? 0} · 唯一值 {columnProfile?.unique_count ?? "-"}</span>;
              })}
            </div>
          </article>
        </section>
      </div>

      <div className="dt-bottom-grid regression-bottom-grid">
        <article className="dt-panel">
          <div className="panel-title split">
            <span>
              <RefreshCcw size={17} aria-hidden="true" />
              <h2>模型版本</h2>
            </span>
            <button className="tiny-button" type="button" onClick={loadRegressionState} disabled={versionsLoading}>刷新</button>
          </div>
          <div className="version-list tall">
            {versions.length ? versions.map((version) => (
              <div className="version-row" key={version.version_id}>
                <div>
                  <strong>{version.version_id}</strong>
                  <span>{version.dataset_name || "unknown"} · {version.target || "-"} · {version.created_at || "-"}</span>
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
            )) : <div className="empty-list">暂无回归版本。</div>}
          </div>
        </article>

        <article className="dt-panel">
          <div className="panel-title split">
            <span>
              <Send size={17} aria-hidden="true" />
              <h2>单条预测</h2>
            </span>
            <small>{trainResult?.version_id ? targetCol : "等待模型"}</small>
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
          <button className="button primary full" type="button" onClick={handlePredict} disabled={!featureCols.length || loading || !trainResult?.version_id}>
            <Brain size={15} aria-hidden="true" />
            执行回归预测
          </button>
          {predictResult ? (
            <div className="prediction-result">
              <strong>预测 {targetCol}：{formatNumber(predictResult.result)}</strong>
              <span>模型版本：{(selectedVersion || trainResult?.version_id || "当前激活版本").slice(0, 24)}</span>
            </div>
          ) : null}
        </article>
      </div>

      <article className="dt-panel regression-batch-panel">
        <div className="panel-title split">
          <span>
            <Upload size={17} aria-hidden="true" />
            <h2>批量预测</h2>
          </span>
          <small>CSV · 最多 10000 行</small>
        </div>
        <div className="batch-toolbar">
          <label className="button ghost file-button">
            <Upload size={15} aria-hidden="true" />
            上传批量 CSV
            <input type="file" accept=".csv" onChange={handleBatchFile} />
          </label>
          <button className="button primary" type="button" disabled={!batchRows.length || batchLoading || !trainResult?.version_id} onClick={handleBatchPredict}>
            {batchLoading ? "预测中..." : "执行批量预测"}
          </button>
          <button className="button ghost" type="button" disabled={!batchRows.some((row) => row.prediction !== undefined)} onClick={downloadBatchResult}>
            <Download size={15} aria-hidden="true" />
            下载结果
          </button>
        </div>
        {batchError ? <div className="inline-error">{batchError}</div> : null}
        {batchRows.length ? (
          <div className="table-wrap compact-table">
            <table className="data-table">
              <thead>
                <tr>
                  {featureCols.slice(0, 8).map((column) => <th key={column}>{column}</th>)}
                  <th>预测_{targetCol}</th>
                </tr>
              </thead>
              <tbody>
                {batchRows.slice(0, 12).map((row, index) => (
                  <tr key={index}>
                    {featureCols.slice(0, 8).map((column) => <td key={column}>{row.source[column]}</td>)}
                    <td>{row.prediction === undefined ? "-" : formatNumber(row.prediction)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : <div className="empty-list">上传包含当前特征列的 CSV 后可批量预测。</div>}
      </article>
    </section>
  );
}
