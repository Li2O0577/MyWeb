import { ChangeEvent, useEffect, useMemo, useState } from "react";
import {
  AlertTriangle,
  Brain,
  CheckCircle2,
  Download,
  Layers3,
  LineChart,
  Play,
  Plus,
  RefreshCcw,
  RotateCcw,
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

type TaskType = "regression" | "classification";
type DeviceType = "cpu" | "cuda";
type DiyTab = "train" | "result" | "predict" | "batch";

interface LayerConfig {
  neurons: number;
  activation: string;
  bn: boolean;
  dropout: number;
}

interface DiyMlpResult extends ApiJson {
  version_id?: string;
  r2?: number;
  mae?: number;
  rmse?: number;
  acc?: number;
  n_classes?: number;
  label_names?: string[];
  reverse_label_map?: Record<string, string>;
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
  prediction?: string | number;
  confidence?: number;
}

const ACTIVATIONS = ["ReLU", "LeakyReLU", "GELU", "Tanh", "Sigmoid", "ELU", "SELU", "无激活"];
const LEARNING_RATES = [0.01, 0.005, 0.001, 0.0005, 0.0001];
const OPTIMIZERS = ["Adam", "AdamW", "SGD", "RMSprop"];
const BATCH_SIZES = [4, 8, 16, 32, 64, 128];

function formatNumber(value: unknown, digits = 4) {
  const n = Number(value);
  return Number.isFinite(n) ? n.toFixed(digits) : "-";
}

function valueToText(value: unknown) {
  if (value === null || value === undefined) return "";
  return String(value);
}

function defaultRegressionTarget(profile: DataProfile) {
  return profile.numeric_cols[profile.numeric_cols.length - 1] ?? "";
}

function defaultClassificationTarget(profile: DataProfile) {
  return profile.categorical_cols[0] ?? profile.columns[profile.columns.length - 1] ?? "";
}

function defaultTarget(profile: DataProfile, task: TaskType) {
  return task === "classification" ? defaultClassificationTarget(profile) : defaultRegressionTarget(profile);
}

function defaultFeatures(profile: DataProfile, target: string) {
  return profile.numeric_cols.filter((column) => column !== target).slice(0, Math.min(8, Math.max(1, profile.numeric_cols.length - 1)));
}

function defaultLayers(featureCount: number): LayerConfig[] {
  return [{ neurons: Math.max(8, featureCount * 2), activation: "ReLU", bn: false, dropout: 0 }];
}

function cleanRowsEstimate(profile: DataProfile, columns: string[]) {
  if (!columns.length) return 0;
  const maxMissing = Math.max(...columns.map((column) => profile.missing_counts[column] ?? 0), 0);
  return Math.max(0, profile.n_rows - maxMissing);
}

function sanitizeLayers(value: unknown, featureCount: number): LayerConfig[] {
  if (!Array.isArray(value)) return defaultLayers(featureCount);
  const layers = value
    .map((item) => {
      if (!item || typeof item !== "object") return null;
      const raw = item as Record<string, unknown>;
      const neurons = Math.max(1, Math.min(2048, Math.round(Number(raw.neurons))));
      if (!Number.isFinite(neurons)) return null;
      return {
        neurons,
        activation: ACTIVATIONS.includes(String(raw.activation)) ? String(raw.activation) : "ReLU",
        bn: Boolean(raw.bn),
        dropout: Math.max(0, Math.min(0.8, Number(raw.dropout) || 0))
      };
    })
    .filter(Boolean) as LayerConfig[];
  return layers.length ? layers : defaultLayers(featureCount);
}

function compatibleDiyMlp(profile: DataProfile, payload: { features?: unknown; target?: unknown; params?: unknown } | null | undefined) {
  const features = payload?.features;
  const target = String(payload?.target ?? "");
  const params = (payload?.params ?? {}) as Record<string, unknown>;
  const task = params.task === "classification" ? "classification" : "regression";
  const targetOk = task === "classification" ? profile.columns.includes(target) : profile.numeric_cols.includes(target);
  return (
    Boolean(target && targetOk) &&
    Array.isArray(features) &&
    features.length > 0 &&
    features.every((column) => profile.numeric_cols.includes(String(column)) && String(column) !== target)
  );
}

function normalizeDiyMlpResult(payload: ApiJson | null): DiyMlpResult | null {
  if (!payload) return null;
  const metrics = (payload.metrics ?? {}) as Record<string, unknown>;
  const params = (payload.params ?? {}) as Record<string, unknown>;
  return {
    ...payload,
    r2: payload.r2 as number | undefined ?? metrics.r2 as number | undefined,
    mae: payload.mae as number | undefined ?? metrics.mae as number | undefined,
    rmse: payload.rmse as number | undefined ?? metrics.rmse as number | undefined,
    acc: payload.acc as number | undefined ?? metrics.acc as number | undefined,
    n_classes: payload.n_classes as number | undefined ?? params.n_classes as number | undefined
  } as DiyMlpResult;
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
  const headers = rows.length ? [...Object.keys(rows[0].source), `prediction_${target}`, "confidence"] : [`prediction_${target}`, "confidence"];
  const escape = (value: unknown) => {
    const text = valueToText(value);
    return /[",\n\r]/.test(text) ? `"${text.replace(/"/g, '""')}"` : text;
  };
  return [
    headers.map(escape).join(","),
    ...rows.map((row) => headers.map((header) => {
      if (header === `prediction_${target}`) return escape(row.prediction);
      if (header === "confidence") return escape(row.confidence);
      return escape(row.source[header]);
    }).join(","))
  ].join("\n");
}

function architectureSummary(featureCount: number, layers: LayerConfig[], task: TaskType, nClasses: number, cleanRows: number) {
  const outputDim = task === "regression" || nClasses === 2 ? 1 : Math.max(1, nClasses);
  const dims = [featureCount, ...layers.map((layer) => layer.neurons), outputDim];
  const totalParams = dims.slice(0, -1).reduce((sum, dim, index) => {
    const next = dims[index + 1];
    const bnParams = index < layers.length && layers[index].bn ? next * 2 : 0;
    return sum + dim * next + next + bnParams;
  }, 0);
  const parts = [
    `Input(${featureCount})`,
    ...layers.map((layer) => {
      const extras = [layer.activation];
      if (layer.bn) extras.push("BN");
      if (layer.dropout > 0) extras.push(`Drop(${layer.dropout.toFixed(2)})`);
      return `${layer.neurons}[${extras.join(" / ")}]`;
    }),
    `Output(${outputDim})`
  ];
  const warnings: string[] = [];
  if (cleanRows && totalParams > cleanRows * 2) warnings.push(`严重过拟合风险：参数量 ${totalParams.toLocaleString()} 已超过样本数的 2 倍。`);
  else if (cleanRows && totalParams > cleanRows * 0.5) warnings.push(`过拟合风险：参数量 ${totalParams.toLocaleString()} 偏高，建议减少层宽或增加 Dropout。`);
  if (totalParams < Math.max(8, featureCount)) warnings.push(`欠拟合风险：参数量 ${totalParams.toLocaleString()} 过少。`);
  const vanishing = layers.filter((layer) => ["Sigmoid", "Tanh"].includes(layer.activation)).length;
  if (vanishing >= 3) warnings.push(`${vanishing} 层使用 Sigmoid/Tanh，深层网络可能出现梯度消失。`);
  if (layers.every((layer) => layer.activation === "无激活")) warnings.push("所有隐藏层均无激活函数，网络会退化为线性模型。");
  return { outputDim, totalParams, parts, warnings };
}

function LossChart({ train, val, task }: { train?: number[]; val?: number[]; task: TaskType }) {
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
        <text x={padding} y={22}>{task === "classification" ? "classification" : "regression"} loss</text>
        <text x={width - 122} y={22}>epochs {count}</text>
      </svg>
      <div className="chart-legend">
        <span><i className="train-dot" />训练损失</span>
        <span><i className="val-dot" />验证损失</span>
      </div>
    </div>
  );
}

function ProbabilityBars({ result }: { result: ApiJson | null }) {
  const probs = (result?.all_probs ?? []) as number[];
  const labels = (result?.label_names ?? []) as string[];
  if (!probs.length) return null;
  return (
    <div className="prob-list">
      {probs.map((prob, index) => (
        <div className="prob-row" key={labels[index] ?? index}>
          <span>{labels[index] ?? index}</span>
          <div><i style={{ width: `${Math.max(2, prob * 100)}%` }} /></div>
          <strong>{(prob * 100).toFixed(1)}%</strong>
        </div>
      ))}
    </div>
  );
}

export default function DiyMlpWorkspace({ profile }: Props) {
  const [taskType, setTaskType] = useState<TaskType>("regression");
  const [targetCol, setTargetCol] = useState(defaultTarget(profile, "regression"));
  const [featureCols, setFeatureCols] = useState<string[]>(defaultFeatures(profile, targetCol));
  const [layers, setLayers] = useState<LayerConfig[]>(defaultLayers(featureCols.length));
  const [learningRate, setLearningRate] = useState(0.005);
  const [optimizer, setOptimizer] = useState("Adam");
  const [epochs, setEpochs] = useState(100);
  const [batchSize, setBatchSize] = useState(16);
  const [valSplit, setValSplit] = useState(0.2);
  const [patience, setPatience] = useState(15);
  const [device, setDevice] = useState<DeviceType>("cpu");
  const [trainResult, setTrainResult] = useState<DiyMlpResult | null>(null);
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
  const [activeTab, setActiveTab] = useState<DiyTab>("train");

  const targetProfile = profile.column_profiles.find((column) => column.name === targetCol);
  const nClasses = taskType === "classification" ? targetProfile?.unique_count ?? 0 : 0;
  const cleanRows = cleanRowsEstimate(profile, [...featureCols, targetCol].filter(Boolean));
  const architecture = architectureSummary(featureCols.length, layers, taskType, nClasses, cleanRows);
  const constantFeatures = useMemo(
    () => featureCols.filter((column) => {
      const columnProfile = profile.column_profiles.find((item) => item.name === column);
      return (columnProfile?.unique_count ?? 2) <= 1;
    }),
    [featureCols, profile.column_profiles]
  );
  const classCounts = useMemo(() => {
    if (taskType !== "classification") return new Map<string, number>();
    const counts = new Map<string, number>();
    for (const row of profile.preview) {
      const value = valueToText(row[targetCol]);
      if (value) counts.set(value, (counts.get(value) ?? 0) + 1);
    }
    return counts;
  }, [profile.preview, targetCol, taskType]);
  const tooManyClasses = taskType === "classification" && (nClasses > 50 || (cleanRows > 0 && nClasses > cleanRows * 0.5));
  const targetOk = taskType === "classification" ? Boolean(targetCol && nClasses >= 2 && !tooManyClasses) : profile.numeric_cols.includes(targetCol);
  const canTrain = Boolean(profile.session_id && targetOk && featureCols.length && cleanRows >= 10 && !constantFeatures.length);

  useEffect(() => {
    const nextTarget = defaultTarget(profile, taskType);
    setTargetCol(nextTarget);
    setFeatureCols(defaultFeatures(profile, nextTarget));
    setLayers(defaultLayers(defaultFeatures(profile, nextTarget).length));
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

  const loadDiyMlpState = async () => {
    setVersionsLoading(true);
    setError("");
    try {
      const [versionPayload, statusPayload] = await Promise.all([
        fetchModelVersions("diy_mlp"),
        fetchModelStatus("diy_mlp")
      ]);
      setVersions(versionPayload.versions ?? []);
      setActiveVersion(versionPayload.active);
      const activeMeta = (versionPayload.versions ?? []).find((version) => version.version_id === versionPayload.active);
      const activeCompatible = compatibleDiyMlp(profile, activeMeta);
      setSelectedVersion(activeCompatible ? versionPayload.active ?? "" : "");
      const status = normalizeDiyMlpResult(statusPayload);
      if (status?.has_model && compatibleDiyMlp(profile, status)) {
        const params = (status.params ?? {}) as Record<string, unknown>;
        const statusTask = params.task === "classification" ? "classification" : "regression";
        setTaskType(statusTask);
        setTrainResult(status);
        if (status.target) setTargetCol(status.target);
        if (status.features?.length) setFeatureCols(status.features);
        setLayers(sanitizeLayers(params.layers, status.features?.length ?? featureCols.length));
        if (typeof params.learning_rate === "number") setLearningRate(params.learning_rate);
        if (typeof params.optimizer === "string") setOptimizer(params.optimizer);
        if (typeof params.epochs === "number") setEpochs(params.epochs);
        if (typeof params.batch_size === "number") setBatchSize(params.batch_size);
        if (typeof params.val_split === "number") setValSplit(params.val_split);
        if (typeof params.patience === "number") setPatience(params.patience);
      } else {
        setTrainResult(null);
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : "读取自定义 MLP 状态失败");
    } finally {
      setVersionsLoading(false);
    }
  };

  useEffect(() => {
    void loadDiyMlpState();
  }, [profile.session_id]);

  const switchTask = (nextTask: TaskType) => {
    const nextTarget = defaultTarget(profile, nextTask);
    const nextFeatures = defaultFeatures(profile, nextTarget);
    setTaskType(nextTask);
    setTargetCol(nextTarget);
    setFeatureCols(nextFeatures);
    setLayers(defaultLayers(nextFeatures.length));
    setTrainResult(null);
    setPredictResult(null);
    setBatchRows([]);
  };

  const availableTargets = taskType === "classification" ? profile.columns : profile.numeric_cols;
  const availableFeatures = profile.numeric_cols.filter((column) => column !== targetCol);

  const toggleFeature = (column: string) => {
    setFeatureCols((current) => {
      const next = current.includes(column) ? current.filter((item) => item !== column) : [...current, column];
      if (!current.length || layers.length === 1 && layers[0].neurons === Math.max(8, current.length * 2)) {
        setLayers(defaultLayers(next.length));
      }
      return next;
    });
  };

  const updateLayer = (index: number, patch: Partial<LayerConfig>) => {
    setLayers((current) => current.map((layer, i) => (i === index ? { ...layer, ...patch } : layer)));
  };

  const addLayer = () => {
    setLayers((current) => {
      const previous = current[current.length - 1]?.neurons ?? Math.max(8, featureCols.length * 2);
      return [...current, { neurons: Math.max(2, Math.floor(previous / 2)), activation: "ReLU", bn: false, dropout: 0 }];
    });
  };

  const removeLayer = (index: number) => {
    setLayers((current) => current.length > 1 ? current.filter((_, i) => i !== index) : current);
  };

  const handleTrain = async () => {
    if (!canTrain) return;
    setLoading(true);
    setError("");
    setPredictResult(null);
    setBatchRows([]);
    try {
      const payload = await trainModel("diy_mlp", {
        session_id: profile.session_id,
        target_col: targetCol,
        feature_cols: featureCols,
        layers,
        task_type: taskType,
        n_classes: taskType === "classification" ? nClasses : 0,
        learning_rate: learningRate,
        optimizer,
        epochs,
        batch_size: batchSize,
        val_split: valSplit,
        patience,
        device
      });
      const normalized = normalizeDiyMlpResult(payload);
      setTrainResult(normalized);
      setActiveTab("result");
      if (normalized?.version_id) {
        setSelectedVersion(normalized.version_id);
        setActiveVersion(normalized.version_id);
      }
      await loadDiyMlpState();
    } catch (err) {
      setError(err instanceof Error ? err.message : "自定义 MLP 训练失败");
    } finally {
      setLoading(false);
    }
  };

  const activateVersion = async (versionId: string) => {
    setError("");
    try {
      await activateModelVersion("diy_mlp", versionId);
      const detail = await fetchModelVersionDetail("diy_mlp", versionId);
      const normalized = normalizeDiyMlpResult({ ...detail, has_model: true });
      if (compatibleDiyMlp(profile, normalized)) {
        const params = (normalized?.params ?? {}) as Record<string, unknown>;
        const nextTask = params.task === "classification" ? "classification" : "regression";
        setTaskType(nextTask);
        setTrainResult(normalized);
        setSelectedVersion(versionId);
        if (normalized?.target) setTargetCol(normalized.target);
        if (normalized?.features?.length) setFeatureCols(normalized.features);
        setLayers(sanitizeLayers(params.layers, normalized?.features?.length ?? featureCols.length));
      } else {
        setTrainResult(null);
        setSelectedVersion("");
      }
      setPredictResult(null);
      setBatchRows([]);
      await loadDiyMlpState();
    } catch (err) {
      setError(err instanceof Error ? err.message : "切换 MLP 版本失败");
    }
  };

  const removeVersion = async (versionId: string) => {
    setError("");
    try {
      await deleteModelVersion("diy_mlp", versionId);
      if (selectedVersion === versionId) setSelectedVersion("");
      await loadDiyMlpState();
    } catch (err) {
      setError(err instanceof Error ? err.message : "删除版本失败");
    }
  };

  const clearActive = async () => {
    setError("");
    try {
      await clearModel("diy_mlp");
      setTrainResult(null);
      setPredictResult(null);
      setBatchRows([]);
      await loadDiyMlpState();
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
      const result = await predictModel("diy_mlp", {
        features,
        device,
        version_id: selectedVersion || undefined
      });
      setPredictResult(result);
    } catch (err) {
      setError(err instanceof Error ? err.message : "MLP 预测失败");
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
      setBatchRows(parsed.map((row) => ({ source: row })));
    } catch (err) {
      setBatchError(err instanceof Error ? err.message : "CSV 读取失败");
    }
  };

  const labelFor = (index: unknown) => {
    const key = String(index);
    return trainResult?.reverse_label_map?.[key] ?? key;
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
      const result = await batchPredictModel("diy_mlp", {
        rows,
        device,
        version_id: selectedVersion || undefined
      });
      if (taskType === "regression") {
        const predictions = (result.predictions ?? []) as number[];
        setBatchRows((current) => current.map((row, index) => ({ ...row, prediction: Number(predictions[index]) })));
      } else {
        const predIndices = (result.pred_indices ?? []) as Array<string | number>;
        const confidences = (result.confidences ?? []) as number[];
        setBatchRows((current) => current.map((row, index) => ({
          ...row,
          prediction: labelFor(predIndices[index]),
          confidence: Number(confidences[index])
        })));
      }
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
    anchor.download = "diy_mlp_predictions.csv";
    anchor.click();
    URL.revokeObjectURL(url);
  };

  return (
    <section className="diy-mlp-workspace" aria-label="自定义 MLP 工作台">
      <div className="dt-toolbar">
        <div>
          <span>当前数据</span>
          <strong>{profile.n_rows.toLocaleString()} 行 · {profile.n_cols} 列</strong>
        </div>
        <div>
          <span>数值特征</span>
          <strong>{profile.numeric_cols.length} 列</strong>
        </div>
        <div>
          <span>网络参数</span>
          <strong>{architecture.totalParams.toLocaleString()}</strong>
        </div>
      </div>

      {error ? <div className="inline-error">{error}</div> : null}
      {activeVersion && !trainResult ? (
        <div className="inline-warning">当前激活的 DIY MLP 模型与当前数据字段不兼容。版本仍保留在列表中，当前数据请重新训练。</div>
      ) : null}
      {cleanRows < 10 && featureCols.length ? <div className="inline-warning">MLP 训练至少需要 10 行无缺失数据。</div> : null}
      {constantFeatures.length ? <div className="inline-warning">常量特征会影响训练：{constantFeatures.join("、")}。建议先到数据处理页移除。</div> : null}
      {tooManyClasses ? <div className="inline-warning">目标列类别数过多，可能是 ID 或连续值；请改选真正的分类标签。</div> : null}

      <div className="diy-tabbar" role="tablist" aria-label="自定义 MLP 功能分区">
        <button className={activeTab === "train" ? "diy-tab active" : "diy-tab"} type="button" role="tab" aria-selected={activeTab === "train"} onClick={() => setActiveTab("train")}>训练配置</button>
        <button className={activeTab === "result" ? "diy-tab active" : "diy-tab"} type="button" role="tab" aria-selected={activeTab === "result"} onClick={() => setActiveTab("result")}>结果分析</button>
        <button className={activeTab === "predict" ? "diy-tab active" : "diy-tab"} type="button" role="tab" aria-selected={activeTab === "predict"} onClick={() => setActiveTab("predict")}>模型预测</button>
        <button className={activeTab === "batch" ? "diy-tab active" : "diy-tab"} type="button" role="tab" aria-selected={activeTab === "batch"} onClick={() => setActiveTab("batch")}>批量预测</button>
      </div>

      {activeTab === "train" || activeTab === "result" ? (
      <div className="dt-layout diy-layout single-pane">
        {activeTab === "train" ? (
        <aside className="dt-config diy-config" aria-label="自定义 MLP 训练配置">
          <div className="panel-title split">
            <span>
              <SlidersHorizontal size={17} aria-hidden="true" />
              <h2>训练配置</h2>
            </span>
            <small>{taskType === "classification" ? "分类" : "回归"}</small>
          </div>

          <div className="segmented">
            <button className={taskType === "regression" ? "selected" : ""} type="button" onClick={() => switchTask("regression")}>回归</button>
            <button className={taskType === "classification" ? "selected" : ""} type="button" onClick={() => switchTask("classification")}>分类</button>
          </div>

          <label className="form-field">
            <span>{taskType === "classification" ? "分类目标列" : "回归目标列"}</span>
            <select value={targetCol} onChange={(event) => {
              const nextTarget = event.target.value;
              const nextFeatures = defaultFeatures(profile, nextTarget);
              setTargetCol(nextTarget);
              setFeatureCols(nextFeatures);
              setLayers(defaultLayers(nextFeatures.length));
              setTrainResult(null);
              setPredictResult(null);
            }}>
              {availableTargets.map((column) => <option key={column}>{column}</option>)}
            </select>
          </label>

          <div className="form-field">
            <span>数值特征列</span>
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

          <div className="layer-list">
            <div className="panel-title split compact-title">
              <span>
                <Layers3 size={16} aria-hidden="true" />
                <h3>隐藏层</h3>
              </span>
              <small>{layers.length} 层</small>
            </div>
            {layers.map((layer, index) => (
              <div className="layer-row" key={index}>
                <label className="form-field">
                  <span>神经元</span>
                  <input min={1} max={2048} type="number" value={layer.neurons} onChange={(event) => updateLayer(index, { neurons: Number(event.target.value) })} />
                </label>
                <label className="form-field">
                  <span>激活</span>
                  <select value={layer.activation} onChange={(event) => updateLayer(index, { activation: event.target.value })}>
                    {ACTIVATIONS.map((name) => <option key={name}>{name}</option>)}
                  </select>
                </label>
                <label className="form-field">
                  <span>Dropout {layer.dropout.toFixed(2)}</span>
                  <input min={0} max={0.8} step={0.05} type="range" value={layer.dropout} onChange={(event) => updateLayer(index, { dropout: Number(event.target.value) })} />
                </label>
                <label className="check-field">
                  <input type="checkbox" checked={layer.bn} onChange={(event) => updateLayer(index, { bn: event.target.checked })} />
                  <span>BN</span>
                </label>
                <button className="icon-button danger-icon" type="button" title="删除层" aria-label={`删除第 ${index + 1} 层`} onClick={() => removeLayer(index)} disabled={layers.length <= 1}>
                  <Trash2 size={14} aria-hidden="true" />
                </button>
              </div>
            ))}
            <div className="prediction-actions">
              <button className="button ghost" type="button" onClick={addLayer}>
                <Plus size={15} aria-hidden="true" />
                添加隐藏层
              </button>
              <button className="button ghost" type="button" onClick={() => setLayers(defaultLayers(featureCols.length))}>
                <RotateCcw size={15} aria-hidden="true" />
                重置结构
              </button>
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
              <span>优化器</span>
              <select value={optimizer} onChange={(event) => setOptimizer(event.target.value)}>
                {OPTIMIZERS.map((name) => <option key={name}>{name}</option>)}
              </select>
            </label>
            <label className="form-field">
              <span>Batch</span>
              <select value={batchSize} onChange={(event) => setBatchSize(Number(event.target.value))}>
                {BATCH_SIZES.map((size) => <option key={size} value={size}>{size}</option>)}
              </select>
            </label>
          </div>

          <div className="operation-columns three">
            <label className="form-field">
              <span>最大轮数：{epochs}</span>
              <input min={20} max={500} step={20} type="range" value={epochs} onChange={(event) => setEpochs(Number(event.target.value))} />
            </label>
            <label className="form-field">
              <span>验证集：{Math.round(valSplit * 100)}%</span>
              <input min={0.1} max={0.4} step={0.05} type="range" value={valSplit} onChange={(event) => setValSplit(Number(event.target.value))} />
            </label>
            <label className="form-field">
              <span>早停：{patience}</span>
              <input min={5} max={50} step={5} type="range" value={patience} onChange={(event) => setPatience(Number(event.target.value))} />
            </label>
          </div>

          <div className="segmented">
            <button className={device === "cpu" ? "selected" : ""} type="button" onClick={() => setDevice("cpu")}>CPU</button>
            <button className={device === "cuda" ? "selected" : ""} type="button" onClick={() => setDevice("cuda")}>CUDA</button>
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
        ) : null}

        {activeTab === "result" ? (
        <section className="dt-main" aria-label="自定义 MLP 训练结果">
          <article className="dt-panel">
            <div className="panel-title split">
              <span>
                <CheckCircle2 size={17} aria-hidden="true" />
                <h2>训练结果</h2>
              </span>
              <small>{trainResult?.version_id ? `version ${trainResult.version_id.slice(0, 12)}` : "等待训练"}</small>
            </div>
            <div className="metric-grid">
              {taskType === "classification" ? (
                <>
                  <div className="mini-metric"><span>准确率</span><strong>{formatNumber(trainResult?.acc)}</strong></div>
                  <div className="mini-metric"><span>类别数</span><strong>{trainResult?.n_classes ?? nClasses ?? "-"}</strong></div>
                </>
              ) : (
                <>
                  <div className="mini-metric"><span>R²</span><strong>{formatNumber(trainResult?.r2)}</strong></div>
                  <div className="mini-metric"><span>MAE</span><strong>{formatNumber(trainResult?.mae)}</strong></div>
                  <div className="mini-metric"><span>RMSE</span><strong>{formatNumber(trainResult?.rmse)}</strong></div>
                </>
              )}
              <div className="mini-metric"><span>训练轮次</span><strong>{trainResult?.train_losses?.length ?? "-"}</strong></div>
              <div className="mini-metric"><span>输出维度</span><strong>{architecture.outputDim}</strong></div>
            </div>
          </article>

          <article className="dt-panel diy-architecture">
            <div className="panel-title split">
              <span>
                <Layers3 size={17} aria-hidden="true" />
                <h2>网络结构摘要</h2>
              </span>
              <small>{architecture.totalParams.toLocaleString()} params</small>
            </div>
            <div className="architecture-flow">
              {architecture.parts.map((part, index) => <span key={`${part}-${index}`}>{part}</span>)}
            </div>
            <div className="diy-param-summary">
              <span>输入 {featureCols.length}</span>
              <span>隐藏层 {layers.length}</span>
              <span>输出 {architecture.outputDim}</span>
            </div>
          </article>

          <article className="dt-panel">
            <div className="panel-title split">
              <span>
                <LineChart size={17} aria-hidden="true" />
                <h2>损失曲线</h2>
              </span>
              <small>{taskType === "classification" ? "classification" : "regression"}</small>
            </div>
            <LossChart train={trainResult?.train_losses} val={trainResult?.val_losses} task={taskType} />
          </article>

          <article className="dt-panel">
            <div className="panel-title split">
              <span>
                <AlertTriangle size={17} aria-hidden="true" />
                <h2>训练风险</h2>
              </span>
              <small>结构与数据</small>
            </div>
            <div className="risk-list">
              {architecture.warnings.length ? architecture.warnings.map((warning) => <span key={warning}>{warning}</span>) : <span>当前结构未触发明显风险提示。</span>}
              {taskType === "classification" && classCounts.size ? <span>预览样本类别分布：{Array.from(classCounts.entries()).slice(0, 5).map(([key, count]) => `${key} ${count}`).join(" · ")}</span> : null}
              {featureCols.map((column) => {
                const columnProfile = profile.column_profiles.find((item) => item.name === column);
                return <span key={column}>{column}: 缺失 {columnProfile?.missing_count ?? 0} · 异常 {columnProfile?.outlier_count ?? 0} · 唯一值 {columnProfile?.unique_count ?? "-"}</span>;
              })}
            </div>
          </article>
        </section>
        ) : null}
      </div>
      ) : null}

      {activeTab === "predict" ? (
      <div className="dt-bottom-grid diy-bottom-grid">
        <article className="dt-panel">
          <div className="panel-title split">
            <span>
              <RefreshCcw size={17} aria-hidden="true" />
              <h2>模型版本</h2>
            </span>
            <button className="tiny-button" type="button" onClick={loadDiyMlpState} disabled={versionsLoading}>刷新</button>
          </div>
          <div className="version-list tall">
            {versions.length ? versions.map((version) => {
              const params = (version.params ?? {}) as Record<string, unknown>;
              return (
                <div className="version-row" key={version.version_id}>
                  <div>
                    <strong>{version.version_id}</strong>
                    <span>{params.task === "classification" ? "分类" : "回归"} · {version.target || "-"} · {version.created_at || "-"}</span>
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
              );
            }) : <div className="empty-list">暂无 DIY MLP 版本。</div>}
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
            执行 MLP 预测
          </button>
          {predictResult ? (
            <div className="prediction-result">
              {taskType === "classification" ? (
                <>
                  <strong>预测类别：{String(predictResult.pred_class)}</strong>
                  {typeof predictResult.prob === "number" ? <span>置信度：{(predictResult.prob * 100).toFixed(1)}%</span> : null}
                  <ProbabilityBars result={predictResult} />
                </>
              ) : (
                <>
                  <strong>预测 {targetCol}：{formatNumber(predictResult.result)}</strong>
                  <span>模型版本：{(selectedVersion || trainResult?.version_id || "当前激活版本").slice(0, 24)}</span>
                </>
              )}
            </div>
          ) : null}
        </article>
      </div>
      ) : null}

      {activeTab === "batch" ? (
      <article className="dt-panel diy-batch-panel">
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
                  {taskType === "classification" ? <th>置信度</th> : null}
                </tr>
              </thead>
              <tbody>
                {batchRows.slice(0, 12).map((row, index) => (
                  <tr key={index}>
                    {featureCols.slice(0, 8).map((column) => <td key={column}>{row.source[column]}</td>)}
                    <td>{row.prediction ?? "-"}</td>
                    {taskType === "classification" ? <td>{row.confidence === undefined ? "-" : formatNumber(row.confidence)}</td> : null}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : <div className="empty-list">上传包含当前特征列的 CSV 后可批量预测。</div>}
      </article>
      ) : null}
    </section>
  );
}
