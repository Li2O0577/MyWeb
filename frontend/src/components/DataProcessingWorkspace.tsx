import { useEffect, useMemo, useState } from "react";
import {
  AlertTriangle,
  Binary,
  Braces,
  Calculator,
  BarChart3,
  Columns3,
  Download,
  Eraser,
  History,
  ListChecks,
  PlayCircle,
  Redo2,
  Rows3,
  Save,
  Sigma,
  Sparkles,
  SplitSquareHorizontal,
  Trash2,
  Type,
  Undo2,
  Wand2
} from "lucide-react";
import {
  applyProcessingPipeline,
  fetchProcessingHistory,
  processData,
  redoProcessing,
  saveProcessingPipeline,
  undoProcessing
} from "../services/api";
import type { DataProfile, DataUploadResponse, ProcessingHistory, ProcessingPipeline } from "../types";

type Operation = Record<string, unknown>;

interface DataProcessingWorkspaceProps {
  profile: DataProfile;
  onProfileChange: (profile: DataUploadResponse) => void;
}

function valueToText(value: unknown) {
  if (value === null || value === undefined) return "";
  if (typeof value === "number" && Number.isNaN(value)) return "";
  if (typeof value === "object") return JSON.stringify(value);
  return String(value);
}

function toggleValue(values: string[], value: string) {
  return values.includes(value) ? values.filter((item) => item !== value) : [...values, value];
}

function ColumnPicker({
  columns,
  selected,
  onChange,
  emptyText = "暂无可选字段"
}: {
  columns: string[];
  selected: string[];
  onChange: (values: string[]) => void;
  emptyText?: string;
}) {
  if (!columns.length) return <div className="empty-list">{emptyText}</div>;
  return (
    <div className="column-select compact">
      {columns.map((column) => (
        <button
          className={selected.includes(column) ? "column-choice selected" : "column-choice"}
          key={column}
          type="button"
          onClick={() => onChange(toggleValue(selected, column))}
        >
          {column}
        </button>
      ))}
    </div>
  );
}

function firstColumn(columns: string[]) {
  return columns[0] ?? "";
}

function formatStat(value: unknown) {
  const n = Number(value);
  if (!Number.isFinite(n)) return "-";
  if (Math.abs(n) >= 1000 || Math.abs(n) < 0.001 && n !== 0) return n.toExponential(3);
  return n.toLocaleString(undefined, { maximumFractionDigits: 4 });
}

function formatTime(seconds: number) {
  if (!seconds) return "-";
  return new Date(seconds * 1000).toLocaleString(undefined, {
    month: "2-digit",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit"
  });
}

function operationLabel(operation: Operation) {
  const labels: Record<string, string> = {
    select_cols: "保留字段",
    drop_na: "删除缺失行",
    drop_duplicates: "删除重复行",
    drop_cols: "删除字段",
    drop_rows: "删除行",
    rename_col: "重命名字段",
    cast_type: "类型转换",
    fill_na: "缺失值填充",
    scale: "数值缩放",
    add_noise: "添加噪声",
    label_encode: "标签编码",
    one_hot_encode: "独热编码",
    custom_formula: "表达式计算",
    unary_calc: "一元计算",
    binary_calc: "二元计算",
    pca: "PCA 降维",
    winsorize_outliers: "异常值缩尾",
    replace_outliers: "异常值替换",
    drop_outliers: "删除异常行"
  };
  return labels[String(operation.op)] ?? String(operation.op ?? "处理步骤");
}

export default function DataProcessingWorkspace({ profile, onProfileChange }: DataProcessingWorkspaceProps) {
  const numericCols = profile.numeric_cols;
  const categoricalCols = profile.categorical_cols;
  const previewColumns = profile.columns.slice(0, 10);
  const previewRows = profile.preview.slice(0, 12);

  const [busy, setBusy] = useState(false);
  const [message, setMessage] = useState("");
  const [error, setError] = useState("");
  const [history, setHistory] = useState<ProcessingHistory | null>(profile.processing_history ?? null);
  const [historyBusy, setHistoryBusy] = useState(false);
  const [pipelineName, setPipelineName] = useState("当前处理流水线");
  const [dropColumns, setDropColumns] = useState<string[]>([]);
  const [keepColumns, setKeepColumns] = useState<string[]>(profile.columns);
  const [rowPosition, setRowPosition] = useState(0);

  const [renameColumn, setRenameColumn] = useState(firstColumn(profile.columns));
  const [newColumnName, setNewColumnName] = useState(firstColumn(profile.columns));
  const [castColumn, setCastColumn] = useState(firstColumn(profile.columns));
  const [castType, setCastType] = useState("float");
  const [statsColumn, setStatsColumn] = useState(firstColumn(numericCols));
  const [outlierColumns, setOutlierColumns] = useState<string[]>(
    numericCols.filter((column) => (profile.outliers[column]?.count ?? 0) > 0)
  );
  const [outlierCoefficient, setOutlierCoefficient] = useState(1.5);
  const [outlierReplaceMethod, setOutlierReplaceMethod] = useState<"median" | "mean">("median");

  const [fillColumns, setFillColumns] = useState<string[]>(numericCols.slice(0, 1));
  const [fillMethod, setFillMethod] = useState("mean");
  const [fillValue, setFillValue] = useState("");
  const [scaleColumns, setScaleColumns] = useState<string[]>(numericCols.slice(0, 2));
  const [scaleMethod, setScaleMethod] = useState("standard");
  const [noiseColumns, setNoiseColumns] = useState<string[]>(numericCols.slice(0, 1));
  const [noiseLevel, setNoiseLevel] = useState(0.1);

  const [encodeColumns, setEncodeColumns] = useState<string[]>(categoricalCols.slice(0, 1));
  const [encodeMethod, setEncodeMethod] = useState("label");

  const [unaryColumn, setUnaryColumn] = useState(firstColumn(numericCols));
  const [unaryOperator, setUnaryOperator] = useState("square");
  const [unaryNewColumn, setUnaryNewColumn] = useState("new_column");
  const [binaryLeft, setBinaryLeft] = useState(firstColumn(numericCols));
  const [binaryRightMode, setBinaryRightMode] = useState<"column" | "constant">("column");
  const [binaryRight, setBinaryRight] = useState(numericCols[1] ?? firstColumn(numericCols));
  const [binaryValue, setBinaryValue] = useState(0);
  const [binaryOperator, setBinaryOperator] = useState("+");
  const [binaryNewColumn, setBinaryNewColumn] = useState("new_column");
  const [formulaExpression, setFormulaExpression] = useState("");
  const [formulaNewColumn, setFormulaNewColumn] = useState("new_column");

  const [pcaColumns, setPcaColumns] = useState<string[]>(numericCols.slice(0, Math.min(4, numericCols.length)));
  const [pcaComponents, setPcaComponents] = useState(2);
  const [pcaDropOriginal, setPcaDropOriginal] = useState(false);

  const formulaHint = useMemo(() => numericCols.filter((column) => /^[A-Za-z_]\w*$/.test(column)).join(", "), [numericCols]);
  const pcaMax = Math.max(1, pcaColumns.length);
  const selectedStatsProfile =
    profile.column_profiles.find((column) => column.name === statsColumn && column.kind === "numeric") ??
    profile.column_profiles.find((column) => column.kind === "numeric");
  const outlierCandidateColumns = useMemo(
    () => numericCols.filter((column) => (profile.outliers[column]?.count ?? 0) > 0),
    [numericCols, profile.outliers]
  );
  const selectedOutlierCount = outlierColumns.reduce((sum, column) => sum + Number(profile.outliers[column]?.count ?? 0), 0);

  useEffect(() => {
    setOutlierColumns((current) => {
      const cleaned = current.filter((column) => outlierCandidateColumns.includes(column));
      return cleaned.length || !outlierCandidateColumns.length ? cleaned : outlierCandidateColumns;
    });
  }, [outlierCandidateColumns]);

  useEffect(() => {
    if (profile.processing_history) {
      setHistory(profile.processing_history);
      return;
    }
    const controller = new AbortController();
    fetchProcessingHistory(profile.session_id, controller.signal)
      .then((payload) => setHistory(payload))
      .catch(() => setHistory(null));
    return () => controller.abort();
  }, [profile.processing_history, profile.session_id]);

  const refreshHistory = async () => {
    try {
      const payload = await fetchProcessingHistory(profile.session_id);
      setHistory(payload);
    } catch {
      // History is supplementary; keep the data page usable even if it fails.
    }
  };

  const applyOperations = async (operations: Operation[], fallbackMessage: string) => {
    if (!operations.length) return;
    setBusy(true);
    setMessage("");
    setError("");
    try {
      const payload = await processData(profile.session_id, operations);
      onProfileChange(payload);
      if (payload.processing_history) setHistory(payload.processing_history);
      const applied = (payload as DataUploadResponse & { operations_applied?: string[] }).operations_applied;
      setMessage(applied?.length ? applied.join("；") : fallbackMessage);
      setDropColumns([]);
    } catch (err) {
      setError(err instanceof Error ? err.message : "数据处理失败");
    } finally {
      setBusy(false);
    }
  };

  const runHistoryAction = async (action: "undo" | "redo") => {
    setHistoryBusy(true);
    setMessage("");
    setError("");
    try {
      const payload = action === "undo" ? await undoProcessing(profile.session_id) : await redoProcessing(profile.session_id);
      onProfileChange(payload);
      if (payload.processing_history) setHistory(payload.processing_history);
      setMessage(action === "undo" ? "已撤销上一步处理。" : "已重做下一步处理。");
    } catch (err) {
      setError(err instanceof Error ? err.message : action === "undo" ? "撤销失败" : "重做失败");
    } finally {
      setHistoryBusy(false);
    }
  };

  const savePipeline = async () => {
    setHistoryBusy(true);
    setMessage("");
    setError("");
    try {
      const payload = await saveProcessingPipeline(profile.session_id, pipelineName);
      setHistory(payload.processing_history);
      setMessage("已保存当前数据处理流水线。");
    } catch (err) {
      setError(err instanceof Error ? err.message : "保存流水线失败");
    } finally {
      setHistoryBusy(false);
    }
  };

  const applyPipeline = async (pipeline: ProcessingPipeline) => {
    setHistoryBusy(true);
    setMessage("");
    setError("");
    try {
      const payload = await applyProcessingPipeline(profile.session_id, pipeline.pipeline_id);
      onProfileChange(payload);
      if (payload.processing_history) setHistory(payload.processing_history);
      setMessage(`已应用流水线：${pipeline.name}`);
    } catch (err) {
      setError(err instanceof Error ? err.message : "应用流水线失败");
    } finally {
      setHistoryBusy(false);
    }
  };

  const exportCsv = () => {
    const headers = profile.columns;
    const rows = profile.preview.map((row) => headers.map((column) => JSON.stringify(valueToText(row[column]))).join(","));
    const csv = [headers.join(","), ...rows].join("\n");
    const blob = new Blob([csv], { type: "text/csv;charset=utf-8" });
    const url = URL.createObjectURL(blob);
    const anchor = document.createElement("a");
    anchor.href = url;
    anchor.download = "processed_preview.csv";
    anchor.click();
    URL.revokeObjectURL(url);
  };

  return (
    <section className="processing-workspace" aria-label="数据处理工作台">
      <div className="processing-toolbar">
        <div>
          <span>当前工作数据</span>
          <strong>{profile.n_rows.toLocaleString()} 行 · {profile.n_cols} 列</strong>
        </div>
        <div className="toolbar-actions">
          <button className="button ghost" type="button" onClick={exportCsv}>
            <Download size={15} aria-hidden="true" />
            导出预览 CSV
          </button>
          <button
            className="button ghost"
            type="button"
            disabled={busy || historyBusy || !history?.can_undo}
            onClick={() => runHistoryAction("undo")}
          >
            <Undo2 size={15} aria-hidden="true" />
            撤销
          </button>
          <button
            className="button ghost"
            type="button"
            disabled={busy || historyBusy || !history?.can_redo}
            onClick={() => runHistoryAction("redo")}
          >
            <Redo2 size={15} aria-hidden="true" />
            重做
          </button>
          <button
            className="button ghost"
            type="button"
            disabled={busy}
            onClick={() => applyOperations([{ op: "drop_duplicates" }], "已删除重复行")}
          >
            <Eraser size={15} aria-hidden="true" />
            删除重复行
          </button>
          <button
            className="button danger"
            type="button"
            disabled={busy}
            onClick={() => applyOperations([{ op: "drop_na" }], "已删除缺失行")}
          >
            <Rows3 size={15} aria-hidden="true" />
            删除缺失行
          </button>
        </div>
      </div>

      {message ? <div className="inline-success">{message}</div> : null}
      {error ? <div className="inline-error">{error}</div> : null}

      <section className="operation-panel span-2 processing-history-panel" aria-label="数据处理历史">
        <div className="panel-title split">
          <span>
            <History size={17} aria-hidden="true" />
            <h2>操作历史与流水线</h2>
          </span>
          <small>{history ? `当前位置 ${history.current_index + 1} / ${history.history.length}` : "正在读取历史"}</small>
        </div>
        <div className="history-layout">
          <div className="history-list" aria-label="操作历史">
            {history?.history.length ? (
              history.history.map((entry, index) => (
                <div className={index === history.current_index ? "history-row current" : "history-row"} key={entry.state_id}>
                  <div>
                    <strong>{entry.label}</strong>
                    <span>{entry.n_rows.toLocaleString()} 行 · {entry.n_cols} 列 · {formatTime(entry.created_at)}</span>
                    {entry.operations?.length ? <small>{entry.operations.map(operationLabel).join(" / ")}</small> : null}
                  </div>
                  <span className="history-index">{index === history.current_index ? "当前" : `#${index + 1}`}</span>
                </div>
              ))
            ) : (
              <div className="empty-list">暂无处理历史。</div>
            )}
          </div>

          <div className="pipeline-panel" aria-label="保存的流水线">
            <div className="operation-columns two">
              <label className="form-field">
                <span>流水线名称</span>
                <input value={pipelineName} onChange={(event) => setPipelineName(event.target.value)} />
              </label>
              <button
                className="button primary align-end"
                type="button"
                disabled={busy || historyBusy || !history?.can_undo}
                onClick={savePipeline}
              >
                <Save size={15} aria-hidden="true" />
                保存流水线
              </button>
            </div>
            <div className="pipeline-list">
              {history?.pipelines?.length ? (
                history.pipelines.map((pipeline) => (
                  <div className="pipeline-row" key={pipeline.pipeline_id}>
                    <div>
                      <strong>{pipeline.name}</strong>
                      <span>{pipeline.step_count} 步 · {formatTime(pipeline.created_at)}</span>
                    </div>
                    <button className="button ghost" type="button" disabled={busy || historyBusy} onClick={() => applyPipeline(pipeline)}>
                      <PlayCircle size={15} aria-hidden="true" />
                      应用
                    </button>
                  </div>
                ))
              ) : (
                <div className="empty-list">保存后可在同一 session 内重复应用。</div>
              )}
            </div>
            <button className="button ghost" type="button" disabled={historyBusy} onClick={refreshHistory}>
              <History size={15} aria-hidden="true" />
              刷新历史
            </button>
          </div>
        </div>
      </section>

      <div className="operation-grid">
        <article className="operation-panel span-2">
          <div className="panel-title split">
            <span>
              <Columns3 size={17} aria-hidden="true" />
              <h2>行列筛选 / 删除 / 重命名</h2>
            </span>
            <small>写回后端 session</small>
          </div>
          <div className="operation-columns two">
            <div className="operation-block">
              <label className="form-field">
                <span>保留指定列</span>
                <ColumnPicker columns={profile.columns} selected={keepColumns} onChange={setKeepColumns} />
              </label>
              <button
                className="button primary"
                type="button"
                disabled={busy || keepColumns.length === 0 || keepColumns.length === profile.columns.length}
                onClick={() => applyOperations([{ op: "select_cols", cols: keepColumns }], `已保留 ${keepColumns.length} 个字段`)}
              >
                <ListChecks size={15} aria-hidden="true" />
                应用列筛选
              </button>
            </div>
            <div className="operation-block">
              <label className="form-field">
                <span>删除指定列</span>
                <ColumnPicker columns={profile.columns} selected={dropColumns} onChange={setDropColumns} />
              </label>
              <button
                className="button danger"
                type="button"
                disabled={busy || dropColumns.length === 0}
                onClick={() => applyOperations([{ op: "drop_cols", cols: dropColumns }], `已删除 ${dropColumns.length} 个字段`)}
              >
                <Trash2 size={15} aria-hidden="true" />
                删除选中列
              </button>
            </div>
          </div>
          <div className="operation-columns three">
            <label className="form-field">
              <span>重命名字段</span>
              <select value={renameColumn} onChange={(event) => {
                setRenameColumn(event.target.value);
                setNewColumnName(event.target.value);
              }}>
                {profile.columns.map((column) => <option key={column}>{column}</option>)}
              </select>
            </label>
            <label className="form-field">
              <span>新列名</span>
              <input value={newColumnName} onChange={(event) => setNewColumnName(event.target.value)} />
            </label>
            <button
              className="button primary align-end"
              type="button"
              disabled={busy || !renameColumn || !newColumnName || renameColumn === newColumnName}
              onClick={() => applyOperations([{ op: "rename_col", old: renameColumn, new: newColumnName }], "已重命名字段")}
            >
              <Type size={15} aria-hidden="true" />
              确认重命名
            </button>
          </div>
          <div className="operation-columns three">
            <label className="form-field">
              <span>删除行号</span>
              <input min={0} max={Math.max(0, profile.n_rows - 1)} type="number" value={rowPosition} onChange={(event) => setRowPosition(Number(event.target.value))} />
            </label>
            <div className="operation-note">行号按当前表格顺序从 0 开始，适合删除少量明显异常记录。</div>
            <button
              className="button danger align-end"
              type="button"
              disabled={busy || rowPosition < 0 || rowPosition >= profile.n_rows}
              onClick={() => applyOperations([{ op: "drop_rows", positions: [rowPosition] }], "已删除指定行")}
            >
              <Rows3 size={15} aria-hidden="true" />
              删除该行
            </button>
          </div>
        </article>

        <article className="operation-panel">
          <div className="panel-title">
            <span>
              <Type size={17} aria-hidden="true" />
              <h2>数据类型修改</h2>
            </span>
          </div>
          <label className="form-field">
            <span>选择列</span>
            <select value={castColumn} onChange={(event) => setCastColumn(event.target.value)}>
              {profile.columns.map((column) => <option key={column}>{column}</option>)}
            </select>
          </label>
          <label className="form-field">
            <span>目标类型</span>
            <select value={castType} onChange={(event) => setCastType(event.target.value)}>
              <option value="int">int</option>
              <option value="float">float</option>
              <option value="string">string</option>
              <option value="datetime">datetime</option>
              <option value="category">category</option>
              <option value="boolean">boolean</option>
            </select>
          </label>
          <button className="button primary" type="button" disabled={busy || !castColumn} onClick={() => applyOperations([{ op: "cast_type", col: castColumn, dtype: castType }], "已转换字段类型")}>
            <Wand2 size={15} aria-hidden="true" />
            转换类型
          </button>
        </article>

        <article className="operation-panel">
          <div className="panel-title">
            <span>
              <BarChart3 size={17} aria-hidden="true" />
              <h2>统计指标计算</h2>
            </span>
          </div>
          {numericCols.length ? (
            <>
              <label className="form-field">
                <span>选择数值列</span>
                <select value={selectedStatsProfile?.name ?? ""} onChange={(event) => setStatsColumn(event.target.value)}>
                  {numericCols.map((column) => <option key={column}>{column}</option>)}
                </select>
              </label>
              <div className="stats-metric-grid">
                <div className="mini-metric"><span>均值</span><strong>{formatStat(selectedStatsProfile?.stats?.mean)}</strong></div>
                <div className="mini-metric"><span>中位数</span><strong>{formatStat(selectedStatsProfile?.stats?.median)}</strong></div>
                <div className="mini-metric"><span>方差</span><strong>{formatStat(selectedStatsProfile?.stats?.var)}</strong></div>
                <div className="mini-metric"><span>标准差</span><strong>{formatStat(selectedStatsProfile?.stats?.std)}</strong></div>
                <div className="mini-metric"><span>最大值</span><strong>{formatStat(selectedStatsProfile?.stats?.max)}</strong></div>
                <div className="mini-metric"><span>最小值</span><strong>{formatStat(selectedStatsProfile?.stats?.min)}</strong></div>
              </div>
            </>
          ) : (
            <div className="empty-list">当前无数值列可计算统计指标。</div>
          )}
        </article>

        <article className="operation-panel span-2">
          <div className="panel-title split">
            <span>
              <AlertTriangle size={17} aria-hidden="true" />
              <h2>异常值处理</h2>
            </span>
            <small>{selectedOutlierCount ? `${selectedOutlierCount} 个待处理` : "当前无异常值"}</small>
          </div>
          {outlierCandidateColumns.length ? (
            <>
              <div className="operation-columns two">
                <label className="form-field">
                  <span>选择异常值字段</span>
                  <ColumnPicker columns={outlierCandidateColumns} selected={outlierColumns} onChange={setOutlierColumns} />
                </label>
                <div className="operation-block">
                  <span className="field-label">检测灵敏度</span>
                  <div className="segmented">
                    <button className={outlierCoefficient === 1.5 ? "selected" : ""} type="button" onClick={() => setOutlierCoefficient(1.5)}>1.5x IQR</button>
                    <button className={outlierCoefficient === 3 ? "selected" : ""} type="button" onClick={() => setOutlierCoefficient(3)}>3.0x IQR</button>
                  </div>
                  <div className="operation-note">处理时后端会基于当前 session 重新计算边界。</div>
                </div>
              </div>

              <div className="outlier-detail-grid">
                {outlierCandidateColumns.slice(0, 6).map((column) => {
                  const info = profile.outliers[column] ?? {};
                  return (
                    <div className="outlier-detail" key={column}>
                      <strong>{column}</strong>
                      <span>{info.count ?? 0} 个 · 边界 [{formatStat(info.lower_bound)}, {formatStat(info.upper_bound)}]</span>
                      <small>{info.method ? `${info.method.toUpperCase()} 检测` : "IQR 检测"}</small>
                    </div>
                  );
                })}
              </div>

              <div className="operation-columns four">
                <button
                  className="button ghost"
                  type="button"
                  disabled={busy || !outlierColumns.length}
                  onClick={() => applyOperations([{ op: "winsorize_outliers", cols: outlierColumns, coefficient: outlierCoefficient }], "已完成异常值缩尾处理")}
                >
                  缩尾处理
                </button>
                <button
                  className={outlierReplaceMethod === "median" ? "button ghost active-soft" : "button ghost"}
                  type="button"
                  disabled={busy || !outlierColumns.length}
                  onClick={() => {
                    setOutlierReplaceMethod("median");
                    void applyOperations([{ op: "replace_outliers", cols: outlierColumns, method: "median", coefficient: outlierCoefficient }], "已将异常值替换为中位数");
                  }}
                >
                  替换为中位数
                </button>
                <button
                  className={outlierReplaceMethod === "mean" ? "button ghost active-soft" : "button ghost"}
                  type="button"
                  disabled={busy || !outlierColumns.length}
                  onClick={() => {
                    setOutlierReplaceMethod("mean");
                    void applyOperations([{ op: "replace_outliers", cols: outlierColumns, method: "mean", coefficient: outlierCoefficient }], "已将异常值替换为均值");
                  }}
                >
                  替换为均值
                </button>
                <button
                  className="button danger"
                  type="button"
                  disabled={busy || !outlierColumns.length}
                  onClick={() => applyOperations([{ op: "drop_outliers", cols: outlierColumns, coefficient: outlierCoefficient }], "已删除包含异常值的行")}
                >
                  删除异常行
                </button>
              </div>
            </>
          ) : (
            <div className="empty-list">当前数据未检测到明显异常值。</div>
          )}
        </article>

        <article className="operation-panel">
          <div className="panel-title">
            <span>
              <Sparkles size={17} aria-hidden="true" />
              <h2>缺失值处理</h2>
            </span>
          </div>
          <label className="form-field">
            <span>选择列</span>
            <ColumnPicker columns={profile.columns} selected={fillColumns} onChange={setFillColumns} />
          </label>
          <div className="operation-columns two">
            <label className="form-field">
              <span>填充方法</span>
              <select value={fillMethod} onChange={(event) => setFillMethod(event.target.value)}>
                <option value="mean">均值</option>
                <option value="median">中位数</option>
                <option value="mode">众数</option>
                <option value="zero">0</option>
                <option value="constant">常数</option>
                <option value="ffill">向前填充</option>
                <option value="bfill">向后填充</option>
              </select>
            </label>
            <label className="form-field">
              <span>常数值</span>
              <input value={fillValue} onChange={(event) => setFillValue(event.target.value)} disabled={fillMethod !== "constant"} />
            </label>
          </div>
          <button className="button primary" type="button" disabled={busy || fillColumns.length === 0} onClick={() => applyOperations([{ op: "fill_na", cols: fillColumns, method: fillMethod, value: fillValue }], "已填充缺失值")}>
            <Eraser size={15} aria-hidden="true" />
            执行填充
          </button>
        </article>

        <article className="operation-panel">
          <div className="panel-title">
            <span>
              <Sigma size={17} aria-hidden="true" />
              <h2>标准化 / 归一化</h2>
            </span>
          </div>
          <ColumnPicker columns={numericCols} selected={scaleColumns} onChange={setScaleColumns} emptyText="当前无数值列" />
          <div className="segmented">
            <button className={scaleMethod === "standard" ? "selected" : ""} type="button" onClick={() => setScaleMethod("standard")}>StandardScaler</button>
            <button className={scaleMethod === "minmax" ? "selected" : ""} type="button" onClick={() => setScaleMethod("minmax")}>MinMaxScaler</button>
          </div>
          <button className="button primary" type="button" disabled={busy || scaleColumns.length === 0} onClick={() => applyOperations([{ op: "scale", cols: scaleColumns, method: scaleMethod }], "已完成数值缩放")}>
            <Sigma size={15} aria-hidden="true" />
            执行缩放
          </button>
        </article>

        <article className="operation-panel">
          <div className="panel-title">
            <span>
              <SplitSquareHorizontal size={17} aria-hidden="true" />
              <h2>类别特征编码</h2>
            </span>
          </div>
          <ColumnPicker columns={categoricalCols} selected={encodeColumns} onChange={setEncodeColumns} emptyText="当前无类别列" />
          <div className="segmented">
            <button className={encodeMethod === "label" ? "selected" : ""} type="button" onClick={() => setEncodeMethod("label")}>标签编码</button>
            <button className={encodeMethod === "onehot" ? "selected" : ""} type="button" onClick={() => setEncodeMethod("onehot")}>独热编码</button>
          </div>
          <button
            className="button primary"
            type="button"
            disabled={busy || encodeColumns.length === 0}
            onClick={() => applyOperations([{ op: encodeMethod === "label" ? "label_encode" : "one_hot_encode", cols: encodeColumns, drop_first: true }], "已完成类别编码")}
          >
            <Binary size={15} aria-hidden="true" />
            执行编码
          </button>
        </article>

        <article className="operation-panel">
          <div className="panel-title">
            <span>
              <Sparkles size={17} aria-hidden="true" />
              <h2>添加数据噪声</h2>
            </span>
          </div>
          <ColumnPicker columns={numericCols} selected={noiseColumns} onChange={setNoiseColumns} emptyText="当前无数值列" />
          <label className="form-field">
            <span>噪声强度：{noiseLevel.toFixed(2)}</span>
            <input min={0.01} max={0.5} step={0.01} type="range" value={noiseLevel} onChange={(event) => setNoiseLevel(Number(event.target.value))} />
          </label>
          <button className="button primary" type="button" disabled={busy || noiseColumns.length === 0} onClick={() => applyOperations([{ op: "add_noise", cols: noiseColumns, level: noiseLevel }], "已添加高斯噪声")}>
            <Sparkles size={15} aria-hidden="true" />
            添加噪声
          </button>
        </article>

        <article className="operation-panel span-2">
          <div className="panel-title split">
            <span>
              <Calculator size={17} aria-hidden="true" />
              <h2>自定义计算列</h2>
            </span>
            <small>支持简单计算和安全表达式</small>
          </div>
          <div className="operation-columns three">
            <label className="form-field">
              <span>一元列</span>
              <select value={unaryColumn} onChange={(event) => setUnaryColumn(event.target.value)}>
                {numericCols.map((column) => <option key={column}>{column}</option>)}
              </select>
            </label>
            <label className="form-field">
              <span>运算</span>
              <select value={unaryOperator} onChange={(event) => setUnaryOperator(event.target.value)}>
                <option value="square">平方</option>
                <option value="sqrt">开方</option>
                <option value="log">取对数</option>
                <option value="abs">绝对值</option>
                <option value="round">四舍五入</option>
                <option value="neg">取负</option>
              </select>
            </label>
            <label className="form-field">
              <span>新列名</span>
              <input value={unaryNewColumn} onChange={(event) => setUnaryNewColumn(event.target.value)} />
            </label>
          </div>
          <button className="button ghost" type="button" disabled={busy || !unaryColumn || !unaryNewColumn} onClick={() => applyOperations([{ op: "unary_calc", col: unaryColumn, operator: unaryOperator, new_col: unaryNewColumn, decimals: 2 }], "已生成一元计算列")}>
            <Calculator size={15} aria-hidden="true" />
            生成一元计算列
          </button>
          <div className="operation-columns three">
            <label className="form-field">
              <span>第一列</span>
              <select value={binaryLeft} onChange={(event) => setBinaryLeft(event.target.value)}>
                {numericCols.map((column) => <option key={column}>{column}</option>)}
              </select>
            </label>
            <label className="form-field">
              <span>运算符</span>
              <select value={binaryOperator} onChange={(event) => setBinaryOperator(event.target.value)}>
                <option value="+">+</option>
                <option value="-">-</option>
                <option value="*">*</option>
                <option value="/">/</option>
                <option value="//">//</option>
                <option value="%">%</option>
                <option value="**">^</option>
              </select>
            </label>
            <label className="form-field">
              <span>新列名</span>
              <input value={binaryNewColumn} onChange={(event) => setBinaryNewColumn(event.target.value)} />
            </label>
          </div>
          <div className="operation-columns three">
            <label className="form-field">
              <span>第二操作数</span>
              <select value={binaryRightMode} onChange={(event) => setBinaryRightMode(event.target.value as "column" | "constant")}>
                <option value="column">选择列</option>
                <option value="constant">常数</option>
              </select>
            </label>
            {binaryRightMode === "column" ? (
              <label className="form-field">
                <span>第二列</span>
                <select value={binaryRight} onChange={(event) => setBinaryRight(event.target.value)}>
                  {numericCols.map((column) => <option key={column}>{column}</option>)}
                </select>
              </label>
            ) : (
              <label className="form-field">
                <span>常数值</span>
                <input type="number" value={binaryValue} onChange={(event) => setBinaryValue(Number(event.target.value))} />
              </label>
            )}
            <button
              className="button ghost align-end"
              type="button"
              disabled={busy || !binaryLeft || !binaryNewColumn}
              onClick={() => applyOperations([{
                op: "binary_calc",
                left_col: binaryLeft,
                right_col: binaryRightMode === "column" ? binaryRight : undefined,
                right_value: binaryRightMode === "constant" ? binaryValue : undefined,
                operator: binaryOperator,
                new_col: binaryNewColumn
              }], "已生成二元计算列")}
            >
              <Binary size={15} aria-hidden="true" />
              生成二元计算列
            </button>
          </div>
          <div className="operation-columns two">
            <label className="form-field">
              <span>表达式</span>
              <textarea placeholder="例如：(col_A + col_B) / 100" value={formulaExpression} onChange={(event) => setFormulaExpression(event.target.value)} />
            </label>
            <label className="form-field">
              <span>新列名</span>
              <input value={formulaNewColumn} onChange={(event) => setFormulaNewColumn(event.target.value)} />
              <small>可用数值列：{formulaHint || "当前列名不适合作为表达式变量，可先重命名"}</small>
            </label>
          </div>
          <button className="button primary" type="button" disabled={busy || !formulaExpression || !formulaNewColumn} onClick={() => applyOperations([{ op: "custom_formula", expr: formulaExpression, new_col: formulaNewColumn }], "已生成表达式计算列")}>
            <Braces size={15} aria-hidden="true" />
            执行表达式
          </button>
        </article>

        <article className="operation-panel span-2">
          <div className="panel-title split">
            <span>
              <Sigma size={17} aria-hidden="true" />
              <h2>PCA 降维</h2>
            </span>
            <small>主成分会追加为 PCA_1、PCA_2...</small>
          </div>
          <ColumnPicker columns={numericCols.filter((column) => !column.startsWith("PCA_"))} selected={pcaColumns} onChange={setPcaColumns} emptyText="至少需要 2 个非 PCA 数值列" />
          <div className="operation-columns three">
            <label className="form-field">
              <span>降维维度：{Math.min(pcaComponents, pcaMax)}</span>
              <input min={1} max={pcaMax} type="range" value={Math.min(pcaComponents, pcaMax)} onChange={(event) => setPcaComponents(Number(event.target.value))} />
            </label>
            <label className="checkbox-line">
              <input type="checkbox" checked={pcaDropOriginal} onChange={(event) => setPcaDropOriginal(event.target.checked)} />
              <span>生成后删除原始列</span>
            </label>
            <button className="button primary align-end" type="button" disabled={busy || pcaColumns.length < 2} onClick={() => applyOperations([{ op: "pca", cols: pcaColumns, n_components: Math.min(pcaComponents, pcaMax), prefix: "PCA", drop_original: pcaDropOriginal }], "已完成 PCA 降维")}>
              <Sigma size={15} aria-hidden="true" />
              执行 PCA
            </button>
          </div>
        </article>
      </div>

      <article className="operation-panel span-2 preview-after-processing">
        <div className="panel-title split">
          <span>
            <ListChecks size={17} aria-hidden="true" />
            <h2>处理后预览</h2>
          </span>
          <small>前 {previewRows.length} 行 · 显示 {previewColumns.length} 列</small>
        </div>
        <div className="table-wrap">
          <table className="data-table">
            <thead>
              <tr>
                {previewColumns.map((column) => <th key={column}>{column}</th>)}
              </tr>
            </thead>
            <tbody>
              {previewRows.map((row, rowIndex) => (
                <tr key={rowIndex}>
                  {previewColumns.map((column) => <td key={column}>{valueToText(row[column])}</td>)}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </article>
    </section>
  );
}
