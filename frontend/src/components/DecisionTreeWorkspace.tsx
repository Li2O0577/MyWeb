import { useEffect, useMemo, useState } from "react";
import {
  AlertTriangle,
  Brain,
  CheckCircle2,
  GitBranch,
  ListTree,
  Play,
  RefreshCcw,
  Rows3,
  Send,
  Trash2
} from "lucide-react";
import {
  activateModelVersion,
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

type TaskType = "classification" | "regression";

interface TreeNode {
  id: number;
  feature: string;
  threshold: string | number;
  impurity: number;
  samples: number;
  type: string;
}

interface DecisionTreeResult extends ApiJson {
  version_id?: string;
  acc?: number;
  r2?: number;
  mae?: number;
  rmse?: number;
  cm?: number[][];
  label_names?: string[];
  tree_rules?: string;
  tree_nodes?: TreeNode[];
  criterion_name?: string;
  metrics?: Record<string, unknown>;
  params?: Record<string, unknown>;
  features?: string[];
  target?: string;
  categorical_features?: string[];
  has_model?: boolean;
}

function formatNumber(value: unknown, digits = 4) {
  const n = Number(value);
  return Number.isFinite(n) ? n.toFixed(digits) : "-";
}

function valueToText(value: unknown) {
  if (value === null || value === undefined) return "";
  return String(value);
}

function isNumericColumn(profile: DataProfile, column: string) {
  return profile.numeric_cols.includes(column);
}

function defaultTarget(profile: DataProfile, task: TaskType) {
  if (task === "classification") {
    return profile.columns.find((column) => !profile.numeric_cols.includes(column)) ?? profile.columns[profile.columns.length - 1] ?? "";
  }
  return profile.numeric_cols[profile.numeric_cols.length - 1] ?? profile.columns[profile.columns.length - 1] ?? "";
}

function defaultFeatures(profile: DataProfile, target: string) {
  return profile.columns.filter((column) => column !== target).slice(0, Math.min(6, Math.max(0, profile.columns.length - 1)));
}

function normalizeResult(payload: ApiJson | null): DecisionTreeResult | null {
  if (!payload) return null;
  const metrics = (payload.metrics ?? {}) as Record<string, unknown>;
  return {
    ...payload,
    acc: payload.acc as number | undefined ?? metrics.acc as number | undefined,
    r2: payload.r2 as number | undefined ?? metrics.r2 as number | undefined,
    mae: payload.mae as number | undefined ?? metrics.mae as number | undefined,
    rmse: payload.rmse as number | undefined ?? metrics.rmse as number | undefined
  } as DecisionTreeResult;
}

function MetricStrip({ result, taskType }: { result: DecisionTreeResult | null; taskType: TaskType }) {
  const metrics =
    taskType === "classification"
      ? [["准确率", result?.acc]]
      : [["R²", result?.r2], ["MAE", result?.mae], ["RMSE", result?.rmse]];
  return (
    <div className="metric-grid">
      {metrics.map(([label, value]) => (
        <div className="mini-metric" key={String(label)}>
          <span>{label}</span>
          <strong>{formatNumber(value)}</strong>
        </div>
      ))}
      <div className="mini-metric">
        <span>版本</span>
        <strong>{result?.version_id ? String(result.version_id).slice(0, 12) : "-"}</strong>
      </div>
    </div>
  );
}

function ConfusionMatrix({ matrix, labels }: { matrix?: number[][]; labels?: string[] }) {
  if (!matrix?.length) return <div className="empty-list">分类训练后显示混淆矩阵。</div>;
  const max = Math.max(1, ...matrix.flat());
  return (
    <div className="confusion-wrap">
      <table className="confusion-table">
        <thead>
          <tr>
            <th>实际 / 预测</th>
            {matrix.map((_, index) => <th key={index}>{labels?.[index] ?? index}</th>)}
          </tr>
        </thead>
        <tbody>
          {matrix.map((row, rowIndex) => (
            <tr key={rowIndex}>
              <th>{labels?.[rowIndex] ?? rowIndex}</th>
              {row.map((value, colIndex) => (
                <td key={`${rowIndex}-${colIndex}`} style={{ backgroundColor: `rgba(164, 81, 42, ${0.08 + (value / max) * 0.42})` }}>
                  {value}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
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

export default function DecisionTreeWorkspace({ profile }: Props) {
  const [taskType, setTaskType] = useState<TaskType>("classification");
  const [targetCol, setTargetCol] = useState(defaultTarget(profile, "classification"));
  const [featureCols, setFeatureCols] = useState<string[]>(defaultFeatures(profile, targetCol));
  const [maxDepth, setMaxDepth] = useState(3);
  const [criterion, setCriterion] = useState("entropy");
  const [trainResult, setTrainResult] = useState<DecisionTreeResult | null>(null);
  const [predictResult, setPredictResult] = useState<ApiJson | null>(null);
  const [versions, setVersions] = useState<ModelVersion[]>([]);
  const [activeVersion, setActiveVersion] = useState<string | null | undefined>(null);
  const [selectedVersion, setSelectedVersion] = useState("");
  const [inputs, setInputs] = useState<Record<string, string>>({});
  const [loading, setLoading] = useState(false);
  const [versionsLoading, setVersionsLoading] = useState(false);
  const [error, setError] = useState("");

  const classificationCriteria = ["entropy", "gini"];
  const regressionCriteria = ["squared_error", "friedman_mse", "absolute_error", "poisson"];
  const criteria = taskType === "classification" ? classificationCriteria : regressionCriteria;
  const targetProfile = profile.column_profiles.find((column) => column.name === targetCol);
  const taskInvalid =
    taskType === "classification"
      ? (targetProfile?.unique_count ?? 0) < 2
      : !profile.numeric_cols.includes(targetCol);
  const cleanRows = profile.n_rows - Math.max(...featureCols.concat(targetCol).map((column) => profile.missing_counts[column] ?? 0), 0);
  const canTrain = Boolean(profile.session_id && targetCol && featureCols.length && !taskInvalid && cleanRows >= 5);

  useEffect(() => {
    const nextTarget = defaultTarget(profile, taskType);
    setTargetCol(nextTarget);
    setFeatureCols(defaultFeatures(profile, nextTarget));
    setCriterion(taskType === "classification" ? "entropy" : "squared_error");
    setTrainResult(null);
    setPredictResult(null);
  }, [profile.session_id, taskType]);

  useEffect(() => {
    const firstRow = profile.preview[0] ?? {};
    const next: Record<string, string> = {};
    for (const column of featureCols) next[column] = valueToText(firstRow[column]);
    setInputs(next);
  }, [profile.session_id, featureCols]);

  const availableFeatures = profile.columns.filter((column) => column !== targetCol);
  const categoricalFeatures = useMemo(
    () => featureCols.filter((column) => !isNumericColumn(profile, column)),
    [featureCols, profile]
  );

  const loadDecisionTreeState = async () => {
    setVersionsLoading(true);
    setError("");
    try {
      const [versionPayload, statusPayload] = await Promise.all([
        fetchModelVersions("decision_tree"),
        fetchModelStatus("decision_tree")
      ]);
      setVersions(versionPayload.versions ?? []);
      setActiveVersion(versionPayload.active);
      setSelectedVersion(versionPayload.active ?? "");
      const status = normalizeResult(statusPayload);
      if (status?.has_model) {
        setTrainResult(status);
        if (status.params?.task_type === "classification" || status.params?.task_type === "regression") {
          setTaskType(status.params.task_type);
        }
        if (status.target) setTargetCol(status.target);
        if (status.features?.length) setFeatureCols(status.features);
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : "读取决策树状态失败");
    } finally {
      setVersionsLoading(false);
    }
  };

  useEffect(() => {
    void loadDecisionTreeState();
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
    try {
      const payload = await trainModel("decision_tree", {
        session_id: profile.session_id,
        target_col: targetCol,
        feature_cols: featureCols,
        task_type: taskType,
        criterion,
        max_depth: maxDepth
      });
      const normalized = normalizeResult(payload);
      setTrainResult(normalized);
      if (normalized?.version_id) {
        setSelectedVersion(normalized.version_id);
        setActiveVersion(normalized.version_id);
      }
      await loadDecisionTreeState();
    } catch (err) {
      setError(err instanceof Error ? err.message : "决策树训练失败");
    } finally {
      setLoading(false);
    }
  };

  const activateVersion = async (versionId: string) => {
    setError("");
    try {
      await activateModelVersion("decision_tree", versionId);
      const detail = await fetchModelVersionDetail("decision_tree", versionId);
      setTrainResult(normalizeResult({ ...detail, has_model: true }));
      setSelectedVersion(versionId);
      await loadDecisionTreeState();
    } catch (err) {
      setError(err instanceof Error ? err.message : "切换决策树版本失败");
    }
  };

  const removeVersion = async (versionId: string) => {
    setError("");
    try {
      await deleteModelVersion("decision_tree", versionId);
      if (selectedVersion === versionId) setSelectedVersion("");
      await loadDecisionTreeState();
    } catch (err) {
      setError(err instanceof Error ? err.message : "删除版本失败");
    }
  };

  const clearActive = async () => {
    setError("");
    try {
      await clearModel("decision_tree");
      setTrainResult(null);
      setPredictResult(null);
      await loadDecisionTreeState();
    } catch (err) {
      setError(err instanceof Error ? err.message : "清除模型失败");
    }
  };

  const handlePredict = async () => {
    if (!featureCols.length) return;
    setLoading(true);
    setError("");
    setPredictResult(null);
    try {
      const inputDict: Record<string, unknown[]> = {};
      for (const column of featureCols) inputDict[column] = [inputs[column] ?? ""];
      const result = await predictModel("decision_tree", {
        input_dict: inputDict,
        task_type: taskType,
        version_id: selectedVersion || undefined
      });
      setPredictResult(result);
    } catch (err) {
      setError(err instanceof Error ? err.message : "预测失败");
    } finally {
      setLoading(false);
    }
  };

  return (
    <section className="decision-tree-workspace" aria-label="决策树工作台">
      <div className="dt-toolbar">
        <div>
          <span>当前数据</span>
          <strong>{profile.n_rows.toLocaleString()} 行 · {profile.n_cols} 列</strong>
        </div>
        <div>
          <span>特征结构</span>
          <strong>{profile.numeric_cols.length} 数值 · {profile.categorical_cols.length} 类别</strong>
        </div>
        <div>
          <span>有效行估计</span>
          <strong>{Math.max(0, cleanRows).toLocaleString()} 行</strong>
        </div>
      </div>

      {error ? <div className="inline-error">{error}</div> : null}
      {taskInvalid ? (
        <div className="inline-warning">
          {taskType === "classification" ? "分类任务的目标列至少需要 2 个类别。" : "回归任务的目标列必须是数值列。"}
        </div>
      ) : null}

      <div className="dt-layout">
        <aside className="dt-config" aria-label="决策树训练配置">
          <div className="panel-title split">
            <span>
              <ListTree size={17} aria-hidden="true" />
              <h2>训练配置</h2>
            </span>
            <small>{taskType === "classification" ? "分类树" : "回归树"}</small>
          </div>

          <div className="segmented">
            <button className={taskType === "classification" ? "selected" : ""} type="button" onClick={() => setTaskType("classification")}>分类</button>
            <button className={taskType === "regression" ? "selected" : ""} type="button" onClick={() => setTaskType("regression")}>回归</button>
          </div>

          <label className="form-field">
            <span>目标列</span>
            <select value={targetCol} onChange={(event) => {
              setTargetCol(event.target.value);
              setFeatureCols(defaultFeatures(profile, event.target.value));
            }}>
              {profile.columns.map((column) => <option key={column}>{column}</option>)}
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

          <div className="operation-columns two">
            <label className="form-field">
              <span>最大深度：{maxDepth}</span>
              <input min={2} max={15} type="range" value={maxDepth} onChange={(event) => setMaxDepth(Number(event.target.value))} />
            </label>
            <label className="form-field">
              <span>划分标准</span>
              <select value={criterion} onChange={(event) => setCriterion(event.target.value)}>
                {criteria.map((item) => <option key={item}>{item}</option>)}
              </select>
            </label>
          </div>

          <div className="dt-feature-summary">
            <span>已选特征 {featureCols.length}</span>
            <span>类别特征 {categoricalFeatures.length}</span>
            <span>目标唯一值 {targetProfile?.unique_count ?? "-"}</span>
          </div>

          <div className="prediction-actions">
            <button className="button primary" type="button" disabled={!canTrain || loading} onClick={handleTrain}>
              <Play size={15} aria-hidden="true" />
              {loading ? "训练中..." : "开始训练"}
            </button>
            <button className="button danger" type="button" onClick={clearActive} disabled={loading}>
              <Trash2 size={15} aria-hidden="true" />
              清除当前模型
            </button>
          </div>
        </aside>

        <section className="dt-main" aria-label="决策树结果">
          <article className="dt-panel">
            <div className="panel-title split">
              <span>
                <CheckCircle2 size={17} aria-hidden="true" />
                <h2>训练结果</h2>
              </span>
              <small>{trainResult?.criterion_name ? `划分标准：${trainResult.criterion_name}` : "等待训练"}</small>
            </div>
            <MetricStrip result={trainResult} taskType={taskType} />
            {taskType === "classification" ? (
              <ConfusionMatrix matrix={trainResult?.cm} labels={trainResult?.label_names} />
            ) : null}
          </article>

          <article className="dt-panel">
            <div className="panel-title split">
              <span>
                <GitBranch size={17} aria-hidden="true" />
                <h2>决策树规则</h2>
              </span>
              <small>{trainResult?.tree_nodes?.length ? `${trainResult.tree_nodes.length} 个节点` : "暂无节点"}</small>
            </div>
            {trainResult?.tree_rules ? (
              <pre className="tree-rules">{String(trainResult.tree_rules)}</pre>
            ) : (
              <div className="empty-list">训练后显示 export_text 规则。</div>
            )}
          </article>

          <article className="dt-panel">
            <div className="panel-title split">
              <span>
                <Rows3 size={17} aria-hidden="true" />
                <h2>节点明细</h2>
              </span>
              <small>内部节点 / 叶子节点</small>
            </div>
            {trainResult?.tree_nodes?.length ? (
              <div className="table-wrap compact-table">
                <table className="data-table">
                  <thead>
                    <tr>
                      <th>ID</th>
                      <th>类型</th>
                      <th>特征</th>
                      <th>阈值</th>
                      <th>Impurity</th>
                      <th>Samples</th>
                    </tr>
                  </thead>
                  <tbody>
                    {trainResult.tree_nodes.slice(0, 160).map((node) => (
                      <tr key={node.id}>
                        <td>{node.id}</td>
                        <td>{node.type}</td>
                        <td>{node.feature}</td>
                        <td>{node.threshold}</td>
                        <td>{formatNumber(node.impurity)}</td>
                        <td>{node.samples}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            ) : (
              <div className="empty-list">训练后显示每层节点信息。</div>
            )}
          </article>
        </section>
      </div>

      <div className="dt-bottom-grid">
        <article className="dt-panel">
          <div className="panel-title split">
            <span>
              <RefreshCcw size={17} aria-hidden="true" />
              <h2>模型版本</h2>
            </span>
            <button className="tiny-button" type="button" onClick={loadDecisionTreeState} disabled={versionsLoading}>刷新</button>
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
            )) : <div className="empty-list">暂无决策树版本。</div>}
          </div>
        </article>

        <article className="dt-panel">
          <div className="panel-title">
            <Send size={17} aria-hidden="true" />
            <h2>单条预测</h2>
          </div>
          {featureCols.length ? (
            <div className="dt-predict-grid">
              {featureCols.slice(0, 12).map((column) => {
                const columnProfile = profile.column_profiles.find((item) => item.name === column);
                const options = columnProfile?.kind === "categorical" || columnProfile?.kind === "boolean" ? columnProfile.sample_values : [];
                return (
                  <label className="form-field" key={column}>
                    <span>{column}</span>
                    {options.length ? (
                      <select value={inputs[column] ?? ""} onChange={(event) => setInputs({ ...inputs, [column]: event.target.value })}>
                        {options.map((value) => <option key={valueToText(value)} value={valueToText(value)}>{valueToText(value)}</option>)}
                      </select>
                    ) : (
                      <input value={inputs[column] ?? ""} onChange={(event) => setInputs({ ...inputs, [column]: event.target.value })} />
                    )}
                  </label>
                );
              })}
            </div>
          ) : <div className="empty-list">请选择特征列。</div>}
          <label className="form-field">
            <span>预测版本</span>
            <select value={selectedVersion} onChange={(event) => setSelectedVersion(event.target.value)}>
              <option value="">当前激活版本</option>
              {versions.map((version) => <option key={version.version_id} value={version.version_id}>{version.version_id}</option>)}
            </select>
          </label>
          <button className="button primary full" type="button" onClick={handlePredict} disabled={!featureCols.length || loading}>
            <Brain size={15} aria-hidden="true" />
            {taskType === "classification" ? "执行分类预测" : "执行回归预测"}
          </button>
          {predictResult ? (
            <div className="prediction-result">
              <strong>{predictResult.pred_class ? `预测类别：${predictResult.pred_class}` : `预测值：${formatNumber(predictResult.pred_value)}`}</strong>
              {typeof predictResult.prob === "number" ? <span>置信度：{(predictResult.prob * 100).toFixed(1)}%</span> : null}
              <ProbabilityBars result={predictResult} />
            </div>
          ) : null}
        </article>
      </div>

      {cleanRows < profile.n_rows ? (
        <div className="inline-warning">
          <AlertTriangle size={15} aria-hidden="true" />
          决策树后端训练会丢弃目标列或特征列中的缺失行。需要更精细控制时，请先到“数据处理”页面填充或删除缺失值。
        </div>
      ) : null}
    </section>
  );
}
