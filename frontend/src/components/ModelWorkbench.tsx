import { useCallback, useEffect, useMemo, useState } from "react";
import { AlertTriangle, Brain, CheckCircle2, Clock3, RefreshCcw } from "lucide-react";
import type { DataProfile, ModelType, TaskRecord } from "../types";
import ClassificationWorkspace from "./ClassificationWorkspace";
import ClusteringWorkspace from "./ClusteringWorkspace";
import DecisionTreeWorkspace from "./DecisionTreeWorkspace";
import DiyMlpWorkspace from "./DiyMlpWorkspace";
import RegressionWorkspace from "./RegressionWorkspace";
import { cancelTask, fetchTasks, TRAINING_TASK_EVENT } from "../services/api";

interface Props {
  profile: DataProfile | null;
}

interface ModelConfig {
  type: ModelType;
  label: string;
}

const MODELS: ModelConfig[] = [
  { type: "decision_tree", label: "决策树" },
  { type: "clustering", label: "聚类" },
  { type: "regression", label: "回归" },
  { type: "classification", label: "分类" },
  { type: "diy_mlp", label: "自定义 MLP" }
];

const MODEL_LABELS: Record<string, string> = {
  decision_tree: "决策树",
  clustering: "聚类",
  regression: "回归",
  classification: "分类",
  diy_mlp: "自定义 MLP"
};

function formatTaskTime(value: number) {
  if (!value) return "--";
  return new Date(value * 1000).toLocaleTimeString("zh-CN", {
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit"
  });
}

function taskModelLabel(task: TaskRecord) {
  const modelType = task.metadata?.model_type;
  return typeof modelType === "string" ? MODEL_LABELS[modelType] ?? modelType : task.label;
}

function taskResultText(task: TaskRecord) {
  if (task.status === "failed") return task.error || "训练失败";
  if (task.status === "cancelled") return "已取消";
  if (task.status === "cancelling") return "正在取消";
  if (task.status === "queued") return "排队中";
  if (task.status === "running") return "训练中";
  const versionId = task.result?.version_id;
  if (typeof versionId === "string" && versionId) return `版本 ${versionId}`;
  const metrics = task.result?.metrics;
  if (metrics && typeof metrics === "object") return "已生成指标";
  return "已完成";
}

export default function ModelWorkbench({ profile }: Props) {
  const [modelType, setModelType] = useState<ModelType>("decision_tree");
  const [tasks, setTasks] = useState<TaskRecord[]>([]);
  const [tasksError, setTasksError] = useState("");
  const [tasksLoading, setTasksLoading] = useState(false);
  const [cancelingTaskId, setCancelingTaskId] = useState("");

  const trainingTasks = useMemo(
    () => tasks.filter((task) => task.kind === "training").slice(0, 5),
    [tasks]
  );

  const loadTasks = useCallback(async (signal?: AbortSignal, quiet = false) => {
    if (!quiet) setTasksLoading(true);
    try {
      const payload = await fetchTasks(signal);
      setTasks(payload.tasks ?? []);
      setTasksError("");
    } catch (error) {
      if ((error as Error).name !== "AbortError") {
        setTasksError((error as Error).message || "读取任务状态失败");
      }
    } finally {
      if (!quiet) setTasksLoading(false);
    }
  }, []);

  useEffect(() => {
    const controller = new AbortController();
    void loadTasks(controller.signal);
    const timer = window.setInterval(() => {
      void loadTasks(controller.signal, true);
    }, 5000);
    return () => {
      controller.abort();
      window.clearInterval(timer);
    };
  }, [loadTasks]);

  const cancelTrainingTask = async (taskId: string) => {
    setCancelingTaskId(taskId);
    try {
      await cancelTask(taskId);
      await loadTasks(undefined, true);
    } finally {
      setCancelingTaskId("");
    }
  };

  useEffect(() => {
    const timeouts: number[] = [];
    const refreshAfterTrainingEvent = () => {
      void loadTasks(undefined, true);
      const timeout = window.setTimeout(() => {
        void loadTasks(undefined, true);
      }, 800);
      timeouts.push(timeout);
    };
    window.addEventListener(TRAINING_TASK_EVENT, refreshAfterTrainingEvent);
    return () => {
      window.removeEventListener(TRAINING_TASK_EVENT, refreshAfterTrainingEvent);
      timeouts.forEach((timeout) => window.clearTimeout(timeout));
    };
  }, [loadTasks]);

  const renderWorkspace = () => {
    if (!profile) return <div className="empty-list">上传数据后即可训练模型。</div>;
    if (modelType === "decision_tree") return <DecisionTreeWorkspace profile={profile} />;
    if (modelType === "clustering") return <ClusteringWorkspace profile={profile} />;
    if (modelType === "regression") return <RegressionWorkspace profile={profile} />;
    if (modelType === "classification") return <ClassificationWorkspace profile={profile} />;
    return <DiyMlpWorkspace profile={profile} />;
  };

  return (
    <section className="model-section" aria-label="模型训练">
      <div className="section-heading">
        <Brain size={18} aria-hidden="true" />
        <div>
          <h2>模型训练与预测</h2>
          <p>每类模型保留独立工作台、版本列表、单条预测和批量预测入口。</p>
        </div>
      </div>

      <div className="model-tabs" role="tablist" aria-label="模型类型">
        {MODELS.map((model) => (
          <button
            className={model.type === modelType ? "model-tab active" : "model-tab"}
            key={model.type}
            type="button"
            onClick={() => setModelType(model.type)}
          >
            {model.label}
          </button>
        ))}
      </div>

      <div className="task-status-panel" aria-label="训练任务状态">
        <div className="task-status-header">
          <div>
            <strong>训练任务状态</strong>
            <span>训练通过后台任务执行，完成后自动同步版本和结果。</span>
          </div>
          <button
            className="icon-button"
            type="button"
            title="刷新任务状态"
            aria-label="刷新任务状态"
            onClick={() => void loadTasks()}
            disabled={tasksLoading}
          >
            <RefreshCcw size={15} aria-hidden="true" className={tasksLoading ? "spin" : undefined} />
          </button>
        </div>
        {tasksError ? (
          <div className="task-inline-error">
            <AlertTriangle size={14} aria-hidden="true" />
            <span>{tasksError}</span>
          </div>
        ) : null}
        {trainingTasks.length ? (
          <div className="task-list">
            {trainingTasks.map((task) => (
              <div className="task-row" key={task.task_id}>
                <div className="task-row-main">
                  {task.status === "succeeded" ? (
                    <CheckCircle2 size={15} aria-hidden="true" />
                  ) : task.status === "failed" ? (
                    <AlertTriangle size={15} aria-hidden="true" />
                  ) : (
                    <Clock3 size={15} aria-hidden="true" />
                  )}
                  <div>
                    <strong>{taskModelLabel(task)}</strong>
                    <span>{taskResultText(task)}</span>
                  </div>
                </div>
                <div className="task-row-meta">
                  <span className={`task-badge ${task.status}`}>
                    {task.status === "succeeded"
                      ? "完成"
                      : task.status === "failed"
                        ? "失败"
                        : task.status === "cancelled"
                          ? "已取消"
                          : task.status === "cancelling"
                            ? "取消中"
                            : task.status === "queued"
                              ? "排队"
                              : "运行中"}
                  </span>
                  <span>{formatTaskTime(task.created_at)}</span>
                  <span>{task.duration_sec ? `${task.duration_sec}s` : "--"}</span>
                  {["queued", "running", "cancelling"].includes(task.status) ? (
                    <button
                      className="tiny-button"
                      type="button"
                      disabled={task.status === "cancelling" || cancelingTaskId === task.task_id}
                      onClick={() => void cancelTrainingTask(task.task_id)}
                    >
                      {task.status === "cancelling" || cancelingTaskId === task.task_id ? "取消中" : "取消"}
                    </button>
                  ) : null}
                </div>
              </div>
            ))}
          </div>
        ) : (
          <div className="empty-list compact">暂无训练任务。</div>
        )}
      </div>

      {renderWorkspace()}
    </section>
  );
}
