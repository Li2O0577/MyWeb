import { useState } from "react";
import { Brain } from "lucide-react";
import type { DataProfile, ModelType } from "../types";
import ClassificationWorkspace from "./ClassificationWorkspace";
import ClusteringWorkspace from "./ClusteringWorkspace";
import DecisionTreeWorkspace from "./DecisionTreeWorkspace";
import DiyMlpWorkspace from "./DiyMlpWorkspace";
import RegressionWorkspace from "./RegressionWorkspace";

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

export default function ModelWorkbench({ profile }: Props) {
  const [modelType, setModelType] = useState<ModelType>("decision_tree");

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

      {renderWorkspace()}
    </section>
  );
}
