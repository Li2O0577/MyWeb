export type BackendStatus = "checking" | "online" | "offline" | "degraded";

export type SavedModels = Record<string, boolean | number | undefined>;

export interface AuthUser {
  user_id: string;
  username: string;
  created_at?: number;
}

export interface SessionMeta {
  session_id?: string;
  source_name?: string;
  rows?: number;
  n_columns?: number;
  updated_at_iso?: string;
  ttl_seconds_remaining?: number;
}

export interface ColumnProfile {
  name: string;
  dtype: string;
  kind: "numeric" | "categorical" | "datetime" | "boolean";
  missing_count: number;
  missing_pct: number;
  unique_count: number;
  sample_values: unknown[];
  outlier_count: number;
  nan_count: number;
  stats?: {
    mean: number;
    median: number;
    var: number;
    std: number;
    min: number;
    max: number;
  } | null;
}

export interface OutlierInfo {
  count?: number;
  nan_count?: number;
  lower_bound?: number | null;
  upper_bound?: number | null;
  method?: string;
  indices?: Array<string | number>;
  values?: unknown[];
}

export type OutlierMap = Record<string, OutlierInfo>;

export interface DataProfile {
  session_id: string;
  session_meta?: SessionMeta;
  n_rows: number;
  n_cols: number;
  columns: string[];
  numeric_cols: string[];
  categorical_cols: string[];
  dtypes: Record<string, string>;
  missing_counts: Record<string, number>;
  missing_total: number;
  outliers: OutlierMap;
  column_profiles: ColumnProfile[];
  preview: Array<Record<string, unknown>>;
  summary: string;
  processing_history?: ProcessingHistory;
}

export interface ProcessingHistoryEntry {
  state_id: string;
  label: string;
  operations: Array<Record<string, unknown>>;
  messages: string[];
  created_at: number;
  n_rows: number;
  n_cols: number;
}

export interface ProcessingPipeline {
  pipeline_id: string;
  name: string;
  operations: Array<Record<string, unknown>>;
  created_at: number;
  step_count: number;
}

export interface ProcessingHistory {
  history: ProcessingHistoryEntry[];
  current_index: number;
  can_undo: boolean;
  can_redo: boolean;
  pipelines: ProcessingPipeline[];
}

export interface HealthResponse {
  status: "ok" | "degraded";
  active_sessions: number;
  recent_sessions: SessionMeta[];
  saved_models: SavedModels;
  error?: string;
}

export type DataUploadResponse = DataProfile;

export type ModelType = "decision_tree" | "clustering" | "regression" | "classification" | "diy_mlp";

export interface ModelVersion {
  version_id: string;
  created_at?: string;
  dataset_name?: string;
  features?: string[];
  target?: string;
  metrics?: Record<string, unknown>;
  params?: Record<string, unknown>;
}

export interface ModelVersionsResponse {
  versions: ModelVersion[];
  active?: string | null;
}

export type TaskStatus = "queued" | "running" | "cancelling" | "succeeded" | "failed" | "cancelled";

export interface TaskRecord {
  task_id: string;
  kind: string;
  label: string;
  status: TaskStatus;
  created_at: number;
  updated_at: number;
  finished_at?: number | null;
  duration_sec?: number | null;
  metadata?: Record<string, unknown>;
  result?: Record<string, unknown> | null;
  result_payload?: Record<string, unknown> | null;
  error?: string | null;
}

export interface TaskListResponse {
  tasks: TaskRecord[];
}

export type ApiJson = Record<string, unknown>;

export type LlmMode = "direct" | "agent";

export interface LlmConfig {
  api_base: string;
  api_key: string;
  model: string;
}

export interface LlmPayloadMessage {
  role: "system" | "user" | "assistant";
  content: string;
}

export interface LlmImage {
  base64: string;
  title?: string;
  alt?: string;
}

export interface LlmStreamEvent {
  chunk?: string;
  full?: string;
  done?: boolean;
  status?: "thinking" | "tool_call" | "tool_result" | "image";
  message?: string;
  tool?: string;
  args?: Record<string, unknown>;
  result?: string;
  images?: LlmImage[];
  base64?: string;
  title?: string;
  alt?: string;
  tools_used?: number;
  error?: {
    code?: string;
    message?: string;
    detail?: string;
  };
}
