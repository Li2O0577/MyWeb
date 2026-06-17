import type {
  ApiJson,
  AuthUser,
  DataUploadResponse,
  HealthResponse,
  LlmStreamEvent,
  ModelType,
  ModelVersionsResponse,
  OutlierMap,
  ProcessingHistory,
  ProcessingPipeline,
  TaskListResponse,
  TaskRecord
} from "../types";

const API_BASE = (import.meta.env.VITE_API_BASE ?? "/api").replace(/\/$/, "");
const REQUEST_CREDENTIALS: RequestCredentials = "include";
export const TRAINING_TASK_EVENT = "myweb1:training-task";

function authMessage(payload: unknown, fallback: string) {
  if (!payload || typeof payload !== "object") return fallback;
  const error = (payload as { error?: { detail?: string; message?: string } | string }).error;
  if (typeof error === "string") return error;
  return error?.detail || error?.message || fallback;
}

export async function fetchMe(signal?: AbortSignal): Promise<AuthUser | null> {
  const response = await fetch(`${API_BASE}/auth/me`, { signal, credentials: REQUEST_CREDENTIALS });
  if (response.status === 401) return null;
  const payload = await response.json().catch(() => null);
  if (!response.ok) throw new Error(authMessage(payload, `读取登录状态失败：HTTP ${response.status}`));
  return (payload as { user?: AuthUser }).user ?? null;
}

export async function login(username: string, password: string, signal?: AbortSignal): Promise<AuthUser> {
  const response = await fetch(`${API_BASE}/auth/login`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ username, password }),
    credentials: REQUEST_CREDENTIALS,
    signal
  });
  const payload = await response.json().catch(() => null);
  if (!response.ok) throw new Error(authMessage(payload, `登录失败：HTTP ${response.status}`));
  return (payload as { user: AuthUser }).user;
}

export async function register(username: string, password: string, signal?: AbortSignal): Promise<AuthUser> {
  const response = await fetch(`${API_BASE}/auth/register`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ username, password }),
    credentials: REQUEST_CREDENTIALS,
    signal
  });
  const payload = await response.json().catch(() => null);
  if (!response.ok) throw new Error(authMessage(payload, `注册失败：HTTP ${response.status}`));
  return (payload as { user: AuthUser }).user;
}

export async function logout(signal?: AbortSignal): Promise<void> {
  await fetch(`${API_BASE}/auth/logout`, {
    method: "POST",
    credentials: REQUEST_CREDENTIALS,
    signal
  });
}

function emitTrainingTaskEvent(
  modelType: ModelType,
  phase: "started" | "settled",
  payload?: ApiJson,
  error?: unknown
) {
  if (typeof window === "undefined") return;
  window.dispatchEvent(new CustomEvent(TRAINING_TASK_EVENT, {
    detail: {
      modelType,
      phase,
      taskId: typeof payload?.task_id === "string" ? payload.task_id : undefined,
      versionId: typeof payload?.version_id === "string" ? payload.version_id : undefined,
      error: error instanceof Error ? error.message : undefined
    }
  }));
}

export async function fetchHealth(signal?: AbortSignal): Promise<HealthResponse> {
  const response = await fetch(`${API_BASE}/health`, { signal, credentials: REQUEST_CREDENTIALS });
  if (!response.ok) {
    throw new Error(`HTTP ${response.status}`);
  }
  return response.json() as Promise<HealthResponse>;
}

export async function uploadDataset(file: File, signal?: AbortSignal): Promise<DataUploadResponse> {
  const formData = new FormData();
  formData.append("file", file);
  const response = await fetch(`${API_BASE}/data/upload`, {
    method: "POST",
    body: formData,
    credentials: REQUEST_CREDENTIALS,
    signal
  });
  const payload = await response.json().catch(() => null);
  if (!response.ok) {
    const message =
      payload?.error?.detail || payload?.error?.message || `上传失败：HTTP ${response.status}`;
    throw new Error(message);
  }
  return payload as DataUploadResponse;
}

export async function fetchDataProfile(sessionId: string, signal?: AbortSignal): Promise<DataUploadResponse> {
  const response = await fetch(`${API_BASE}/data/${sessionId}/profile`, { signal, credentials: REQUEST_CREDENTIALS });
  const payload = await response.json().catch(() => null);
  if (!response.ok) {
    const message =
      payload?.error?.detail || payload?.error?.message || `读取数据概览失败：HTTP ${response.status}`;
    throw new Error(message);
  }
  return payload as DataUploadResponse;
}

export async function fetchOutliers(sessionId: string, coefficient = 1.5, signal?: AbortSignal): Promise<OutlierMap> {
  const params = new URLSearchParams({ coefficient: String(coefficient) });
  const response = await fetch(`${API_BASE}/data/${sessionId}/outliers?${params.toString()}`, {
    signal,
    credentials: REQUEST_CREDENTIALS
  });
  const payload = await response.json().catch(() => null);
  if (!response.ok) {
    const message =
      payload?.error?.detail || payload?.error?.message || `读取异常值失败：HTTP ${response.status}`;
    throw new Error(message);
  }
  return ((payload as { outliers?: OutlierMap })?.outliers ?? {}) as OutlierMap;
}

export async function processData(
  sessionId: string,
  operations: Array<Record<string, unknown>>,
  signal?: AbortSignal
): Promise<DataUploadResponse> {
  const response = await fetch(`${API_BASE}/data/${sessionId}/process`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ operations }),
    credentials: REQUEST_CREDENTIALS,
    signal
  });
  const payload = await response.json().catch(() => null);
  if (!response.ok) {
    const message =
      payload?.error?.detail || payload?.error?.message || `数据处理失败：HTTP ${response.status}`;
    throw new Error(message);
  }
  return payload as DataUploadResponse;
}

export async function fetchProcessingHistory(sessionId: string, signal?: AbortSignal): Promise<ProcessingHistory> {
  const response = await fetch(`${API_BASE}/data/${sessionId}/history`, { signal, credentials: REQUEST_CREDENTIALS });
  const payload = await response.json().catch(() => null);
  if (!response.ok) {
    const message =
      payload?.error?.detail || payload?.error?.message || `读取处理历史失败：HTTP ${response.status}`;
    throw new Error(message);
  }
  return payload as ProcessingHistory;
}

export async function undoProcessing(sessionId: string, signal?: AbortSignal): Promise<DataUploadResponse> {
  const response = await fetch(`${API_BASE}/data/${sessionId}/undo`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: "{}",
    credentials: REQUEST_CREDENTIALS,
    signal
  });
  const payload = await response.json().catch(() => null);
  if (!response.ok) {
    const message =
      payload?.error?.detail || payload?.error?.message || `撤销失败：HTTP ${response.status}`;
    throw new Error(message);
  }
  return payload as DataUploadResponse;
}

export async function redoProcessing(sessionId: string, signal?: AbortSignal): Promise<DataUploadResponse> {
  const response = await fetch(`${API_BASE}/data/${sessionId}/redo`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: "{}",
    credentials: REQUEST_CREDENTIALS,
    signal
  });
  const payload = await response.json().catch(() => null);
  if (!response.ok) {
    const message =
      payload?.error?.detail || payload?.error?.message || `重做失败：HTTP ${response.status}`;
    throw new Error(message);
  }
  return payload as DataUploadResponse;
}

export async function saveProcessingPipeline(
  sessionId: string,
  name: string,
  operations?: Array<Record<string, unknown>>,
  signal?: AbortSignal
): Promise<{ pipeline: ProcessingPipeline; processing_history: ProcessingHistory }> {
  const response = await fetch(`${API_BASE}/data/${sessionId}/pipelines`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ name, operations }),
    credentials: REQUEST_CREDENTIALS,
    signal
  });
  const payload = await response.json().catch(() => null);
  if (!response.ok) {
    const message =
      payload?.error?.detail || payload?.error?.message || `保存流水线失败：HTTP ${response.status}`;
    throw new Error(message);
  }
  return payload as { pipeline: ProcessingPipeline; processing_history: ProcessingHistory };
}

export async function applyProcessingPipeline(
  sessionId: string,
  pipelineId: string,
  signal?: AbortSignal
): Promise<DataUploadResponse> {
  const response = await fetch(`${API_BASE}/data/${sessionId}/pipelines/${encodeURIComponent(pipelineId)}/apply`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: "{}",
    credentials: REQUEST_CREDENTIALS,
    signal
  });
  const payload = await response.json().catch(() => null);
  if (!response.ok) {
    const message =
      payload?.error?.detail || payload?.error?.message || `应用流水线失败：HTTP ${response.status}`;
    throw new Error(message);
  }
  return payload as DataUploadResponse;
}

export async function fetchVisualization(
  sessionId: string,
  payload: ApiJson,
  signal?: AbortSignal
): Promise<ApiJson> {
  const response = await fetch(`${API_BASE}/data/${sessionId}/visualize`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
    credentials: REQUEST_CREDENTIALS,
    signal
  });
  const data = await response.json().catch(() => null);
  if (!response.ok) {
    const message =
      data?.error?.detail || data?.error?.message || `图表数据生成失败：HTTP ${response.status}`;
    throw new Error(message);
  }
  return data as ApiJson;
}

async function postJson<T>(path: string, body: ApiJson, signal?: AbortSignal): Promise<T> {
  const response = await fetch(`${API_BASE}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
    credentials: REQUEST_CREDENTIALS,
    signal
  });
  const payload = await response.json().catch(() => null);
  if (!response.ok) {
    const message = payload?.error?.detail || payload?.error?.message || `请求失败：HTTP ${response.status}`;
    throw new Error(message);
  }
  return payload as T;
}

function wait(ms: number, signal?: AbortSignal) {
  return new Promise<void>((resolve, reject) => {
    if (signal?.aborted) {
      reject(new DOMException("Request aborted", "AbortError"));
      return;
    }
    const timeout = window.setTimeout(resolve, ms);
    signal?.addEventListener("abort", () => {
      window.clearTimeout(timeout);
      reject(new DOMException("Request aborted", "AbortError"));
    }, { once: true });
  });
}

export async function fetchTask(taskId: string, signal?: AbortSignal): Promise<TaskRecord> {
  const response = await fetch(`${API_BASE}/tasks/${encodeURIComponent(taskId)}`, { signal, credentials: REQUEST_CREDENTIALS });
  const payload = await response.json().catch(() => null);
  if (!response.ok) {
    const message = payload?.error?.detail || payload?.error?.message || `读取任务详情失败：HTTP ${response.status}`;
    throw new Error(message);
  }
  return payload as TaskRecord;
}

async function waitForTaskResult(taskId: string, signal?: AbortSignal): Promise<ApiJson> {
  while (true) {
    const task = await fetchTask(taskId, signal);
    if (task.status === "succeeded") {
      if (task.result_payload && typeof task.result_payload === "object") {
        return task.result_payload as ApiJson;
      }
      if (task.result && typeof task.result === "object") {
        return task.result as ApiJson;
      }
      return { task_id: task.task_id };
    }
    if (task.status === "failed") {
      throw new Error(task.error || "训练任务失败");
    }
    if (task.status === "cancelled") {
      throw new Error(task.error || "训练任务已取消");
    }
    await wait(750, signal);
  }
}

export async function trainModel(modelType: ModelType, payload: ApiJson, signal?: AbortSignal): Promise<ApiJson> {
  emitTrainingTaskEvent(modelType, "started");
  try {
    const path = `/${modelType}/train?async=1`;
    const initial = await postJson<ApiJson>(path, payload, signal);
    const taskId = typeof initial.task_id === "string" ? initial.task_id : "";
    if (initial.async === true && taskId) {
      emitTrainingTaskEvent(modelType, "started", initial);
    }
    const result = initial.async === true && taskId ? await waitForTaskResult(taskId, signal) : initial;
    emitTrainingTaskEvent(modelType, "settled", result);
    return result;
  } catch (error) {
    emitTrainingTaskEvent(modelType, "settled", undefined, error);
    throw error;
  }
}

export async function fetchClusteringElbow(payload: ApiJson, signal?: AbortSignal): Promise<ApiJson> {
  return postJson<ApiJson>("/clustering/elbow", payload, signal);
}

export async function predictModel(modelType: ModelType, payload: ApiJson, signal?: AbortSignal): Promise<ApiJson> {
  return postJson<ApiJson>(`/${modelType}/predict`, payload, signal);
}

export async function batchPredictModel(modelType: ModelType, payload: ApiJson, signal?: AbortSignal): Promise<ApiJson> {
  return postJson<ApiJson>(`/${modelType}/batch_predict`, payload, signal);
}

export async function fetchModelVersions(modelType: ModelType, signal?: AbortSignal): Promise<ModelVersionsResponse> {
  const response = await fetch(`${API_BASE}/${modelType}/versions`, { signal, credentials: REQUEST_CREDENTIALS });
  const payload = await response.json().catch(() => null);
  if (!response.ok) {
    const message = payload?.error?.detail || payload?.error?.message || `读取模型版本失败：HTTP ${response.status}`;
    throw new Error(message);
  }
  return payload as ModelVersionsResponse;
}

export async function fetchModelStatus(modelType: ModelType, signal?: AbortSignal): Promise<ApiJson> {
  const response = await fetch(`${API_BASE}/${modelType}/status`, { signal, credentials: REQUEST_CREDENTIALS });
  const payload = await response.json().catch(() => null);
  if (!response.ok) {
    const message = payload?.error?.detail || payload?.error?.message || `读取模型状态失败：HTTP ${response.status}`;
    throw new Error(message);
  }
  return payload as ApiJson;
}

export async function fetchTasks(signal?: AbortSignal): Promise<TaskListResponse> {
  const response = await fetch(`${API_BASE}/tasks`, { signal, credentials: REQUEST_CREDENTIALS });
  const payload = await response.json().catch(() => null);
  if (!response.ok) {
    const message = payload?.error?.detail || payload?.error?.message || `读取任务状态失败：HTTP ${response.status}`;
    throw new Error(message);
  }
  return payload as TaskListResponse;
}

export async function cancelTask(taskId: string, signal?: AbortSignal): Promise<TaskRecord> {
  const response = await fetch(`${API_BASE}/tasks/${encodeURIComponent(taskId)}/cancel`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: "{}",
    credentials: REQUEST_CREDENTIALS,
    signal
  });
  const payload = await response.json().catch(() => null);
  if (!response.ok) {
    const message = payload?.error?.detail || payload?.error?.message || `取消任务失败：HTTP ${response.status}`;
    throw new Error(message);
  }
  return payload as TaskRecord;
}

export async function fetchModelVersionDetail(modelType: ModelType, versionId: string, signal?: AbortSignal): Promise<ApiJson> {
  const response = await fetch(`${API_BASE}/${modelType}/version/${encodeURIComponent(versionId)}`, { signal, credentials: REQUEST_CREDENTIALS });
  const payload = await response.json().catch(() => null);
  if (!response.ok) {
    const message = payload?.error?.detail || payload?.error?.message || `读取模型版本失败：HTTP ${response.status}`;
    throw new Error(message);
  }
  return payload as ApiJson;
}

export async function activateModelVersion(modelType: ModelType, versionId: string, signal?: AbortSignal): Promise<ApiJson> {
  return postJson<ApiJson>(`/${modelType}/activate`, { version_id: versionId }, signal);
}

export async function deleteModelVersion(modelType: ModelType, versionId: string, signal?: AbortSignal): Promise<ApiJson> {
  const response = await fetch(`${API_BASE}/${modelType}/version/${encodeURIComponent(versionId)}`, {
    method: "DELETE",
    credentials: REQUEST_CREDENTIALS,
    signal
  });
  const payload = await response.json().catch(() => null);
  if (!response.ok) {
    const message = payload?.error?.detail || payload?.error?.message || `删除模型版本失败：HTTP ${response.status}`;
    throw new Error(message);
  }
  return payload as ApiJson;
}

export async function clearModel(modelType: ModelType, signal?: AbortSignal): Promise<ApiJson> {
  return postJson<ApiJson>(`/${modelType}/clear`, {}, signal);
}

function responseErrorMessage(payload: unknown, fallback: string) {
  if (!payload || typeof payload !== "object") return fallback;
  const error = (payload as { error?: { detail?: string; message?: string } | string }).error;
  if (typeof error === "string") return error;
  return error?.detail || error?.message || fallback;
}

function parseSseBuffer(buffer: string, onEvent: (event: LlmStreamEvent) => void) {
  const events = buffer.split(/\n\n/);
  const rest = events.pop() ?? "";
  for (const eventText of events) {
    const dataLines = eventText
      .split(/\r?\n/)
      .filter((line) => line.startsWith("data: "))
      .map((line) => line.slice(6));
    if (!dataLines.length) continue;
    let parsed: LlmStreamEvent;
    try {
      parsed = JSON.parse(dataLines.join("\n")) as LlmStreamEvent;
    } catch {
      // Ignore malformed partial SSE events instead of breaking a live stream.
      continue;
    }
    onEvent(parsed);
  }
  return rest;
}

export async function streamLlmChat(
  mode: "direct" | "agent",
  payload: ApiJson,
  onEvent: (event: LlmStreamEvent) => void,
  signal?: AbortSignal
): Promise<void> {
  const path = mode === "agent" ? "/llm/agent" : "/llm/chat";
  const response = await fetch(`${API_BASE}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
    credentials: REQUEST_CREDENTIALS,
    signal
  });

  if (!response.ok) {
    const payload = await response.json().catch(() => null);
    throw new Error(responseErrorMessage(payload, `LLM 请求失败：HTTP ${response.status}`));
  }
  if (!response.body) {
    throw new Error("当前浏览器不支持流式响应。");
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    buffer = parseSseBuffer(buffer, onEvent);
  }

  buffer += decoder.decode();
  parseSseBuffer(`${buffer}\n\n`, onEvent);
}

export { API_BASE };
