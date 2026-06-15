import type {
  ApiJson,
  DataUploadResponse,
  HealthResponse,
  LlmStreamEvent,
  ModelType,
  ModelVersionsResponse
} from "../types";

const API_BASE = (import.meta.env.VITE_API_BASE ?? "/api").replace(/\/$/, "");

export async function fetchHealth(signal?: AbortSignal): Promise<HealthResponse> {
  const response = await fetch(`${API_BASE}/health`, { signal });
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
  const response = await fetch(`${API_BASE}/data/${sessionId}/profile`, { signal });
  const payload = await response.json().catch(() => null);
  if (!response.ok) {
    const message =
      payload?.error?.detail || payload?.error?.message || `读取数据概览失败：HTTP ${response.status}`;
    throw new Error(message);
  }
  return payload as DataUploadResponse;
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

export async function fetchVisualization(
  sessionId: string,
  payload: ApiJson,
  signal?: AbortSignal
): Promise<ApiJson> {
  const response = await fetch(`${API_BASE}/data/${sessionId}/visualize`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
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
    signal
  });
  const payload = await response.json().catch(() => null);
  if (!response.ok) {
    const message = payload?.error?.detail || payload?.error?.message || `请求失败：HTTP ${response.status}`;
    throw new Error(message);
  }
  return payload as T;
}

export async function trainModel(modelType: ModelType, payload: ApiJson, signal?: AbortSignal): Promise<ApiJson> {
  return postJson<ApiJson>(`/${modelType}/train`, payload, signal);
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
  const response = await fetch(`${API_BASE}/${modelType}/versions`, { signal });
  const payload = await response.json().catch(() => null);
  if (!response.ok) {
    const message = payload?.error?.detail || payload?.error?.message || `读取模型版本失败：HTTP ${response.status}`;
    throw new Error(message);
  }
  return payload as ModelVersionsResponse;
}

export async function fetchModelStatus(modelType: ModelType, signal?: AbortSignal): Promise<ApiJson> {
  const response = await fetch(`${API_BASE}/${modelType}/status`, { signal });
  const payload = await response.json().catch(() => null);
  if (!response.ok) {
    const message = payload?.error?.detail || payload?.error?.message || `读取模型状态失败：HTTP ${response.status}`;
    throw new Error(message);
  }
  return payload as ApiJson;
}

export async function fetchModelVersionDetail(modelType: ModelType, versionId: string, signal?: AbortSignal): Promise<ApiJson> {
  const response = await fetch(`${API_BASE}/${modelType}/version/${encodeURIComponent(versionId)}`, { signal });
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
