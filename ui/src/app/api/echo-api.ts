const API_BASE = (import.meta.env.VITE_ECHO_API_BASE_URL || "http://127.0.0.1:8765/api").replace(/\/$/, "");

export type HomeSummary = {
  counts: {
    meetings_processed: number;
    pending_review: number;
    queue_running: number;
    queue_waiting: number;
    action_items_open: number;
  };
  active_jobs: Array<{
    id: string;
    meeting_id: string | null;
    source_type: string;
    title: string;
    template_name: string | null;
    stage: string;
    progress: number;
    status: string;
    error_code?: string;
    error_message?: string;
    created_at: string;
    updated_at: string;
    events: Array<{
      stage: string;
      progress: number;
      message: string;
      level: "info" | "warning" | "error";
      created_at: string;
    }>;
  }>;
  ready_for_review: Array<Record<string, unknown>>;
  recent_meetings: Array<{
    id: string;
    title: string;
    meeting_type: string;
    status: string;
    source_label: string;
    created_at: string;
    updated_at: string;
    decisions_count: number;
    action_items_count: number;
    mom_version: number;
  }>;
  open_action_items: Array<{
    id: string;
    description: string;
    owner: string;
    due_date: string | null;
    status: string;
    project: string;
    source_meeting: string;
  }>;
};

export type EchoSettings = Record<string, any>;

export type BackendTestResponse = {
  backend_kind: string;
  ok: boolean;
  message: string;
};

export type SetupStatus = {
  ollama: { installed: boolean; path: string | null; model_name: string };
  browser: { installed: boolean; name: string | null; path: string | null };
  ffmpeg: { installed: boolean; path: string | null };
  guidance: Record<string, { download_url: string; install: string }>;
};

export type MeetingSummary = {
  id: string;
  title: string;
  meeting_type: string;
  project: string;
  host: string;
  source_type: string;
  source_label: string;
  status: string;
  confidentiality: string;
  created_at: string;
  updated_at: string;
  decisions_count: number;
  action_items_count: number;
  mom_version: number;
};

export type MomVersion = {
  id: string;
  version_number: number;
  title: string;
  summary: string;
  content_markdown: string;
  status: string;
  backend_kind: string;
  created_at: string;
  approved_at: string | null;
};

export type MeetingDetail = {
  meeting: MeetingSummary & {
    tags_json?: string;
    meeting_occurred_at?: string | null;
    mom_generated_at?: string | null;
    source_info?: {
      label: string;
      display: string;
      href: string;
      kind: string;
    };
  };
  latest_mom: MomVersion | null;
  mom_versions: MomVersion[];
  transcript: {
    id: string;
    text: string;
    source: string;
    language: string;
    created_at: string;
  } | null;
  decisions: Array<{
    id: string;
    description: string;
    context: string;
    owner: string;
    source_ref: string;
    created_at: string;
  }>;
  action_items: Array<{
    id: string;
    description: string;
    owner: string;
    due_date: string | null;
    status: string;
    project: string;
    confidence: number;
    source_ref: string;
    created_at: string;
  }>;
  risks: Array<{
    id: string;
    description: string;
    severity: string;
    mitigation: string;
    source_ref: string;
    created_at: string;
  }>;
  jobs: Array<{
    id: string;
    stage: string;
    progress: number;
    status: string;
    error_code: string;
    error_message: string;
    created_at: string;
    updated_at: string;
    events: Array<{
      stage: string;
      progress: number;
      message: string;
      level: "info" | "warning" | "error";
      created_at: string;
    }>;
  }>;
};

export type RegenerateMeetingRequest = {
  template_name: string;
  backend_kind?: string;
  run_now?: boolean;
};

export type JobCreateRequest = {
  source_type: "url" | "upload" | "transcript";
  sources: string[];
  meeting_type: string;
  template_name?: string;
  confidentiality?: string;
  project?: string;
  host?: string;
  run_now?: boolean;
};

export type EchoTemplate = {
  id: string;
  name: string;
  meeting_type: string;
  description: string;
  sections: string[];
  system_prompt: string;
  is_default: boolean;
  is_locked: boolean;
  created_at: string;
  updated_at: string;
};

export type TemplateCreateRequest = {
  name: string;
  meeting_type?: string;
  description?: string;
  sections?: string[];
  system_prompt?: string;
  is_default?: boolean;
};

export type TemplateUpdateRequest = Partial<TemplateCreateRequest>;

export type TemplatePreset = {
  meeting_type: string;
  description: string;
  sections: string[];
  system_prompt: string;
};

export type TemplatePromptResponse = {
  template_id: string;
  template_name: string;
  prompt: string;
  effective_prompt?: string;
};

export type TemplateVersion = {
  id: string;
  template_id: string;
  version_number: number;
  name: string;
  meeting_type: string;
  description: string;
  sections: string[];
  system_prompt: string;
  change_note: string;
  created_at: string;
};

async function request<T>(path: string): Promise<T> {
  const response = await fetch(`${API_BASE}${path}`, {
    headers: { Accept: "application/json" },
  });
  if (!response.ok) {
    throw new Error(`ECHO API ${response.status}: ${response.statusText}`);
  }
  return response.json() as Promise<T>;
}

async function send<T>(path: string, method: "POST" | "PUT" | "PATCH" | "DELETE", body?: unknown): Promise<T> {
  const options: RequestInit = {
    method,
    headers: {
      Accept: "application/json",
    },
  };
  if (body) {
    (options.headers as any)["Content-Type"] = "application/json";
    options.body = JSON.stringify(body);
  }
  const response = await fetch(`${API_BASE}${path}`, options);
  if (!response.ok) {
    let detail = `${response.status}: ${response.statusText}`;
    try {
      const data = await response.json();
      detail = data.detail || detail;
    } catch {
      // Keep HTTP status fallback.
    }
    throw new Error(detail);
  }
  return response.json() as Promise<T>;
}

async function command<T>(path: string): Promise<T> {
  const response = await fetch(`${API_BASE}${path}`, {
    method: "POST",
    headers: { Accept: "application/json" },
  });
  if (!response.ok) {
    let detail = `${response.status}: ${response.statusText}`;
    try {
      const data = await response.json();
      detail = data.detail || detail;
    } catch {
      // Keep HTTP status fallback.
    }
    throw new Error(detail);
  }
  return response.json() as Promise<T>;
}

export function getHomeSummary() {
  return request<HomeSummary>("/home");
}

export function getMeetings() {
  return request<{ meetings: MeetingSummary[] }>("/meetings");
}

export function getMeetingDetail(meetingId: string) {
  return request<MeetingDetail>(`/meetings/${encodeURIComponent(meetingId)}`);
}

export function regenerateMeeting(meetingId: string, payload: RegenerateMeetingRequest) {
  return send<{ job: Record<string, unknown>; started: Record<string, unknown> | null }>(
    `/meetings/${encodeURIComponent(meetingId)}/regenerate`,
    "POST",
    payload,
  );
}

export function meetingExportUrl(meetingId: string, exportType: "pdf" | "email" | "text" | "html") {
  return `${API_BASE}/meetings/${encodeURIComponent(meetingId)}/exports/${exportType}`;
}

export function getSettings() {
  return request<EchoSettings>("/settings");
}

export function getHealth() {
  return request<any>("/health");
}

export function updateSettings(values: EchoSettings) {
  return send<EchoSettings>("/settings", "PUT", { values });
}

export function testBackendConnection(backend_kind: string, values: EchoSettings) {
  return send<BackendTestResponse>("/settings/test-backend", "POST", { backend_kind, values });
}

export function getSetupStatus() {
  return request<SetupStatus>("/settings/setup");
}

export function pullOllamaModel(model?: string) {
  return send<{ ok: boolean; message: string; model: string }>("/settings/ollama/pull-model", "POST", { model });
}

export function getTemplates() {
  return request<{ templates: EchoTemplate[] }>("/templates");
}

export function getTemplate(templateId: string) {
  return request<EchoTemplate>(`/templates/${encodeURIComponent(templateId)}`);
}

export function getTemplatePresets() {
  return request<{ presets: TemplatePreset[] }>("/templates/presets");
}

export function getTemplatePrompt(templateId: string) {
  return request<TemplatePromptResponse>(`/templates/${encodeURIComponent(templateId)}/prompt`);
}

export function getTemplateVersions(templateId: string) {
  return request<{ versions: TemplateVersion[] }>(`/templates/${encodeURIComponent(templateId)}/versions`);
}

export function restoreTemplateVersion(templateId: string, versionId: string) {
  return send<EchoTemplate>(
    `/templates/${encodeURIComponent(templateId)}/versions/${encodeURIComponent(versionId)}/restore`,
    "POST",
  );
}

export function createTemplate(payload: TemplateCreateRequest) {
  return send<EchoTemplate>("/templates", "POST", payload);
}

export function updateTemplate(templateId: string, payload: TemplateUpdateRequest) {
  return send<EchoTemplate>(`/templates/${encodeURIComponent(templateId)}`, "PATCH", payload);
}

export function duplicateTemplate(templateId: string) {
  return send<EchoTemplate>(`/templates/${encodeURIComponent(templateId)}/duplicate`, "POST");
}

export function deleteTemplate(templateId: string) {
  return send<{ status: string }>(`/templates/${encodeURIComponent(templateId)}`, "DELETE");
}

export function testTemplate(payload: { transcript_text: string; template_name: string; backend_kind?: string }) {
  return send<{ output: string }>("/sandbox/test-template", "POST", payload);
}

export function createJobs(payload: JobCreateRequest) {
  return send<{ jobs: Array<Record<string, unknown>> }>("/jobs", "POST", payload);
}

export function startJob(jobId: string) {
  return command<{ job: Record<string, unknown> }>(`/jobs/${encodeURIComponent(jobId)}/start`);
}

export function processNextJob() {
  return command<{ started: Record<string, unknown> | null; message: string }>("/jobs/process-next");
}

export function processAvailableJobs() {
  return command<{ started: Array<Record<string, unknown>>; message: string }>("/jobs/process-available");
}

export function retryFailedJobs() {
  return command<{ started: Array<Record<string, unknown>>; message: string }>("/jobs/retry-failed");
}

export async function createUploadJobs(payload: Omit<JobCreateRequest, "source_type" | "sources"> & { files: File[] }) {
  const form = new FormData();
  payload.files.forEach((file) => form.append("files", file));
  form.append("meeting_type", payload.meeting_type);
  form.append("template_name", payload.template_name || "Executive MoM");
  form.append("confidentiality", payload.confidentiality || "Internal");
  form.append("project", payload.project || "");
  form.append("host", payload.host || "");
  form.append("run_now", String(payload.run_now ?? true));

  const response = await fetch(`${API_BASE}/jobs/upload`, {
    method: "POST",
    headers: { Accept: "application/json" },
    body: form,
  });
  if (!response.ok) {
    let detail = `${response.status}: ${response.statusText}`;
    try {
      const data = await response.json();
      detail = data.detail || detail;
    } catch {
      // Keep HTTP status fallback.
    }
    throw new Error(detail);
  }
  return response.json() as Promise<{ jobs: Array<Record<string, unknown>> }>;
}
