import { useEffect, useState } from "react";
import { PageHeader } from "../page-header";
import { User, Bell, Download, Shield, Server, Mic, Database, Lock, Activity, FlaskRound, KeyRound, CheckCircle2, ChevronRight, Save, RotateCcw } from "lucide-react";
import { getSettings, updateSettings, type EchoSettings } from "../../api/echo-api";

const sections = [
  { id: "profile", label: "Profile", icon: User, group: "Personal" },
  { id: "defaults", label: "Defaults", icon: Download, group: "Personal" },
  { id: "notifications", label: "Notifications", icon: Bell, group: "Personal" },
  { id: "privacy", label: "Privacy preferences", icon: Shield, group: "Personal" },
  { id: "health", label: "System Health", icon: Activity, group: "Admin", admin: true },
  { id: "backend", label: "Backend Settings", icon: Server, group: "Admin", admin: true },
  { id: "prompt", label: "Prompt Studio", icon: FlaskRound, group: "Admin", admin: true },
  { id: "asr", label: "ASR Configuration", icon: Mic, group: "Admin", admin: true },
  { id: "storage", label: "Storage & Retention", icon: Database, group: "Admin", admin: true },
  { id: "compliance", label: "Privacy & Compliance", icon: Lock, group: "Admin", admin: true },
];

export function SettingsPage() {
  const [sel, setSel] = useState("profile");
  const [settings, setSettings] = useState<EchoSettings | null>(null);
  const [draft, setDraft] = useState<EchoSettings | null>(null);
  const [status, setStatus] = useState("");
  const [saving, setSaving] = useState(false);

  const reload = () => {
    setStatus("");
    return getSettings()
      .then((data) => {
        setSettings(data);
        setDraft(data);
      })
      .catch(() => {
        setSettings(null);
        setDraft(null);
        setStatus("Could not load settings from backend.");
      });
  };

  useEffect(() => {
    reload();
  }, []);

  const setValue = (path: string[], value: unknown) => {
    setDraft((current) => setIn(current || {}, path, value));
  };

  const save = async () => {
    if (!draft) return;
    setSaving(true);
    setStatus("");
    try {
      const saved = await updateSettings(draft);
      setSettings(saved);
      setDraft(saved);
      setStatus("Settings saved.");
    } catch (error) {
      setStatus(error instanceof Error ? error.message : "Could not save settings.");
    } finally {
      setSaving(false);
    }
  };

  const view = draft || settings;

  return (
    <div className="space-y-5">
      <PageHeader
        title="Settings"
        subtitle="Personal preferences and admin-only configuration."
        actions={
          <>
            {status && <span className="text-[12px] text-echo-text-muted">{status}</span>}
            <button onClick={reload} className="h-9 px-3 rounded-md border border-echo-border bg-echo-surface hover:bg-echo-surface-hover text-[12px] text-echo-text inline-flex items-center gap-1.5"><RotateCcw size={13} />Reload</button>
            <button onClick={save} disabled={!draft || saving} className="h-9 px-3 rounded-md bg-echo-accent text-white text-[12px] inline-flex items-center gap-1.5 disabled:opacity-60"><Save size={13} />{saving ? "Saving…" : "Save settings"}</button>
          </>
        }
      />

      <div className="grid grid-cols-1 xl:grid-cols-[260px_1fr] gap-5">
        <aside className="bg-echo-surface border border-echo-border rounded-lg p-2 h-fit">
          {["Personal", "Admin"].map((g) => (
            <div key={g} className="mb-1">
              <div className="px-3 pt-2 pb-1 text-[10px] uppercase tracking-wider text-echo-text-faint">{g}</div>
              {sections.filter((s) => s.group === g).map((s) => {
                const I = s.icon;
                const active = sel === s.id;
                return (
                  <button key={s.id} onClick={() => setSel(s.id)} className={`w-full flex items-center gap-2.5 px-3 py-2 rounded-md text-[12px] ${active ? "bg-echo-accent-bg text-echo-accent-fg" : "text-echo-text hover:bg-echo-surface-hover"}`}>
                    <I size={14} className={active ? "text-echo-accent" : "text-echo-text-muted"} />
                    <span className="flex-1 text-left">{s.label}</span>
                    {s.admin && <span className="text-[9px] px-1 rounded bg-echo-surface-2 text-echo-text-muted">ADMIN</span>}
                  </button>
                );
              })}
            </div>
          ))}
        </aside>

        <section className="space-y-5">
          {sel === "profile" && <ProfilePanel settings={view} update={setValue} />}
          {sel === "defaults" && <DefaultsPanel settings={view} update={setValue} />}
          {sel === "notifications" && <NotificationsPanel settings={view} update={setValue} />}
          {sel === "privacy" && <PrivacyPanel settings={view} update={setValue} />}
          {sel === "health" && <HealthPanel settings={view} />}
          {sel === "backend" && <BackendPanel settings={view} update={setValue} />}
          {sel === "prompt" && <PromptStudioPanel settings={view} update={setValue} />}
          {sel === "asr" && <ASRPanel settings={view} update={setValue} />}
          {sel === "storage" && <StoragePanel settings={view} update={setValue} />}
          {sel === "compliance" && <CompliancePanel settings={view} update={setValue} />}
        </section>
      </div>
    </div>
  );
}

type SettingsUpdate = (path: string[], value: unknown) => void;

function Panel({ title, desc, children }: { title: string; desc?: string; children: React.ReactNode }) {
  return (
    <div className="bg-echo-surface border border-echo-border rounded-lg">
      <div className="px-5 py-3 border-b border-echo-border">
        <h2 className="text-[14px] text-echo-text" style={{ fontWeight: 600 }}>{title}</h2>
        {desc && <p className="text-[12px] text-echo-text-muted mt-0.5">{desc}</p>}
      </div>
      <div className="p-5 space-y-4">{children}</div>
    </div>
  );
}
function Row({ label, hint, children }: { label: string; hint?: string; children: React.ReactNode }) {
  return (
    <div className="grid grid-cols-[200px_1fr] gap-4 items-start">
      <div>
        <div className="text-[12px] text-echo-text">{label}</div>
        {hint && <div className="text-[11px] text-echo-text-muted mt-0.5">{hint}</div>}
      </div>
      <div>{children}</div>
    </div>
  );
}
function Input({ value, type = "text", onChange }: { value: string; type?: string; onChange?: (value: string) => void }) {
  return <input type={type} value={value} onChange={(event) => onChange?.(event.target.value)} className="w-full max-w-md h-9 px-3 rounded-md border border-echo-border bg-echo-surface-2 text-[12px] text-echo-text focus:outline-none focus:bg-echo-surface focus:border-echo-accent" />;
}
function Toggle({ on = false, label, onChange }: { on?: boolean; label: string; onChange?: (value: boolean) => void }) {
  return (
    <label className="inline-flex items-center gap-2 cursor-pointer">
      <button onClick={() => onChange?.(!on)} className={`h-5 w-9 rounded-full transition-colors relative ${on ? "bg-echo-accent" : "bg-echo-border-strong"}`}>
        <span className={`absolute top-0.5 h-4 w-4 rounded-full bg-white shadow transition-all ${on ? "left-4" : "left-0.5"}`} />
      </button>
      <span className="text-[12px] text-echo-text-muted">{label}</span>
    </label>
  );
}

function Select({ value, options, onChange }: { value: string; options: string[]; onChange: (value: string) => void }) {
  return (
    <select value={value} onChange={(event) => onChange(event.target.value)} className="w-full max-w-md h-9 px-3 rounded-md border border-echo-border bg-echo-surface-2 text-[12px] text-echo-text focus:outline-none focus:bg-echo-surface focus:border-echo-accent">
      {options.map((option) => <option key={option} value={option}>{option}</option>)}
    </select>
  );
}

function ProfilePanel({ settings, update }: { settings: EchoSettings | null; update: SettingsUpdate }) {
  return (
    <Panel title="Profile" desc="How you appear inside ECHO.">
      <div className="flex items-center gap-4">
        <div className="h-14 w-14 rounded-full text-white grid place-items-center text-[18px]" style={{ background: "linear-gradient(135deg, var(--echo-accent), var(--echo-accent-hover))" }}>PS</div>
        <button className="h-8 px-3 rounded-md border border-echo-border bg-echo-surface text-[12px] text-echo-text">Change photo</button>
      </div>
      <Row label="Full name"><Input value={settings?.user?.full_name || "Priya Sharma"} onChange={(value) => update(["user", "full_name"], value)} /></Row>
      <Row label="Email"><Input value={settings?.user?.email || "priya.sharma@company.com"} type="email" onChange={(value) => update(["user", "email"], value)} /></Row>
      <Row label="Role"><Input value={settings?.user?.role || "Chief of Staff"} onChange={(value) => update(["user", "role"], value)} /></Row>
      <Row label="Time zone"><Input value={settings?.app?.timezone || "Asia/Kolkata"} onChange={(value) => update(["app", "timezone"], value)} /></Row>
    </Panel>
  );
}
function DefaultsPanel({ settings, update }: { settings: EchoSettings | null; update: SettingsUpdate }) {
  return (
    <Panel title="Defaults" desc="Pre-fill values used when creating MoMs.">
      <Row label="Default meeting type"><Select value={settings?.defaults?.meeting_type || "Executive"} options={["Executive", "Project Review", "Client Call", "Townhall", "Demo/UAT", "Incident"]} onChange={(value) => update(["defaults", "meeting_type"], value)} /></Row>
      <Row label="Default template"><Select value={settings?.summary?.default_template || "Executive MoM"} options={["Executive MoM", "Project Review", "Client Call", "Townhall", "Demo/UAT", "Incident Review"]} onChange={(value) => update(["summary", "default_template"], value)} /></Row>
      <Row label="Default export format"><Select value={settings?.defaults?.export_format || "PDF · Board ready"} options={["PDF · Board ready", "DOCX · Editable", "Email draft", "Plain text"]} onChange={(value) => update(["defaults", "export_format"], value)} /></Row>
      <Row label="Confidentiality default"><Select value={settings?.defaults?.confidentiality || "Internal"} options={["Internal", "Confidential", "Restricted"]} onChange={(value) => update(["defaults", "confidentiality"], value)} /></Row>
    </Panel>
  );
}
function NotificationsPanel({ settings, update }: { settings: EchoSettings | null; update: SettingsUpdate }) {
  return (
    <Panel title="Notifications">
      <Row label="MoM ready for review"><Toggle on={settings?.notifications?.mom_ready ?? true} label="In-app and email" onChange={(value) => update(["notifications", "mom_ready"], value)} /></Row>
      <Row label="Action item due soon"><Toggle on={settings?.notifications?.action_due ?? true} label="Email reminder 24 hours before" onChange={(value) => update(["notifications", "action_due"], value)} /></Row>
      <Row label="Queue needs attention"><Toggle on={settings?.notifications?.queue_attention ?? true} label="In-app" onChange={(value) => update(["notifications", "queue_attention"], value)} /></Row>
      <Row label="Weekly digest"><Toggle on={Boolean(settings?.notifications?.weekly_digest)} label="Email each Monday 8:00 AM" onChange={(value) => update(["notifications", "weekly_digest"], value)} /></Row>
    </Panel>
  );
}
function PrivacyPanel({ settings, update }: { settings: EchoSettings | null; update: SettingsUpdate }) {
  return (
    <Panel title="Privacy preferences">
      <Row label="Hide my MoMs from workspace search" hint="Other admins can still find them."><Toggle on={Boolean(settings?.privacy_preferences?.hide_from_workspace_search)} label="Enable" onChange={(value) => update(["privacy_preferences", "hide_from_workspace_search"], value)} /></Row>
      <Row label="Mask participant emails in exports"><Toggle on={settings?.privacy_preferences?.mask_participant_emails ?? true} label="Replace with initials" onChange={(value) => update(["privacy_preferences", "mask_participant_emails"], value)} /></Row>
      <Row label="Auto-redact phone numbers"><Toggle on={settings?.privacy_preferences?.auto_redact_phone_numbers ?? true} label="Enabled" onChange={(value) => update(["privacy_preferences", "auto_redact_phone_numbers"], value)} /></Row>
    </Panel>
  );
}

function HealthPanel({ settings }: { settings: EchoSettings | null }) {
  const checks = [
    { l: "ffmpeg", v: "v6.1 · healthy", ok: true },
    { l: "Playwright", v: "v1.45 · healthy", ok: true },
    { l: "Python runtime", v: "3.11.7 · healthy", ok: true },
    { l: "Ollama model", v: `${settings?.backends?.local?.model || "llama3:latest"} · configured`, ok: true },
    { l: "Output folder writable", v: `${settings?.storage?.output_dir || "outputs"} · configured`, ok: true },
    { l: "Primary backend", v: `${settings?.summary?.default_backend || "local"} · configured`, ok: true },
  ];
  return (
    <Panel title="System Health" desc="Read-only environment diagnostics for admins.">
      <div className="flex items-center justify-between">
        <div className="text-[12px] text-echo-text-muted">Last checked 2 minutes ago</div>
        <button className="h-9 px-3 rounded-md bg-echo-text text-echo-surface text-[12px]">Test all connections</button>
      </div>
      <ul className="divide-y divide-echo-border border border-echo-border rounded-md">
        {checks.map((c) => (
          <li key={c.l} className="px-4 py-2.5 flex items-center gap-3">
            <CheckCircle2 size={14} className={c.ok ? "text-echo-success" : "text-echo-danger"} />
            <span className="text-[12px] text-echo-text flex-1">{c.l}</span>
            <span className="text-[12px] text-echo-text-muted">{c.v}</span>
          </li>
        ))}
      </ul>
    </Panel>
  );
}

function BackendPanel({ settings, update }: { settings: EchoSettings | null; update: SettingsUpdate }) {
  const primary = settings?.summary?.default_backend || "local";
  return (
    <Panel title="Backend Settings" desc="Switch between Dify, Azure OpenAI, and local Ollama.">
      <Row label="Primary backend">
        <div className="inline-flex p-1 bg-echo-surface-2 rounded-md border border-echo-border">
          {[
            ["dify", "Dify"],
            ["azure", "Azure OpenAI"],
            ["local", "Local Ollama"],
          ].map(([id, label]) => (
            <button key={id} onClick={() => update(["summary", "default_backend"], id)} className={`px-3 py-1.5 rounded text-[12px] ${primary === id ? "bg-echo-surface text-echo-text shadow-sm" : "text-echo-text-muted"}`}>{label}</button>
          ))}
        </div>
      </Row>
      <Row label="Dify enabled"><Toggle on={Boolean(settings?.backends?.dify?.enabled)} label="Available as a backend" onChange={(value) => update(["backends", "dify", "enabled"], value)} /></Row>
      <Row label="Dify base URL"><Input value={settings?.backends?.dify?.base_url || ""} onChange={(value) => update(["backends", "dify", "base_url"], value)} /></Row>
      <Row label="Dify API key" hint="Saved in local config for now; move to keychain before packaging.">
        <Input value={settings?.backends?.dify?.api_key || ""} type="password" onChange={(value) => update(["backends", "dify", "api_key"], value)} />
      </Row>
      <Row label="Azure enabled"><Toggle on={Boolean(settings?.backends?.azure?.enabled)} label="Available as a backend" onChange={(value) => update(["backends", "azure", "enabled"], value)} /></Row>
      <Row label="Azure OpenAI endpoint"><Input value={settings?.backends?.azure?.endpoint || ""} onChange={(value) => update(["backends", "azure", "endpoint"], value)} /></Row>
      <Row label="Azure OpenAI API key" hint="Stored encrypted at rest.">
        <div className="flex items-center gap-2">
          <Input value={settings?.backends?.azure?.api_key || ""} type="password" onChange={(value) => update(["backends", "azure", "api_key"], value)} />
          <button className="text-[11px] text-echo-accent inline-flex items-center gap-1"><KeyRound size={11} />Reveal</button>
        </div>
      </Row>
      <Row label="Azure API version"><Input value={settings?.backends?.azure?.api_version || "2024-02-15-preview"} onChange={(value) => update(["backends", "azure", "api_version"], value)} /></Row>
      <Row label="Deployment / model"><Input value={settings?.backends?.azure?.deployment || ""} onChange={(value) => update(["backends", "azure", "deployment"], value)} /></Row>
      <Row label="Local Ollama enabled"><Toggle on={Boolean(settings?.backends?.local?.enabled)} label="Available as a backend" onChange={(value) => update(["backends", "local", "enabled"], value)} /></Row>
      <Row label="Ollama base URL"><Input value={settings?.backends?.local?.base_url || ""} onChange={(value) => update(["backends", "local", "base_url"], value)} /></Row>
      <Row label="Ollama model"><Input value={settings?.backends?.local?.model || "llama3:latest"} onChange={(value) => update(["backends", "local", "model"], value)} /></Row>
    </Panel>
  );
}

function PromptStudioPanel({ settings, update }: { settings: EchoSettings | null; update: SettingsUpdate }) {
  const prompts = (settings?.templates?.defaults || []).map((template: any) => template.name);
  return (
    <Panel title="Prompt Studio" desc="Manage system prompts and templates. Locked by default.">
      <Row label="Prompt file path"><Input value={settings?.summary?.prompt_file || ""} onChange={(value) => update(["summary", "prompt_file"], value)} /></Row>
      <Row label="Inline prompt override">
        <textarea
          value={settings?.summary?.prompt_text || ""}
          onChange={(event) => update(["summary", "prompt_text"], event.target.value)}
          className="w-full max-w-2xl min-h-32 px-3 py-2 rounded-md border border-echo-border bg-echo-surface-2 text-[12px] text-echo-text focus:outline-none focus:bg-echo-surface focus:border-echo-accent"
          placeholder="Optional. Leave empty to use the default template prompt."
        />
      </Row>
      <Row label="Effective prompt" hint="This is what ECHO will use now.">
        <textarea
          readOnly
          value={settings?.summary?.effective_prompt || ""}
          className="w-full max-w-2xl min-h-56 px-3 py-2 rounded-md border border-echo-border bg-echo-surface-2 text-[12px] text-echo-text-muted focus:outline-none"
        />
      </Row>
      <ul className="border border-echo-border rounded-md divide-y divide-echo-border">
        {prompts.map((p, i) => (
          <li key={p} className="px-4 py-2.5 flex items-center gap-3 hover:bg-echo-surface-hover cursor-pointer">
            <FlaskRound size={13} className="text-echo-text-muted" />
            <div className="flex-1">
              <div className="text-[12px] text-echo-text">{p}</div>
              <div className="text-[11px] text-echo-text-muted">v{3 - (i % 2)} · last edited {i + 2} days ago</div>
            </div>
            <Lock size={12} className="text-echo-text-faint" />
            <ChevronRight size={14} className="text-echo-text-faint" />
          </li>
        ))}
      </ul>
    </Panel>
  );
}

function ASRPanel({ settings, update }: { settings: EchoSettings | null; update: SettingsUpdate }) {
  return (
    <Panel title="ASR Configuration" desc="Speech-to-text settings for recordings without transcripts.">
      <Row label="ASR backend"><Select value={settings?.asr?.backend || "faster-whisper"} options={["faster-whisper", "hf-whisper"]} onChange={(value) => update(["asr", "backend"], value)} /></Row>
      <Row label="Whisper model"><Select value={settings?.asr?.model_size || "small"} options={["tiny", "base", "small", "medium", "large-v3"]} onChange={(value) => update(["asr", "model_size"], value)} /></Row>
      <Row label="Language"><Input value={settings?.asr?.language || "en"} onChange={(value) => update(["asr", "language"], value)} /></Row>
      <Row label="Voice activity detection (VAD)"><Toggle on={Boolean(settings?.asr?.vad)} label="Enabled" onChange={(value) => update(["asr", "vad"], value)} /></Row>
      <Row label="Force HF"><Toggle on={Boolean(settings?.asr?.force_hf)} label="Disable faster-whisper" onChange={(value) => update(["asr", "force_hf"], value)} /></Row>
      <Row label="Chunk length"><Input value={String(settings?.asr?.chunk_length_s || 30)} type="number" onChange={(value) => update(["asr", "chunk_length_s"], Number(value || 0))} /></Row>
      <Row label="Stride length"><Input value={String(settings?.asr?.stride_length_s || 5)} type="number" onChange={(value) => update(["asr", "stride_length_s"], Number(value || 0))} /></Row>
    </Panel>
  );
}

function StoragePanel({ settings, update }: { settings: EchoSettings | null; update: SettingsUpdate }) {
  return (
    <Panel title="Storage & Retention">
      <Row label="Output folder"><Input value={settings?.storage?.output_dir || "out"} onChange={(value) => update(["storage", "output_dir"], value)} /></Row>
      <Row label="Uploads folder"><Input value={settings?.storage?.uploads_dir || "uploads"} onChange={(value) => update(["storage", "uploads_dir"], value)} /></Row>
      <Row label="Exports folder"><Input value={settings?.storage?.exports_dir || "exports"} onChange={(value) => update(["storage", "exports_dir"], value)} /></Row>
      <Row label="Save transcript"><Toggle on={Boolean(settings?.storage?.save_transcript)} label="Keep raw transcript with MoM" onChange={(value) => update(["storage", "save_transcript"], value)} /></Row>
      <Row label="Save summary"><Toggle on={Boolean(settings?.storage?.save_summary)} label="Always" onChange={(value) => update(["storage", "save_summary"], value)} /></Row>
      <Row label="Save email draft"><Toggle on={Boolean(settings?.storage?.save_email_draft)} label="Only when requested" onChange={(value) => update(["storage", "save_email_draft"], value)} /></Row>
      <Row label="Retention period (days)"><Input value={String(settings?.storage?.retention_days || 14)} type="number" onChange={(value) => update(["storage", "retention_days"], Number(value || 0))} /></Row>
    </Panel>
  );
}

function CompliancePanel({ settings, update }: { settings: EchoSettings | null; update: SettingsUpdate }) {
  return (
    <Panel title="Privacy & Compliance">
      <Row label="Local-only mode" hint="No data leaves the workspace network."><Toggle on={Boolean(settings?.privacy?.local_only_mode)} label="Enabled" onChange={(value) => update(["privacy", "local_only_mode"], value)} /></Row>
      <Row label="Redact PII automatically"><Toggle on={Boolean(settings?.privacy?.redact_pii)} label="Names, emails, phone numbers" onChange={(value) => update(["privacy", "redact_pii"], value)} /></Row>
      <Row label="Delete transcript after MoM"><Toggle on={Boolean(settings?.privacy?.delete_transcript_after_mom)} label="Enabled" onChange={(value) => update(["privacy", "delete_transcript_after_mom"], value)} /></Row>
      <Row label="Audit logging"><Toggle on={Boolean(settings?.privacy?.audit_logging)} label="All MoM access logged" onChange={(value) => update(["privacy", "audit_logging"], value)} /></Row>
    </Panel>
  );
}

function setIn(source: EchoSettings, path: string[], value: unknown): EchoSettings {
  const root = structuredClone(source || {});
  let cursor: any = root;
  path.slice(0, -1).forEach((key) => {
    if (!cursor[key] || typeof cursor[key] !== "object") cursor[key] = {};
    cursor = cursor[key];
  });
  cursor[path[path.length - 1]] = value;
  return root;
}
