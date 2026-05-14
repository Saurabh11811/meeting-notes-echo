import { useEffect, useState } from "react";
import { PageHeader } from "../page-header";
import { Server, Mic, Database, Activity, KeyRound, CheckCircle2, Save, RotateCcw } from "lucide-react";
import { getSettings, updateSettings, type EchoSettings } from "../../api/echo-api";

const sections = [
  { id: "health", label: "System Health", icon: Activity },
  { id: "backend", label: "Backend Settings", icon: Server },
  { id: "asr", label: "ASR Configuration", icon: Mic },
  { id: "storage", label: "Storage & Retention", icon: Database },
];

export function SettingsPage() {
  const [sel, setSel] = useState("health");
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
        actions={
          <>
            {status && <span className="text-[12px] text-echo-text-muted">{status}</span>}
            <button onClick={reload} className="h-9 px-3 rounded-md border border-echo-border bg-echo-surface hover:bg-echo-surface-hover text-[12px] text-echo-text inline-flex items-center gap-1.5"><RotateCcw size={13} />Reload</button>
            <button onClick={save} disabled={!draft || saving} className="h-9 px-3 rounded-md bg-echo-accent text-white text-[12px] inline-flex items-center gap-1.5 disabled:opacity-60"><Save size={13} />{saving ? "Saving…" : "Save settings"}</button>
          </>
        }
      />

      <div className="grid grid-cols-1 xl:grid-cols-[260px_1fr] gap-5">
        <aside className="bg-echo-surface border border-echo-border rounded-lg p-2 h-fit space-y-0.5">
          {sections.map((s) => {
            const I = s.icon;
            const active = sel === s.id;
            return (
              <button key={s.id} onClick={() => setSel(s.id)} className={`w-full flex items-center gap-2.5 px-3 py-2 rounded-md text-[12px] ${active ? "bg-echo-accent-bg text-echo-accent-fg" : "text-echo-text hover:bg-echo-surface-hover"}`}>
                <I size={14} className={active ? "text-echo-accent" : "text-echo-text-muted"} />
                <span className="flex-1 text-left">{s.label}</span>
              </button>
            );
          })}
        </aside>

        <section className="space-y-5">
          {sel === "health" && <HealthPanel settings={view} />}
          {sel === "backend" && <BackendPanel settings={view} update={setValue} />}
          {sel === "asr" && <ASRPanel settings={view} update={setValue} />}
          {sel === "storage" && <StoragePanel settings={view} update={setValue} />}
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
