import { useEffect, useMemo, useState, type ChangeEvent, type ReactNode } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import {
  Briefcase,
  Check,
  ChevronDown,
  ClipboardCheck,
  Copy,
  FileText,
  FlaskConical,
  History,
  LayoutTemplate,
  Loader2,
  Lock,
  Megaphone,
  MonitorPlay,
  Pencil,
  Phone,
  Plus,
  Save,
  ShieldAlert,
  Sparkles,
  Star,
  Trash2,
  X,
  type LucideIcon,
} from "lucide-react";
import { PageHeader } from "../page-header";
import {
  createTemplate,
  deleteTemplate,
  duplicateTemplate,
  getTemplatePrompt,
  getTemplatePresets,
  getTemplates,
  getTemplateVersions,
  restoreTemplateVersion,
  testTemplate,
  updateTemplate,
  type EchoTemplate,
  type TemplatePromptResponse,
  type TemplatePreset,
  type TemplateVersion,
} from "../../api/echo-api";

const iconMap: Record<string, LucideIcon> = {
  "Executive MoM": Briefcase,
  "Project Review": ClipboardCheck,
  "Client Call": Phone,
  Townhall: Megaphone,
  "Demo/UAT": MonitorPlay,
  "Incident Review": ShieldAlert,
  Incident: ShieldAlert,
};

const meetingTypes = ["General", "Executive", "Project Review", "Client Call", "Townhall", "Demo/UAT", "Incident"];

function getIcon(template: EchoTemplate | null) {
  if (!template) return LayoutTemplate;
  return iconMap[template.name] || iconMap[template.meeting_type] || LayoutTemplate;
}

export function TemplatesPage() {
  const [templates, setTemplates] = useState<EchoTemplate[]>([]);
  const [presets, setPresets] = useState<TemplatePreset[]>([]);
  const [activeId, setActiveId] = useState("");
  const [loading, setLoading] = useState(true);
  const [showEditor, setShowEditor] = useState(false);
  const [showSandbox, setShowSandbox] = useState(false);
  const [editingTemplate, setEditingTemplate] = useState<EchoTemplate | null>(null);
  const [prompt, setPrompt] = useState<TemplatePromptResponse | null>(null);
  const [versions, setVersions] = useState<TemplateVersion[]>([]);
  const [detailTab, setDetailTab] = useState<"prompt" | "history">("prompt");
  const [message, setMessage] = useState("");

  const current = useMemo(
    () => templates.find((template) => template.id === activeId) || templates[0] || null,
    [templates, activeId],
  );
  const CurrentIcon = getIcon(current);

  const refreshTemplates = async (preferredId?: string) => {
    setLoading(true);
    try {
      const [templateData, presetData] = await Promise.all([getTemplates(), getTemplatePresets()]);
      setTemplates(templateData.templates);
      setPresets(presetData.presets);
      const nextId = preferredId || activeId || templateData.templates[0]?.id || "";
      setActiveId(templateData.templates.some((template) => template.id === nextId) ? nextId : templateData.templates[0]?.id || "");
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    refreshTemplates().catch((err) => setMessage(err instanceof Error ? err.message : "Could not load templates."));
  }, []);

  useEffect(() => {
    if (!current) {
      setPrompt(null);
      setVersions([]);
      return;
    }
    getTemplatePrompt(current.id)
      .then(setPrompt)
      .catch(() => setPrompt(null));
    getTemplateVersions(current.id)
      .then((data) => setVersions(data.versions))
      .catch(() => setVersions([]));
  }, [current?.id]);

  const handleSetDefault = async (templateId: string) => {
    try {
      const updated = await updateTemplate(templateId, { is_default: true });
      setMessage(`${updated.name} is now the default template.`);
      await refreshTemplates(updated.id);
    } catch (err) {
      setMessage(err instanceof Error ? err.message : "Failed to set default.");
    }
  };

  const handleDelete = async (templateId: string) => {
    if (!confirm("Delete this template? This cannot be undone.")) return;
    try {
      await deleteTemplate(templateId);
      setMessage("Template deleted.");
      await refreshTemplates();
    } catch (err) {
      setMessage(err instanceof Error ? err.message : "Failed to delete template.");
    }
  };

  const handleDuplicate = async (templateId: string) => {
    try {
      const copy = await duplicateTemplate(templateId);
      setMessage(`Created ${copy.name}.`);
      await refreshTemplates(copy.id);
    } catch (err) {
      setMessage(err instanceof Error ? err.message : "Failed to duplicate template.");
    }
  };

  const handleRestoreVersion = async (version: TemplateVersion) => {
    if (!current) return;
    if (!confirm(`Restore ${current.name} to v${version.version_number}?`)) return;
    try {
      const restored = await restoreTemplateVersion(current.id, version.id);
      setMessage(`Restored ${restored.name} from v${version.version_number}.`);
      await refreshTemplates(restored.id);
    } catch (err) {
      setMessage(err instanceof Error ? err.message : "Failed to restore template version.");
    }
  };

  const openEditor = (template: EchoTemplate | null) => {
    setEditingTemplate(template);
    setShowEditor(true);
  };

  if (loading && !templates.length) {
    return <div className="p-10 text-center text-echo-text-muted">Loading templates...</div>;
  }

  return (
    <div className="space-y-5">
      <PageHeader
        title="Template Studio"
        subtitle="Design the exact prompt and structure used when ECHO generates MoM output."
        actions={
          <button
            onClick={() => openEditor(null)}
            className="h-9 px-3 rounded-md bg-echo-text text-echo-surface text-[14px] inline-flex items-center gap-1.5 hover:opacity-90 transition-opacity"
          >
            <Plus size={14} />New template
          </button>
        }
      />

      {message && (
        <div className="px-4 py-3 rounded-md border border-echo-border bg-echo-surface text-[14px] text-echo-text-muted">
          {message}
        </div>
      )}

      <div className="grid grid-cols-1 xl:grid-cols-[minmax(0,1fr)_460px] gap-5">
        <section className="grid sm:grid-cols-2 gap-3 h-fit">
          {templates.map((template) => {
            const TemplateIcon = getIcon(template);
            const active = current?.id === template.id;
            return (
              <button
                key={template.id}
                onClick={() => setActiveId(template.id)}
                className={`text-left bg-echo-surface border rounded-lg p-4 transition-colors ${
                  active ? "border-echo-accent ring-2 ring-echo-accent/20" : "border-echo-border hover:border-echo-border-strong"
                }`}
              >
                <div className="flex items-start gap-3">
                  <div className="h-10 w-10 rounded-md bg-echo-accent-bg text-echo-accent grid place-items-center shrink-0">
                    <TemplateIcon size={17} />
                  </div>
                  <div className="min-w-0 flex-1">
                    <div className="flex items-center gap-2">
                      <div className="text-[15px] text-echo-text truncate" style={{ fontWeight: 700 }}>
                        {template.name}
                      </div>
                      {template.is_default && <Badge tone="accent">Default</Badge>}
                      {template.is_locked && <Lock size={12} className="text-echo-text-faint shrink-0" />}
                    </div>
                    <p className="text-[14px] text-echo-text-muted mt-1 line-clamp-2">{template.description}</p>
                    <div className="mt-3 flex items-center justify-between gap-2">
                      <span className="text-[12px] text-echo-text-faint uppercase tracking-wide">{template.meeting_type}</span>
                      <span className="text-[12px] text-echo-text-muted">{template.sections.length} sections</span>
                    </div>
                  </div>
                </div>
              </button>
            );
          })}
        </section>

        {current && (
          <aside className="bg-echo-surface border border-echo-border rounded-lg overflow-hidden h-fit sticky top-5 shadow-sm">
            <div className="px-5 py-4 border-b border-echo-border flex items-start gap-3">
              <div className="h-11 w-11 rounded-md bg-echo-accent-bg text-echo-accent grid place-items-center">
                <CurrentIcon size={19} />
              </div>
              <div className="flex-1 min-w-0">
                <div className="flex items-center gap-2">
                  <h2 className="text-[16px] text-echo-text truncate" style={{ fontWeight: 750 }}>
                    {current.name}
                  </h2>
                  {current.is_locked && <Badge>Built-in</Badge>}
                </div>
                <p className="text-[14px] text-echo-text-muted mt-1">{current.description}</p>
              </div>
            </div>

            <div className="p-5 space-y-4">
              <div>
                <div className="text-[12px] text-echo-text-muted uppercase tracking-wide font-semibold mb-2">Sections</div>
                <div className="flex flex-wrap gap-1.5">
                  {current.sections.map((section) => (
                    <span key={section} className="px-2 py-1 rounded bg-echo-surface-2 border border-echo-border text-echo-text-muted text-[13px]">
                      {section}
                    </span>
                  ))}
                </div>
              </div>

              <div className="flex rounded-md border border-echo-border bg-echo-surface-2 p-1">
                <TabButton active={detailTab === "prompt"} onClick={() => setDetailTab("prompt")} icon={FileText} label="Prompt" />
                <TabButton active={detailTab === "history"} onClick={() => setDetailTab("history")} icon={History} label="History" />
              </div>

              {detailTab === "prompt" ? (
                <div className="space-y-3">
                  <PromptBlock title="Prompt used for generation" value={current.system_prompt || prompt?.prompt || fallbackPrompt(current)} />
                </div>
              ) : (
                <div className="space-y-2 max-h-[360px] overflow-y-auto pr-1">
                  {versions.length === 0 && <p className="text-[14px] text-echo-text-muted">No saved revisions yet.</p>}
                  {versions.map((version) => (
                    <div key={version.id} className="rounded-md border border-echo-border bg-echo-surface-2 p-3">
                      <div className="flex items-center justify-between gap-2">
                        <div className="text-[14px] text-echo-text" style={{ fontWeight: 700 }}>v{version.version_number}</div>
                        <div className="text-[12px] text-echo-text-faint">{formatDateTime(version.created_at)}</div>
                      </div>
                      <div className="text-[13px] text-echo-text-muted mt-1">{version.change_note || "Saved revision"}</div>
                      <div className="mt-2 flex items-center justify-between gap-2">
                        <div className="text-[12px] text-echo-text-faint">{version.sections.length} sections</div>
                        <button onClick={() => handleRestoreVersion(version)} className="h-7 px-2 rounded border border-echo-border bg-echo-surface text-[12px] text-echo-text hover:bg-echo-surface-hover">
                          Restore
                        </button>
                      </div>
                    </div>
                  ))}
                </div>
              )}

              <div className="grid grid-cols-2 gap-2">
                {!current.is_default && (
                  <button onClick={() => handleSetDefault(current.id)} className="h-10 rounded-md bg-echo-accent text-white text-[14px] inline-flex items-center justify-center gap-1.5">
                    <Star size={14} />Use default
                  </button>
                )}
                <button onClick={() => openEditor(current)} className="h-10 rounded-md border border-echo-border bg-echo-surface text-echo-text text-[14px] inline-flex items-center justify-center gap-1.5 hover:bg-echo-surface-hover">
                  <Pencil size={14} />Edit
                </button>
                <button onClick={() => handleDuplicate(current.id)} className="h-10 rounded-md border border-echo-border bg-echo-surface text-echo-text text-[14px] inline-flex items-center justify-center gap-1.5 hover:bg-echo-surface-hover">
                  <Copy size={14} />Duplicate
                </button>
                {!current.is_locked && (
                  <button onClick={() => handleDelete(current.id)} className="h-10 rounded-md border border-echo-danger/20 bg-echo-surface text-echo-danger text-[14px] inline-flex items-center justify-center gap-1.5 hover:bg-echo-danger/5">
                    <Trash2 size={14} />Delete
                  </button>
                )}
                <button onClick={() => setShowSandbox(true)} className="h-10 rounded-md border border-echo-accent/30 bg-echo-accent/5 text-echo-accent text-[14px] inline-flex items-center justify-center gap-1.5 col-span-2 hover:bg-echo-accent/10">
                  <FlaskConical size={14} />Test with transcript
                </button>
              </div>
            </div>
          </aside>
        )}
      </div>

      {showEditor && (
        <TemplateEditor
          template={editingTemplate}
          presets={presets}
          onClose={() => setShowEditor(false)}
          onSaved={async (template) => {
            setShowEditor(false);
            setMessage(`${template.name} saved.`);
            await refreshTemplates(template.id);
          }}
        />
      )}

      {showSandbox && current && <TemplateSandbox template={current} onClose={() => setShowSandbox(false)} />}
    </div>
  );
}

function TemplateEditor({
  template,
  presets,
  onClose,
  onSaved,
}: {
  template: EchoTemplate | null;
  presets: TemplatePreset[];
  onClose: () => void;
  onSaved: (template: EchoTemplate) => void;
}) {
  const firstPreset = presets.find((preset) => preset.meeting_type === "Executive") || presets[0];
  const [name, setName] = useState(template?.name || (firstPreset?.meeting_type ? `${firstPreset.meeting_type} MoM` : ""));
  const [meetingType, setMeetingType] = useState(template?.meeting_type || firstPreset?.meeting_type || "General");
  const [description, setDescription] = useState(template?.description || firstPreset?.description || "");
  const [sections, setSections] = useState<string[]>(template?.sections || firstPreset?.sections || []);
  const [systemPrompt, setSystemPrompt] = useState(template?.system_prompt || (template ? fallbackPrompt(template) : firstPreset?.system_prompt || ""));
  const [newSection, setNewSection] = useState("");
  const [saving, setSaving] = useState(false);

  const applyPreset = (type: string) => {
    const preset = presets.find((item) => item.meeting_type === type);
    setMeetingType(type);
    if (!preset) return;
    if (!template) setName((current) => current || `${preset.meeting_type} MoM`);
    setDescription(preset.description);
    setSections(preset.sections);
    setSystemPrompt(preset.system_prompt);
  };

  const handleSave = async () => {
    if (!name.trim()) {
      alert("Template name is required.");
      return;
    }
    setSaving(true);
    try {
      const payload = {
        name: name.trim(),
        meeting_type: meetingType,
        description,
        sections,
        system_prompt: systemPrompt,
      };
      const saved = template ? await updateTemplate(template.id, payload) : await createTemplate(payload);
      onSaved(saved);
    } catch (err) {
      alert(err instanceof Error ? err.message : "Failed to save template.");
    } finally {
      setSaving(false);
    }
  };

  const addSection = () => {
    const value = newSection.trim();
    if (!value || sections.includes(value)) {
      setNewSection("");
      return;
    }
    setSections([...sections, value]);
    setNewSection("");
  };

  const removeSection = (section: string) => {
    setSections(sections.filter((item) => item !== section));
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/45 backdrop-blur-sm p-4">
      <div className="bg-echo-surface border border-echo-border rounded-xl shadow-2xl w-full max-w-5xl overflow-hidden flex flex-col max-h-[92vh]">
        <div className="px-6 py-4 border-b border-echo-border flex items-center justify-between bg-echo-surface-2">
          <div>
            <h2 className="text-[18px] text-echo-text" style={{ fontWeight: 750 }}>{template ? "Edit template" : "New template"}</h2>
            <p className="text-[13px] text-echo-text-muted">
              {template?.is_locked ? "Built-in templates can be customized but not renamed or deleted." : "Define the structure and exact AI instructions."}
            </p>
          </div>
          <button onClick={onClose} className="text-echo-text-faint hover:text-echo-text p-1">
            <X size={20} />
          </button>
        </div>

        <div className="p-6 overflow-y-auto grid lg:grid-cols-[360px_minmax(0,1fr)] gap-6">
          <div className="space-y-4">
            <Field label="Name">
              <input
                value={name}
                onChange={(event) => setName(event.target.value)}
                disabled={Boolean(template?.is_locked)}
                className="w-full h-11 px-3 rounded-md border border-echo-border bg-echo-surface-2 text-[15px] focus:outline-none focus:border-echo-accent disabled:opacity-65"
              />
            </Field>
            <Field label="Meeting type">
              <SelectShell>
                <select value={meetingType} onChange={(event) => applyPreset(event.target.value)} className="w-full bg-transparent text-echo-text focus:outline-none">
                  {meetingTypes.map((type) => <option key={type}>{type}</option>)}
                </select>
              </SelectShell>
              <p className="text-[12px] text-echo-text-faint mt-1">Changing type applies the recommended sections and prompt for that meeting type.</p>
            </Field>
            <Field label="Description">
              <textarea
                value={description}
                onChange={(event) => setDescription(event.target.value)}
                className="w-full h-24 p-3 rounded-md border border-echo-border bg-echo-surface-2 text-[15px] focus:outline-none focus:border-echo-accent"
              />
            </Field>
            <Field label="Sections">
              <div className="flex gap-2">
                <input
                  value={newSection}
                  onChange={(event) => setNewSection(event.target.value)}
                  onKeyDown={(event) => {
                    if (event.key === "Enter") {
                      event.preventDefault();
                      addSection();
                    }
                  }}
                  placeholder="Add a section"
                  className="flex-1 h-10 px-3 rounded-md border border-echo-border bg-echo-surface-2 text-[14px] focus:outline-none focus:border-echo-accent"
                />
                <button onClick={addSection} className="px-4 rounded-md bg-echo-accent text-white text-[14px]">Add</button>
              </div>
              <div className="mt-3 flex flex-wrap gap-2 rounded-md border border-echo-border bg-echo-surface-2/60 p-3 min-h-20">
                {sections.map((section) => (
                  <span key={section} className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-md bg-echo-accent/10 text-echo-accent border border-echo-accent/20 text-[13px]">
                    {section}
                    <button onClick={() => removeSection(section)}><X size={12} /></button>
                  </span>
                ))}
              </div>
            </Field>
          </div>

          <div className="space-y-3">
            <div className="flex items-center justify-between gap-3">
              <div>
                <div className="text-[12px] text-echo-text-muted uppercase tracking-wide font-semibold">Prompt</div>
                <p className="text-[13px] text-echo-text-muted">This is the full prompt sent to the summary backend for this template.</p>
              </div>
            </div>
            <textarea
              value={systemPrompt}
              onChange={(event) => setSystemPrompt(event.target.value)}
              className="w-full min-h-[360px] p-4 rounded-md border border-echo-border bg-echo-surface-2 text-[14px] font-mono leading-relaxed focus:outline-none focus:border-echo-accent"
            />
          </div>
        </div>

        <div className="px-6 py-4 border-t border-echo-border flex items-center justify-end gap-3 bg-echo-surface-2">
          <button onClick={onClose} className="h-10 px-5 rounded-md border border-echo-border bg-echo-surface text-echo-text text-[14px] hover:bg-echo-surface-hover">Cancel</button>
          <button onClick={handleSave} disabled={saving} className="h-10 px-5 rounded-md bg-echo-accent text-white text-[14px] inline-flex items-center gap-2 disabled:opacity-55">
            {saving ? <Loader2 size={16} className="animate-spin" /> : <Save size={16} />}
            {saving ? "Saving..." : "Save template"}
          </button>
        </div>
      </div>
    </div>
  );
}

function TemplateSandbox({ template, onClose }: { template: EchoTemplate; onClose: () => void }) {
  const [transcript, setTranscript] = useState("");
  const [backendKind, setBackendKind] = useState("");
  const [output, setOutput] = useState("");
  const [testing, setTesting] = useState(false);

  const handleTest = async () => {
    if (!transcript.trim()) return;
    setTesting(true);
    setOutput("");
    try {
      const result = await testTemplate({
        transcript_text: transcript,
        template_name: template.name,
        backend_kind: backendKind || undefined,
      });
      setOutput(result.output);
    } catch (err) {
      alert(err instanceof Error ? err.message : "Test failed.");
    } finally {
      setTesting(false);
    }
  };

  const handleFileUpload = (event: ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;
    file.text().then(setTranscript);
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/45 backdrop-blur-sm p-4">
      <div className="bg-echo-surface border border-echo-border rounded-xl shadow-2xl w-full max-w-6xl overflow-hidden flex flex-col h-[88vh]">
        <div className="px-6 py-4 border-b border-echo-border flex items-center justify-between bg-echo-surface-2">
          <div className="flex items-center gap-3">
            <div className="h-9 w-9 rounded bg-echo-accent/10 text-echo-accent grid place-items-center"><FlaskConical size={17} /></div>
            <div>
              <h2 className="text-[18px] text-echo-text" style={{ fontWeight: 750 }}>Template sandbox</h2>
              <p className="text-[13px] text-echo-text-muted">Test {template.name} with transcript text before using it in production.</p>
            </div>
          </div>
          <button onClick={onClose} className="text-echo-text-faint hover:text-echo-text p-1"><X size={20} /></button>
        </div>

        <div className="flex-1 grid lg:grid-cols-2 overflow-hidden">
          <div className="border-r border-echo-border flex flex-col p-4 space-y-3 min-h-0">
            <div className="flex items-center justify-between gap-3">
              <label className="text-[12px] font-bold text-echo-text-muted uppercase tracking-wide">Input transcript</label>
              <label className="text-[13px] text-echo-accent hover:underline cursor-pointer flex items-center gap-1">
                <FileText size={13} />Import .txt
                <input type="file" accept=".txt,.md,.vtt,.srt,text/plain,text/markdown" className="hidden" onChange={handleFileUpload} />
              </label>
            </div>
            <textarea
              value={transcript}
              onChange={(event) => setTranscript(event.target.value)}
              placeholder="Paste transcript text here..."
              className="flex-1 w-full p-4 rounded-md border border-echo-border bg-echo-surface-2 text-[14px] font-mono leading-relaxed focus:outline-none focus:border-echo-accent resize-none min-h-0"
            />
            <div className="flex items-center gap-2">
              <SelectShell className="w-44">
                <select value={backendKind} onChange={(event) => setBackendKind(event.target.value)} className="w-full bg-transparent text-echo-text focus:outline-none">
                  <option value="">Default backend</option>
                  <option value="local">Local</option>
                  <option value="dify">Dify</option>
                  <option value="azure">Azure</option>
                </select>
              </SelectShell>
              <button onClick={handleTest} disabled={testing || !transcript.trim()} className="flex-1 h-11 rounded-md bg-echo-accent text-white text-[14px] flex items-center justify-center gap-2 disabled:opacity-55">
                {testing ? <Loader2 size={17} className="animate-spin" /> : <Sparkles size={17} />}
                {testing ? "Generating..." : "Generate preview"}
              </button>
            </div>
          </div>

          <div className="flex flex-col bg-echo-surface-2/30 p-4 space-y-3 min-h-0">
            <div className="flex items-center justify-between">
              <label className="text-[12px] font-bold text-echo-text-muted uppercase tracking-wide">Rendered output</label>
              {output && <div className="text-[12px] text-echo-success flex items-center gap-1"><Check size={12} />Generated</div>}
            </div>
            <div className="flex-1 w-full p-5 rounded-md border border-echo-border bg-echo-surface overflow-y-auto min-h-0">
              {output ? <MarkdownPreview text={output} /> : (
                <div className="h-full flex flex-col items-center justify-center text-center p-8">
                  <div className="h-12 w-12 rounded-full bg-echo-surface-2 border border-echo-border grid place-items-center mb-3">
                    <Sparkles size={24} className="text-echo-text-faint" />
                  </div>
                  <h3 className="text-[15px] text-echo-text font-medium">No preview yet</h3>
                  <p className="text-[13px] text-echo-text-muted mt-1">Paste a transcript and generate a Markdown-rendered preview.</p>
                </div>
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

function Field({ label, children }: { label: string; children: ReactNode }) {
  return (
    <div>
      <label className="block text-[12px] font-semibold text-echo-text-muted uppercase tracking-wide mb-1.5">{label}</label>
      {children}
    </div>
  );
}

function SelectShell({ children, className = "" }: { children: ReactNode; className?: string }) {
  return (
    <div className={`h-11 px-3 rounded-md border border-echo-border bg-echo-surface-2 text-[15px] flex items-center gap-2 focus-within:border-echo-accent ${className}`}>
      {children}
      <ChevronDown size={13} className="text-echo-text-faint" />
    </div>
  );
}

function TabButton({ active, onClick, icon: Icon, label }: { active: boolean; onClick: () => void; icon: LucideIcon; label: string }) {
  return (
    <button onClick={onClick} className={`flex-1 h-8 rounded text-[13px] inline-flex items-center justify-center gap-1.5 ${active ? "bg-echo-surface text-echo-text shadow-sm" : "text-echo-text-muted"}`}>
      <Icon size={13} />{label}
    </button>
  );
}

function Badge({ children, tone = "muted" }: { children: ReactNode; tone?: "muted" | "accent" }) {
  return <span className={`text-[11px] px-1.5 py-0.5 rounded ${tone === "accent" ? "bg-echo-accent text-white" : "bg-echo-surface-2 text-echo-text-muted border border-echo-border"}`}>{children}</span>;
}

function PromptBlock({ title, value, muted = false }: { title: string; value: string; muted?: boolean }) {
  return (
    <div>
      <div className="text-[12px] text-echo-text-muted uppercase tracking-wide font-semibold mb-1.5">{title}</div>
      <pre className={`max-h-60 overflow-auto rounded-md border border-echo-border bg-echo-surface-2 p-3 text-[12px] leading-relaxed whitespace-pre-wrap ${muted ? "text-echo-text-muted" : "text-echo-text"}`}>
        {value}
      </pre>
    </div>
  );
}

function MarkdownPreview({ text }: { text: string }) {
  return (
    <div className="text-[15px] leading-7 text-echo-text">
      <ReactMarkdown
        remarkPlugins={[remarkGfm]}
        components={{
          h1: ({ children }) => <h1 className="text-[23px] mt-1 mb-3 text-echo-text" style={{ fontWeight: 800 }}>{children}</h1>,
          h2: ({ children }) => <h2 className="text-[20px] mt-5 mb-2 text-echo-text" style={{ fontWeight: 750 }}>{children}</h2>,
          h3: ({ children }) => <h3 className="text-[17px] mt-4 mb-2 text-echo-text" style={{ fontWeight: 700 }}>{children}</h3>,
          p: ({ children }) => <p className="my-2 text-echo-text-muted">{children}</p>,
          ul: ({ children }) => <ul className="my-3 list-disc pl-6 space-y-1.5">{children}</ul>,
          ol: ({ children }) => <ol className="my-3 list-decimal pl-6 space-y-1.5">{children}</ol>,
          li: ({ children }) => <li className="text-echo-text-muted">{children}</li>,
          table: ({ children }) => <div className="my-4 overflow-x-auto rounded-md border border-echo-border"><table className="w-full border-collapse text-[14px]">{children}</table></div>,
          thead: ({ children }) => <thead className="bg-echo-surface-2 text-echo-text">{children}</thead>,
          th: ({ children }) => <th className="border-b border-r border-echo-border px-3 py-2 text-left align-top last:border-r-0">{children}</th>,
          td: ({ children }) => <td className="border-b border-r border-echo-border px-3 py-2 align-top text-echo-text-muted last:border-r-0">{children}</td>,
          tr: ({ children }) => <tr className="last:[&_td]:border-b-0">{children}</tr>,
          code: ({ children }) => <code className="rounded bg-echo-surface-2 px-1.5 py-0.5 text-[13px] text-echo-text">{children}</code>,
        }}
      >
        {text}
      </ReactMarkdown>
    </div>
  );
}

function fallbackPrompt(template: EchoTemplate) {
  return buildPrompt(template.name, template.description, template.sections);
}

function buildPrompt(name: string, description: string, sections: string[]) {
  const sectionLines = sections.length ? sections.map((section) => `- ${section}`).join("\n") : "- Summary\n- Decisions\n- Action Items\n- Risks\n- Next Steps";
  return [
    "You are a professional scribe. Produce detailed, structured, and comprehensive Minutes of Meeting (MoM) from the provided transcript.",
    "The MoM must be clear enough for someone who did not attend to understand the discussion, outcomes, decisions, and next steps.",
    "",
    `Template: ${name || "Meeting MoM"}`,
    `Purpose: ${description || "Create clear, evidence-based meeting notes."}`,
    "",
    "Write in professional Markdown with clear headings and tables where they improve scanability.",
    "Use only facts from the transcript. If evidence is missing, write 'Not captured in transcript'.",
    "Capture speakers, owners, dates, deadlines, dependencies, risks, decisions, and exact values whenever they appear.",
    "Do not invent commitments, attendees, dates, risks, or decisions.",
    "",
    "Required sections:",
    sectionLines,
  ].join("\n");
}

function formatDateTime(value: string) {
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return value;
  return date.toLocaleString(undefined, { month: "short", day: "2-digit", hour: "2-digit", minute: "2-digit" });
}
