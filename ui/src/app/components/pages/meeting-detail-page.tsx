import { useEffect, useMemo, useRef, useState, type ReactNode } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import {
  ArrowLeft,
  Calendar,
  ChevronDown,
  ChevronRight,
  Copy,
  Download,
  FileText,
  FileType2,
  Link2,
  Mail,
  RefreshCw,
  Search,
  Sparkles,
} from "lucide-react";
import {
  getMeetingDetail,
  getSettings,
  getTemplates,
  meetingExportUrl,
  regenerateMeeting,
  type EchoTemplate,
  type EchoSettings,
  type MeetingDetail,
  type MomVersion,
} from "../../api/echo-api";

const tabs = ["MoM", "Transcript", "Versions", "Exports"];

export function MeetingDetailPage({ meetingId, onBack }: { meetingId: string | null; onBack: () => void }) {
  const [tab, setTab] = useState("MoM");
  const [detail, setDetail] = useState<MeetingDetail | null>(null);
  const [settings, setSettings] = useState<EchoSettings | null>(null);
  const [templates, setTemplates] = useState<EchoTemplate[]>([]);
  const [selectedVersionId, setSelectedVersionId] = useState("");
  const [expandedVersionIds, setExpandedVersionIds] = useState<string[]>([]);
  const [templateName, setTemplateName] = useState("");
  const [backendKind, setBackendKind] = useState("");
  const [loading, setLoading] = useState(false);
  const [regenerating, setRegenerating] = useState(false);
  const [message, setMessage] = useState("");
  const [error, setError] = useState("");

  const loadDetail = async () => {
    if (!meetingId) {
      setDetail(null);
      setError("");
      return;
    }
    setLoading(true);
    try {
      const data = await getMeetingDetail(meetingId);
      setDetail(data);
      setError("");
      setSelectedVersionId((current) => current || data.latest_mom?.id || data.mom_versions[0]?.id || "");
      setBackendKind((current) => current || data.latest_mom?.backend_kind || "");
    } catch (err) {
      setDetail(null);
      setError(err instanceof Error ? err.message : "Could not load meeting.");
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadDetail();
  }, [meetingId]);

  useEffect(() => {
    getSettings().then((data) => {
      setSettings(data);
      setBackendKind((current) => current || data?.summary?.default_backend || "");
    }).catch(() => setSettings(null));
    getTemplates().then((data) => setTemplates(data.templates)).catch(() => setTemplates([]));
  }, []);

  useEffect(() => {
    if (!meeting || !templates.length || templateName) return;
    const matching = templates.filter((template) => template.meeting_type === meeting.meeting_type);
    const preferred = matching.find((template) => template.is_default) || matching[0] || templates.find((template) => template.is_default) || templates[0];
    setTemplateName(preferred?.name || "");
  }, [meeting, templates, templateName]);

  const meeting = detail?.meeting;
  const selectedMom = useMemo(() => {
    if (!detail) return null;
    return detail.mom_versions.find((version) => version.id === selectedVersionId) || detail.latest_mom;
  }, [detail, selectedVersionId]);
  const templateOptions = useMemo(() => templateNames(templates, meeting?.meeting_type, templateName), [templates, meeting?.meeting_type, templateName]);
  const backendOptions = useMemo(() => backendProfiles(settings, selectedMom?.backend_kind), [settings, selectedMom?.backend_kind]);

  const handleViewVersion = (versionId: string) => {
    setSelectedVersionId(versionId);
    setTab("MoM");
  };

  const toggleVersionExpanded = (versionId: string) => {
    setExpandedVersionIds((current) => (
      current.includes(versionId)
        ? current.filter((id) => id !== versionId)
        : [...current, versionId]
    ));
  };

  const handleCopy = async (text: string, label = "Copied") => {
    if (!text.trim()) {
      setMessage("Nothing to copy.");
      return;
    }
    await navigator.clipboard.writeText(text);
    setMessage(label);
  };

  const handleCopyRich = async (text: string, html: string, label = "Copied") => {
    if (!text.trim()) {
      setMessage("Nothing to copy.");
      return;
    }
    if (html.trim() && "ClipboardItem" in window) {
      await navigator.clipboard.write([
        new ClipboardItem({
          "text/html": new Blob([htmlDocumentFragment(html)], { type: "text/html" }),
          "text/plain": new Blob([text], { type: "text/plain" }),
        }),
      ]);
    } else {
      await navigator.clipboard.writeText(text);
    }
    setMessage(label);
  };

  const handleRegenerate = async () => {
    if (!meetingId) return;
    setRegenerating(true);
    setMessage("");
    try {
      await regenerateMeeting(meetingId, {
        template_name: templateName,
        backend_kind: backendKind,
        run_now: true,
      });
      setMessage("Regeneration queued. The next version will appear when processing finishes.");
      await loadDetail();
    } catch (err) {
      setMessage(err instanceof Error ? err.message : "Could not regenerate this MoM.");
    } finally {
      setRegenerating(false);
    }
  };

  if (!meetingId) {
    return (
      <EmptyShell onBack={onBack} title="No meeting selected">
        Open a completed meeting from Home or Meeting Notes to review the generated MoM.
      </EmptyShell>
    );
  }

  if (loading && !detail) {
    return (
      <EmptyShell onBack={onBack} title="Loading meeting">
        Reading meeting notes and transcript from the local database.
      </EmptyShell>
    );
  }

  if (error || !detail || !meeting) {
    return (
      <EmptyShell onBack={onBack} title="Meeting not found">
        {error || "This meeting is not available in the local database."}
      </EmptyShell>
    );
  }

  return (
    <div className="space-y-5">
      <button onClick={onBack} className="text-[14px] text-echo-text-muted hover:text-echo-text inline-flex items-center gap-1">
        <ArrowLeft size={13} />Back to Meeting Notes
      </button>

      <section className="bg-echo-surface border border-echo-border rounded-lg p-5">
        <div className="flex items-start gap-5">
          <div className="flex-1 min-w-0">
            <h1 className="text-[26px] text-echo-text leading-tight" style={{ fontWeight: 700 }}>{meeting.title}</h1>
            <div className="mt-4 grid grid-cols-1 md:grid-cols-3 gap-3 text-[14px]">
              <MetadataItem icon={<Calendar size={15} />} label="Meeting date" value={formatDateTimeNullable(meeting.meeting_occurred_at)} />
              <MetadataItem icon={<FileText size={15} />} label="MoM generated" value={formatDateTimeNullable(meeting.mom_generated_at || selectedMom?.created_at)} />
              <SourceItem source={meeting.source_info} fallback={meeting.source_label || meeting.source_type} />
            </div>
          </div>
        </div>
      </section>

      <div className="grid grid-cols-1 xl:grid-cols-[1fr_340px] gap-5">
        <div className="space-y-3 min-w-0">
          <div className="bg-echo-surface border border-echo-border rounded-lg px-2 overflow-hidden">
            <div className="flex gap-1">
              {tabs.map((item) => (
                <button
                  key={item}
                  onClick={() => setTab(item)}
                  className={`px-3.5 py-3 text-[14px] whitespace-nowrap border-b-2 -mb-px ${tab === item ? "border-echo-accent text-echo-accent-fg" : "border-transparent text-echo-text-muted hover:text-echo-text"}`}
                >
                  {item}
                </button>
              ))}
            </div>
          </div>

          <section className="bg-echo-surface border border-echo-border rounded-lg p-6 min-h-[420px]">
            {tab === "MoM" && <MomTab mom={selectedMom} onCopy={handleCopyRich} />}
            {tab === "Transcript" && <TranscriptTab transcript={detail.transcript} onCopy={() => handleCopy(detail.transcript?.text || "", "Transcript copied.")} />}
            {tab === "Versions" && (
              <VersionsTab
                versions={detail.mom_versions}
                selectedId={selectedVersionId}
                expandedIds={expandedVersionIds}
                onSelect={setSelectedVersionId}
                onToggle={toggleVersionExpanded}
                onView={handleViewVersion}
              />
            )}
            {tab === "Exports" && <ExportsTab meetingId={meetingId} mom={selectedMom} onCopy={handleCopyRich} />}
          </section>
        </div>

        <aside className="space-y-4">
          <section className="bg-echo-surface border border-echo-border rounded-lg">
            <div className="px-4 py-3 border-b border-echo-border text-[16px] text-echo-text" style={{ fontWeight: 700 }}>Regenerate</div>
            <div className="p-4 space-y-3 text-[14px]">
              <SelectField label="Template" value={templateName} options={templateOptions} onChange={setTemplateName} />
              <SelectField label="Summary backend" value={backendKind} options={backendOptions} onChange={setBackendKind} />
              <button
                onClick={handleRegenerate}
                disabled={regenerating || !detail.transcript}
                className="w-full h-10 rounded-md bg-echo-accent text-white inline-flex items-center justify-center gap-1.5 disabled:opacity-55 disabled:cursor-not-allowed"
              >
                <Sparkles size={15} />{regenerating ? "Regenerating..." : "Regenerate MoM"}
              </button>
              <p className="text-[13px] text-echo-text-muted">
                Creates a new version from the saved transcript using the selected template and backend.
              </p>
            </div>
          </section>

          <section className="bg-echo-surface border border-echo-border rounded-lg">
            <div className="px-4 py-3 border-b border-echo-border text-[16px] text-echo-text" style={{ fontWeight: 700 }}>Version history</div>
            <VersionList versions={detail.mom_versions} selectedId={selectedVersionId} onSelect={setSelectedVersionId} />
          </section>

          {message && (
            <div className="bg-echo-surface border border-echo-border rounded-lg p-4 text-[14px] text-echo-text-muted">
              {message}
            </div>
          )}
        </aside>
      </div>
    </div>
  );
}

function EmptyShell({ onBack, title, children }: { onBack: () => void; title: string; children: ReactNode }) {
  return (
    <div className="space-y-5">
      <button onClick={onBack} className="text-[14px] text-echo-text-muted hover:text-echo-text inline-flex items-center gap-1"><ArrowLeft size={13} />Back to Meeting Notes</button>
      <section className="bg-echo-surface border border-echo-border rounded-lg p-10 text-center">
        <h1 className="text-[22px] text-echo-text" style={{ fontWeight: 700 }}>{title}</h1>
        <p className="text-[15px] text-echo-text-muted mt-2">{children}</p>
      </section>
    </div>
  );
}

function MetadataItem({ icon, label, value }: { icon: ReactNode; label: string; value: string }) {
  return (
    <div className="rounded-md border border-echo-border bg-echo-surface-2 px-3 py-2.5 min-w-0">
      <div className="flex items-center gap-1.5 text-[12px] text-echo-text-faint uppercase tracking-wide">{icon}{label}</div>
      <div className="mt-1 text-[14px] text-echo-text truncate">{value}</div>
    </div>
  );
}

function SourceItem({ source, fallback }: { source?: MeetingDetail["meeting"]["source_info"]; fallback: string }) {
  const label = source?.label || "Source";
  const display = source?.display || fallback || "N/A";
  return (
    <div className="rounded-md border border-echo-border bg-echo-surface-2 px-3 py-2.5 min-w-0">
      <div className="flex items-center gap-1.5 text-[12px] text-echo-text-faint uppercase tracking-wide"><Link2 size={15} />{label}</div>
      {source?.href ? (
        <a href={source.href} target="_blank" rel="noreferrer" className="mt-1 text-[14px] text-echo-accent hover:text-echo-accent-hover truncate block">
          {display}
        </a>
      ) : (
        <div className="mt-1 text-[14px] text-echo-text truncate">{display || "N/A"}</div>
      )}
    </div>
  );
}

function SelectField({ label, value, options, onChange }: { label: string; value: string; options: Array<{ value: string; label: string }>; onChange: (value: string) => void }) {
  return (
    <label className="block">
      <span className="text-[12px] text-echo-text-muted mb-1 uppercase tracking-wide block">{label}</span>
      <select
        value={value}
        onChange={(event) => onChange(event.target.value)}
        className="w-full h-10 px-3 rounded-md border border-echo-border bg-echo-surface-2 text-[14px] text-echo-text focus:outline-none focus:bg-echo-surface focus:border-echo-accent"
      >
        {options.map((option) => (
          <option key={option.value} value={option.value}>{option.label}</option>
        ))}
      </select>
    </label>
  );
}

function MomTab({ mom, onCopy }: { mom: MomVersion | null; onCopy: (text: string, html: string, label?: string) => void }) {
  const text = momText(mom);
  const renderedRef = useRef<HTMLElement | null>(null);
  if (!mom) return <EmptyState title="No MoM generated yet" body="This meeting has no saved MoM version." />;
  return (
    <div>
      <div className="flex items-center justify-between gap-3 mb-4">
        <div>
          <h2 className="text-[18px] text-echo-text" style={{ fontWeight: 700 }}>MoM v{mom.version_number}</h2>
          <p className="text-[13px] text-echo-text-muted mt-0.5">Generated {formatDateTimeNullable(mom.created_at)}{mom.backend_kind ? ` using ${mom.backend_kind}` : ""}</p>
        </div>
        <button onClick={() => onCopy(text, renderedRef.current?.innerHTML || "", "MoM copied with formatting.")} className="h-9 px-3 rounded-md border border-echo-border bg-echo-surface text-[14px] text-echo-text inline-flex items-center gap-1.5 hover:bg-echo-surface-hover">
          <Copy size={14} />Copy MoM
        </button>
      </div>
      <article ref={renderedRef}>{text ? <MarkdownMom text={text} /> : <p className="text-[15px] text-echo-text">No MoM content was stored.</p>}</article>
    </div>
  );
}

function TranscriptTab({ transcript, onCopy }: { transcript: MeetingDetail["transcript"]; onCopy: () => void }) {
  const [query, setQuery] = useState("");
  const text = transcript?.text || "";
  const lines = useMemo(() => {
    const parts = text.split(/\n+/).map((line) => line.trim()).filter(Boolean);
    const filtered = query.trim()
      ? parts.filter((line) => line.toLowerCase().includes(query.trim().toLowerCase()))
      : parts;
    return filtered.length ? filtered : (text ? [text] : []);
  }, [text, query]);

  if (!transcript) return <EmptyState title="No transcript stored" body="This meeting has no transcript row in the local database." />;
  return (
    <div>
      <div className="flex items-center gap-2 mb-4">
        <div className="flex-1 relative">
          <Search size={14} className="absolute left-3 top-1/2 -translate-y-1/2 text-echo-text-faint" />
          <input value={query} onChange={(event) => setQuery(event.target.value)} placeholder="Search this transcript..." className="w-full h-10 pl-10 pr-3 rounded-md border border-echo-border bg-echo-surface-2 text-[14px] text-echo-text focus:outline-none focus:bg-echo-surface focus:border-echo-accent" />
        </div>
        <button onClick={onCopy} className="h-10 px-4 rounded-md border border-echo-border bg-echo-surface text-[14px] text-echo-text inline-flex items-center gap-1.5 hover:bg-echo-surface-hover"><Copy size={14} />Copy</button>
      </div>
      <div className="rounded-md border border-echo-border bg-echo-surface-2 p-4 max-h-[560px] overflow-auto">
        <ul className="space-y-2">
          {lines.map((line, i) => (
            <li key={`${i}-${line.slice(0, 20)}`} className="text-[14px] text-echo-text whitespace-pre-wrap">{line}</li>
          ))}
        </ul>
      </div>
    </div>
  );
}

function VersionsTab({
  versions,
  selectedId,
  expandedIds,
  onSelect,
  onToggle,
  onView,
}: {
  versions: MomVersion[];
  selectedId: string;
  expandedIds: string[];
  onSelect: (id: string) => void;
  onToggle: (id: string) => void;
  onView: (id: string) => void;
}) {
  if (versions.length === 0) return <EmptyState title="No versions" body="No MoM versions are stored for this meeting." />;
  return (
    <div className="space-y-3">
      {versions.map((version, index) => {
        const expanded = expandedIds.includes(version.id);
        const isCurrent = index === 0;
        const isSelected = selectedId === version.id;
        return (
          <div
            key={version.id}
            className={`border rounded-md ${isCurrent ? "border-echo-accent bg-echo-accent-bg" : isSelected ? "border-echo-border-strong bg-echo-surface-2" : "border-echo-border bg-echo-surface"}`}
          >
            <div className="p-4">
              <div className="flex items-start justify-between gap-3">
                <button onClick={() => onSelect(version.id)} className="text-left min-w-0">
                  <div className="text-[15px] text-echo-text" style={{ fontWeight: 650 }}>Version {version.version_number}{isCurrent ? " - current" : ""}</div>
                  <div className="text-[13px] text-echo-text-muted mt-0.5">{formatDateTimeNullable(version.created_at)}{version.backend_kind ? ` using ${version.backend_kind}` : ""}</div>
                </button>
                <div className="flex items-center gap-2 shrink-0">
                  <button onClick={() => onView(version.id)} className="h-8 px-3 rounded-md border border-echo-border bg-echo-surface text-[13px] text-echo-text hover:bg-echo-surface-hover">
                    View in MoM
                  </button>
                  <button onClick={() => onToggle(version.id)} className="h-8 px-3 rounded-md border border-echo-border bg-echo-surface text-[13px] text-echo-text inline-flex items-center gap-1 hover:bg-echo-surface-hover">
                    {expanded ? "Collapse" : "Expand"} <ChevronDown size={14} className={expanded ? "rotate-180" : ""} />
                  </button>
                </div>
              </div>
              {!expanded && <p className="text-[14px] text-echo-text-muted mt-2 line-clamp-2">{markdownSnippet(version.summary || version.content_markdown || "No summary stored.")}</p>}
            </div>
            {expanded && (
              <div className="border-t border-echo-border p-4 bg-echo-surface">
                <MarkdownMom text={momText(version)} compact />
              </div>
            )}
          </div>
        );
      })}
    </div>
  );
}

function ExportsTab({ meetingId, mom, onCopy }: { meetingId: string; mom: MomVersion | null; onCopy: (text: string, html: string, label?: string) => void }) {
  const renderedRef = useRef<HTMLDivElement | null>(null);
  const text = momText(mom);
  if (!mom) return <EmptyState title="No export available" body="Generate a MoM before exporting." />;
  return (
    <>
    <div ref={renderedRef} className="hidden"><MarkdownMom text={text} /></div>
    <div className="grid grid-cols-1 md:grid-cols-2 gap-3 text-[14px]">
      <ExportLink href={meetingExportUrl(meetingId, "pdf")} icon={<FileType2 size={16} />} title="Download PDF" body="Readable PDF copy with cleaned notes." />
      <ExportLink href={meetingExportUrl(meetingId, "html")} icon={<Download size={16} />} title="Open rendered HTML" body="Rendered headings, tables, and lists in a browser." />
      <ExportLink href={meetingExportUrl(meetingId, "email")} icon={<Mail size={16} />} title="Email draft" body="Download a formatted .eml draft with HTML and text." />
      <ExportLink href={meetingExportUrl(meetingId, "text")} icon={<Download size={16} />} title="Download text" body="Clean plain text copy for editing or archiving." />
      <button onClick={() => onCopy(text, renderedRef.current?.innerHTML || "", "Full MoM copied with formatting.")} className="flex items-center justify-between px-4 py-3 border border-echo-border rounded-md text-left hover:bg-echo-surface-hover">
        <div>
          <div className="text-echo-text" style={{ fontWeight: 600 }}>Copy all MoM</div>
          <div className="text-[14px] text-echo-text-muted mt-0.5">Copy selected version with rich formatting when supported.</div>
        </div>
        <Copy size={16} />
      </button>
    </div>
    </>
  );
}

function ExportLink({ href, icon, title, body }: { href: string; icon: ReactNode; title: string; body: string }) {
  return (
    <a href={href} target="_blank" rel="noreferrer" className="flex items-center justify-between px-4 py-3 border border-echo-border rounded-md text-left hover:bg-echo-surface-hover">
      <div>
        <div className="text-echo-text" style={{ fontWeight: 600 }}>{title}</div>
        <div className="text-[14px] text-echo-text-muted mt-0.5">{body}</div>
      </div>
      {icon}
    </a>
  );
}

function VersionList({ versions, selectedId, onSelect }: { versions: MomVersion[]; selectedId: string; onSelect: (id: string) => void }) {
  if (versions.length === 0) return <div className="p-4 text-[14px] text-echo-text-muted">No versions stored.</div>;
  return (
    <ul className="p-2 text-[14px]">
      {versions.map((version, index) => (
        <li key={version.id}>
          <button onClick={() => onSelect(version.id)} className={`w-full flex items-center justify-between px-2 py-2 rounded text-left ${index === 0 ? "bg-echo-accent-bg text-echo-accent-fg" : selectedId === version.id ? "bg-echo-surface-2 text-echo-text" : "hover:bg-echo-surface-hover text-echo-text"}`}>
            <div>
              <div>v{version.version_number}{index === 0 && <span className="ml-1.5 text-[12px]">current</span>}</div>
              <div className="text-[13px] text-echo-text-muted">{formatDateTimeNullable(version.created_at)}</div>
            </div>
            <ChevronRight size={13} className="text-echo-text-faint" />
          </button>
        </li>
      ))}
    </ul>
  );
}

function EmptyState({ title, body }: { title: string; body: string }) {
  return (
    <div className="h-[300px] grid place-items-center text-center">
      <div>
        <h3 className="text-[18px] text-echo-text" style={{ fontWeight: 700 }}>{title}</h3>
        <p className="text-[14px] text-echo-text-muted mt-1 max-w-md">{body}</p>
      </div>
    </div>
  );
}

function MarkdownMom({ text, compact = false }: { text: string; compact?: boolean }) {
  return (
    <div className={`text-echo-text ${compact ? "text-[14px]" : "text-[15px]"} leading-relaxed`}>
      <ReactMarkdown
        remarkPlugins={[remarkGfm]}
        components={{
          h1: ({ children }) => <h1 className="text-[22px] text-echo-text mt-5 mb-3 leading-snug" style={{ fontWeight: 700 }}>{children}</h1>,
          h2: ({ children }) => <h2 className="text-[20px] text-echo-text mt-5 mb-3 leading-snug" style={{ fontWeight: 700 }}>{children}</h2>,
          h3: ({ children }) => <h3 className="text-[18px] text-echo-text mt-4 mb-2 leading-snug" style={{ fontWeight: 700 }}>{children}</h3>,
          h4: ({ children }) => <h4 className="text-[16px] text-echo-text mt-4 mb-2" style={{ fontWeight: 700 }}>{children}</h4>,
          p: ({ children }) => <p className="my-2">{children}</p>,
          strong: ({ children }) => <strong style={{ fontWeight: 700 }}>{children}</strong>,
          ul: ({ children }) => <ul className="my-3 list-disc pl-6 space-y-1.5">{children}</ul>,
          ol: ({ children }) => <ol className="my-3 list-decimal pl-6 space-y-1.5">{children}</ol>,
          li: ({ children }) => <li className="pl-1">{children}</li>,
          a: ({ href, children }) => <a href={href} target="_blank" rel="noreferrer" className="text-echo-accent hover:text-echo-accent-hover underline underline-offset-2">{children}</a>,
          table: ({ children }) => (
            <div className="my-4 overflow-x-auto rounded-md border border-echo-border">
              <table className="w-full border-collapse text-[14px]">{children}</table>
            </div>
          ),
          thead: ({ children }) => <thead className="bg-echo-surface-2 text-echo-text">{children}</thead>,
          th: ({ children }) => <th className="border-b border-r border-echo-border px-3 py-2 text-left align-top last:border-r-0" style={{ fontWeight: 700 }}>{children}</th>,
          td: ({ children }) => <td className="border-b border-r border-echo-border px-3 py-2 align-top text-echo-text-muted last:border-r-0">{children}</td>,
          tr: ({ children }) => <tr className="last:[&_td]:border-b-0">{children}</tr>,
          blockquote: ({ children }) => <blockquote className="my-4 border-l-2 border-echo-accent pl-4 text-echo-text-muted">{children}</blockquote>,
          code: ({ children }) => <code className="rounded bg-echo-surface-2 px-1.5 py-0.5 text-[13px] text-echo-text">{children}</code>,
          pre: ({ children }) => <pre className="my-4 overflow-x-auto rounded-md bg-echo-surface-2 p-4 text-[13px]">{children}</pre>,
        }}
      >
        {text}
      </ReactMarkdown>
    </div>
  );
}

function momText(mom: MomVersion | null) {
  return mom?.content_markdown || mom?.summary || "";
}

function markdownSnippet(value: string) {
  return value
    .replace(/```[\s\S]*?```/g, " ")
    .replace(/`([^`]+)`/g, "$1")
    .replace(/\*\*([^*]+)\*\*/g, "$1")
    .replace(/__([^_]+)__/g, "$1")
    .replace(/^#+\s*/gm, "")
    .replace(/^\s*[-*]\s+/gm, "")
    .replace(/\|/g, " ")
    .replace(/\s+/g, " ")
    .trim();
}

function htmlDocumentFragment(innerHtml: string) {
  return `
    <div style="font-family: Inter, Arial, sans-serif; color: #111318; line-height: 1.55;">
      ${innerHtml}
    </div>
  `;
}

function templateNames(templates: EchoTemplate[], meetingType?: string, current?: string): Array<{ value: string; label: string }> {
  const matching = meetingType ? templates.filter((template) => template.meeting_type === meetingType) : [];
  const ordered = matching.length ? matching : templates;
  const names = ordered.map((template) => template.name).filter(Boolean);
  if (current && !names.includes(current)) names.unshift(current);
  return names.map((name) => ({ value: name, label: name }));
}

function backendProfiles(settings: EchoSettings | null, current?: string): Array<{ value: string; label: string }> {
  const backends = settings?.backends || {};
  const options = Object.entries(backends)
    .filter(([, config]: [string, any]) => config?.enabled)
    .map(([kind, config]: [string, any]) => ({ value: kind, label: config?.name || kind }));
  if (current && !options.some((option) => option.value === current)) {
    options.unshift({ value: current, label: current });
  }
  return options.length ? options : [{ value: current || "local", label: current || "Local" }];
}

function formatDateTimeNullable(value?: string | null) {
  if (!value) return "N/A";
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return value || "N/A";
  return date.toLocaleString(undefined, { month: "short", day: "2-digit", year: "numeric", hour: "2-digit", minute: "2-digit" });
}
