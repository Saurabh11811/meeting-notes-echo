import { useEffect, useMemo, useState, type ReactNode } from "react";
import { ArrowLeft, Share2, Download, RefreshCw, Calendar, Link2, FileText, ChevronRight, Copy, Mail, FileType2, Sparkles, AlertTriangle, Search, Clock, CheckCircle2 } from "lucide-react";
import { getMeetingDetail, type MeetingDetail, type MomVersion } from "../../api/echo-api";

const tabs = ["Executive Summary", "Decisions", "Action Items", "Risks & Blockers", "Full MoM", "Transcript", "Versions", "Exports"];

export function MeetingDetailPage({ meetingId, onBack }: { meetingId: string | null; onBack: () => void }) {
  const [tab, setTab] = useState("Executive Summary");
  const [detail, setDetail] = useState<MeetingDetail | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  useEffect(() => {
    if (!meetingId) {
      setDetail(null);
      setError("");
      return;
    }
    setLoading(true);
    getMeetingDetail(meetingId)
      .then((data) => {
        setDetail(data);
        setError("");
      })
      .catch((err) => {
        setDetail(null);
        setError(err instanceof Error ? err.message : "Could not load meeting.");
      })
      .finally(() => setLoading(false));
  }, [meetingId]);

  const meeting = detail?.meeting;
  const latestMom = detail?.latest_mom;

  if (!meetingId) {
    return (
      <EmptyShell onBack={onBack} title="No meeting selected">
        Open a completed meeting from Home or Meeting Notes to review the generated MoM.
      </EmptyShell>
    );
  }

  if (loading) {
    return (
      <EmptyShell onBack={onBack} title="Loading meeting">
        Reading meeting notes, transcript, and job history from the local database.
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
      <div>
        <button onClick={onBack} className="text-[12px] text-echo-text-muted hover:text-echo-text inline-flex items-center gap-1 mb-3"><ArrowLeft size={13} />Back to Meeting Notes</button>

        <div className="bg-echo-surface border border-echo-border rounded-lg p-5">
          <div className="flex items-start gap-4 flex-wrap">
            <div className="flex-1 min-w-[300px]">
              <div className="flex items-center gap-2 mb-1.5 flex-wrap">
                <span className="text-[11px] px-1.5 py-0.5 rounded bg-echo-surface-2 text-echo-text-muted">{meeting.meeting_type}</span>
                <span className="text-[11px] px-1.5 py-0.5 rounded bg-echo-surface-2 text-echo-text-muted">{meeting.confidentiality}</span>
                <span className="text-[11px] px-1.5 py-0.5 rounded bg-echo-surface-2 text-echo-text-muted">{meeting.status}</span>
              </div>
              <h1 className="text-[20px] text-echo-text" style={{ fontWeight: 600 }}>{meeting.title}</h1>
              <div className="mt-2 flex items-center gap-4 text-[12px] text-echo-text-muted flex-wrap">
                <span className="inline-flex items-center gap-1"><Calendar size={12} />{formatDateTime(meeting.updated_at || meeting.created_at)}</span>
                <span className="inline-flex items-center gap-1"><Link2 size={12} />{meeting.source_label || meeting.source_type}</span>
                <span className="inline-flex items-center gap-1"><FileText size={12} />{detail.transcript ? "Transcript available" : "No transcript stored"}</span>
                <span>v{latestMom?.version_number || 0} {latestMom ? `via ${latestMom.backend_kind || "backend"}` : ""}</span>
              </div>
            </div>
            <div className="flex items-center gap-2">
              <button disabled className="h-9 px-3 rounded-md border border-echo-border bg-echo-surface text-[12px] text-echo-text-muted inline-flex items-center gap-1.5 cursor-not-allowed"><Share2 size={13} />Share</button>
              <button disabled className="h-9 px-3 rounded-md border border-echo-border bg-echo-surface text-[12px] text-echo-text-muted inline-flex items-center gap-1.5 cursor-not-allowed"><Download size={13} />Export</button>
              <button disabled className="h-9 px-4 rounded-md bg-echo-accent text-white/80 text-[12px] inline-flex items-center gap-1.5 cursor-not-allowed"><RefreshCw size={13} />Regenerate</button>
            </div>
          </div>
        </div>
      </div>

      <div className="grid grid-cols-1 xl:grid-cols-[1fr_320px] gap-5">
        <div className="space-y-3 min-w-0">
          <div className="bg-echo-surface border border-echo-border rounded-lg px-2 overflow-x-auto">
            <div className="flex gap-1">
              {tabs.map((t) => (
                <button key={t} onClick={() => setTab(t)} className={`px-3 py-2.5 text-[12px] whitespace-nowrap border-b-2 -mb-px ${tab === t ? "border-echo-accent text-echo-accent-fg" : "border-transparent text-echo-text-muted hover:text-echo-text"}`}>{t}</button>
              ))}
            </div>
          </div>

          <section className="bg-echo-surface border border-echo-border rounded-lg p-6 min-h-[420px]">
            {tab === "Executive Summary" && <ExecutiveSummary mom={latestMom} />}
            {tab === "Decisions" && <DecisionsTab decisions={detail.decisions} />}
            {tab === "Action Items" && <ActionItemsTab items={detail.action_items} />}
            {tab === "Risks & Blockers" && <RisksTab risks={detail.risks} />}
            {tab === "Full MoM" && <FullMoMTab mom={latestMom} />}
            {tab === "Transcript" && <TranscriptTab transcript={detail.transcript} />}
            {tab === "Versions" && <VersionsTab versions={detail.mom_versions} />}
            {tab === "Exports" && <ExportsTab />}
          </section>
        </div>

        <aside className="space-y-4">
          <section className="bg-echo-surface border border-echo-border rounded-lg">
            <div className="px-4 py-3 border-b border-echo-border text-[12px] text-echo-text" style={{ fontWeight: 600 }}>Regenerate</div>
            <div className="p-4 space-y-3 text-[12px]">
              <InfoField label="Template" value={meeting.meeting_type || "Default"} />
              <InfoField label="Summary backend" value={latestMom?.backend_kind || "Not generated"} />
              <button disabled className="w-full h-9 rounded-md bg-echo-text/80 text-echo-surface inline-flex items-center justify-center gap-1.5 cursor-not-allowed"><Sparkles size={13} />Regenerate MoM</button>
              <p className="text-[11px] text-echo-text-muted">Regenerate from saved transcript is the next workflow to wire.</p>
            </div>
          </section>

          <section className="bg-echo-surface border border-echo-border rounded-lg">
            <div className="px-4 py-3 border-b border-echo-border text-[12px] text-echo-text" style={{ fontWeight: 600 }}>Version history</div>
            <VersionList versions={detail.mom_versions} />
          </section>

          <section className="bg-echo-surface border border-echo-border rounded-lg">
            <div className="px-4 py-3 border-b border-echo-border text-[12px] text-echo-text" style={{ fontWeight: 600 }}>Job log</div>
            <JobLog jobs={detail.jobs} />
          </section>
        </aside>
      </div>
    </div>
  );
}

function EmptyShell({ onBack, title, children }: { onBack: () => void; title: string; children: ReactNode }) {
  return (
    <div className="space-y-5">
      <button onClick={onBack} className="text-[12px] text-echo-text-muted hover:text-echo-text inline-flex items-center gap-1"><ArrowLeft size={13} />Back to Meeting Notes</button>
      <section className="bg-echo-surface border border-echo-border rounded-lg p-10 text-center">
        <h1 className="text-[18px] text-echo-text" style={{ fontWeight: 600 }}>{title}</h1>
        <p className="text-[13px] text-echo-text-muted mt-2">{children}</p>
      </section>
    </div>
  );
}

function InfoField({ label, value }: { label: string; value: string }) {
  return (
    <div>
      <div className="text-[10px] text-echo-text-muted mb-1 uppercase tracking-wide">{label}</div>
      <div className="w-full min-h-8 px-2 py-1.5 rounded-md border border-echo-border bg-echo-surface-2 text-[12px] text-echo-text">{value}</div>
    </div>
  );
}

function ExecutiveSummary({ mom }: { mom: MomVersion | null }) {
  if (!mom) return <EmptyState title="No MoM generated yet" body="This meeting has no saved summary version." />;
  return (
    <div className="space-y-5 text-[13px] text-echo-text leading-relaxed">
      <div>
        <h3 className="text-[11px] text-echo-text-muted uppercase tracking-wide mb-2">Brief summary</h3>
        <p className="whitespace-pre-wrap">{mom.summary || "No separate brief summary was stored for this MoM."}</p>
      </div>
      <div>
        <h3 className="text-[11px] text-echo-text-muted uppercase tracking-wide mb-2">Generated output</h3>
        <div className="rounded-md border border-echo-border bg-echo-surface-2 p-4 whitespace-pre-wrap max-h-[480px] overflow-auto">{mom.content_markdown || "No MoM content was stored."}</div>
      </div>
    </div>
  );
}

function DecisionsTab({ decisions }: { decisions: MeetingDetail["decisions"] }) {
  if (decisions.length === 0) return <EmptyState title="No structured decisions yet" body="The raw MoM is stored, but decision extraction has not produced separate rows for this meeting." />;
  return (
    <div className="divide-y divide-echo-border -mx-6 -my-6">
      {decisions.map((d, i) => (
        <div key={d.id} className="px-6 py-4">
          <div className="flex items-start gap-3">
            <div className="h-6 w-6 rounded-full bg-echo-accent-bg text-echo-accent-fg grid place-items-center text-[11px] shrink-0">{i + 1}</div>
            <div className="flex-1">
              <div className="text-[13px] text-echo-text" style={{ fontWeight: 500 }}>{d.description}</div>
              {d.context && <p className="text-[12px] text-echo-text-muted mt-1">{d.context}</p>}
              <div className="text-[11px] text-echo-text-muted mt-2">{d.source_ref ? <>Transcript reference <span className="font-mono">{d.source_ref}</span></> : "No transcript reference stored"}</div>
            </div>
          </div>
        </div>
      ))}
    </div>
  );
}

function ActionItemsTab({ items }: { items: MeetingDetail["action_items"] }) {
  if (items.length === 0) return <EmptyState title="No structured action items yet" body="The raw MoM is stored, but action item extraction has not produced separate rows for this meeting." />;
  return (
    <table className="w-full text-[12px] -mx-6 -my-6" style={{ width: "calc(100% + 3rem)" }}>
      <thead className="bg-echo-surface-2 text-echo-text-muted">
        <tr>
          <th className="text-left px-6 py-2.5" style={{ fontWeight: 500 }}>Action</th>
          <th className="text-left px-3 py-2.5" style={{ fontWeight: 500 }}>Owner</th>
          <th className="text-left px-3 py-2.5" style={{ fontWeight: 500 }}>Due</th>
          <th className="text-left px-6 py-2.5" style={{ fontWeight: 500 }}>Status</th>
        </tr>
      </thead>
      <tbody className="divide-y divide-echo-border">
        {items.map((a) => (
          <tr key={a.id} className="hover:bg-echo-surface-hover">
            <td className="px-6 py-3 text-echo-text">{a.description}</td>
            <td className="px-3 py-3 text-echo-text-muted">{a.owner || "Unassigned"}</td>
            <td className="px-3 py-3 text-echo-text-muted">{a.due_date || "No due date"}</td>
            <td className="px-6 py-3 text-echo-text">{a.status}</td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}

function RisksTab({ risks }: { risks: MeetingDetail["risks"] }) {
  if (risks.length === 0) return <EmptyState title="No structured risks yet" body="The raw MoM is stored, but risk extraction has not produced separate rows for this meeting." />;
  return (
    <ul className="space-y-3">
      {risks.map((r) => (
        <li key={r.id} className="border border-echo-border rounded-md p-4 flex items-start gap-3 bg-echo-surface">
          <div className="h-8 w-8 rounded-md bg-rose-500/[0.12] grid place-items-center text-echo-danger shrink-0"><AlertTriangle size={14} /></div>
          <div className="flex-1">
            <div className="flex items-center gap-2">
              <div className="text-[13px] text-echo-text">{r.description}</div>
              <span className="text-[10px] px-1.5 py-0.5 rounded bg-amber-500/[0.15] text-echo-warning">{r.severity}</span>
            </div>
            {r.mitigation && <div className="text-[12px] text-echo-text-muted mt-1"><span className="text-echo-text-faint">Mitigation:</span> {r.mitigation}</div>}
          </div>
        </li>
      ))}
    </ul>
  );
}

function FullMoMTab({ mom }: { mom: MomVersion | null }) {
  if (!mom) return <EmptyState title="No full MoM stored" body="Generate a MoM first, then the saved markdown will appear here." />;
  return <div className="text-[13px] text-echo-text leading-relaxed whitespace-pre-wrap">{mom.content_markdown || "No MoM content was stored."}</div>;
}

function TranscriptTab({ transcript }: { transcript: MeetingDetail["transcript"] }) {
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
          <Search size={13} className="absolute left-3 top-1/2 -translate-y-1/2 text-echo-text-faint" />
          <input value={query} onChange={(event) => setQuery(event.target.value)} placeholder="Search this transcript..." className="w-full h-9 pl-9 pr-3 rounded-md border border-echo-border bg-echo-surface-2 text-[12px] text-echo-text focus:outline-none focus:bg-echo-surface focus:border-echo-accent" />
        </div>
        <button disabled className="h-9 px-3 rounded-md bg-echo-text/80 text-echo-surface text-[12px] inline-flex items-center gap-1.5 cursor-not-allowed"><Sparkles size={12} />Regenerate from transcript</button>
      </div>
      <div className="rounded-md border border-echo-border bg-echo-surface-2 p-4 max-h-[560px] overflow-auto">
        <ul className="space-y-2">
          {lines.map((line, i) => (
            <li key={`${i}-${line.slice(0, 20)}`} className="text-[13px] text-echo-text whitespace-pre-wrap">{line}</li>
          ))}
        </ul>
      </div>
    </div>
  );
}

function VersionsTab({ versions }: { versions: MomVersion[] }) {
  if (versions.length === 0) return <EmptyState title="No versions" body="No MoM versions are stored for this meeting." />;
  return (
    <div className="text-[13px] text-echo-text space-y-2">
      {versions.map((version, index) => (
        <div key={version.id} className="border border-echo-border rounded-md p-3 bg-echo-surface">
          <div className="flex items-center justify-between gap-3">
            <div><span className="text-echo-text">v{version.version_number}{index === 0 ? " (current)" : ""}</span> - {formatDateTime(version.created_at)} - {version.backend_kind || "backend"}</div>
            <button disabled className="text-[11px] text-echo-text-muted cursor-not-allowed">Compare</button>
          </div>
          <p className="text-[12px] text-echo-text-muted mt-1">{version.summary || "No separate summary stored."}</p>
        </div>
      ))}
    </div>
  );
}

function ExportsTab() {
  return (
    <div className="grid grid-cols-2 gap-3 text-[12px]">
      {[
        ["PDF", <FileType2 size={13} />],
        ["DOCX", <FileText size={13} />],
        ["Email draft", <Mail size={13} />],
        ["Copy summary", <Copy size={13} />],
      ].map(([label, icon]) => (
        <button key={String(label)} disabled className="flex items-center justify-between px-4 py-3 border border-echo-border rounded-md text-left opacity-60 cursor-not-allowed">
          <div>
            <div className="text-echo-text">{label}</div>
            <div className="text-[11px] text-echo-text-muted mt-0.5">Export workflow is not wired yet</div>
          </div>
          {icon}
        </button>
      ))}
    </div>
  );
}

function VersionList({ versions }: { versions: MomVersion[] }) {
  if (versions.length === 0) return <div className="p-4 text-[12px] text-echo-text-muted">No versions stored.</div>;
  return (
    <ul className="p-2 text-[12px]">
      {versions.map((v, index) => (
        <li key={v.id} className={`flex items-center justify-between px-2 py-2 rounded ${index === 0 ? "bg-echo-accent-bg" : "hover:bg-echo-surface-hover"}`}>
          <div>
            <div className="text-echo-text">v{v.version_number}{index === 0 && <span className="ml-1.5 text-[10px] text-echo-accent-fg">current</span>}</div>
            <div className="text-[10px] text-echo-text-muted">{formatDateTime(v.created_at)}</div>
          </div>
          <ChevronRight size={13} className="text-echo-text-faint" />
        </li>
      ))}
    </ul>
  );
}

function JobLog({ jobs }: { jobs: MeetingDetail["jobs"] }) {
  if (jobs.length === 0) return <div className="p-4 text-[12px] text-echo-text-muted">No job history stored.</div>;
  const events = jobs.flatMap((job) => job.events.map((event) => ({ ...event, jobId: job.id })));
  if (events.length === 0) return <div className="p-4 text-[12px] text-echo-text-muted">No detailed job events stored.</div>;
  return (
    <ol className="p-3 space-y-2">
      {events.map((event, index) => (
        <li key={`${event.jobId}-${event.created_at}-${index}`} className="flex items-start gap-2 text-[11.5px]">
          <span className={`mt-0.5 ${event.level === "error" ? "text-echo-danger" : event.progress >= 100 ? "text-echo-success" : "text-echo-accent"}`}>
            {event.level === "error" ? <AlertTriangle size={12} /> : event.progress >= 100 ? <CheckCircle2 size={12} /> : <Clock size={12} />}
          </span>
          <div className="flex-1">
            <div className="text-echo-text">{event.message}</div>
            <div className="text-[10px] text-echo-text-faint">{formatDateTime(event.created_at)} - {event.stage} - {event.progress}%</div>
          </div>
        </li>
      ))}
    </ol>
  );
}

function EmptyState({ title, body }: { title: string; body: string }) {
  return (
    <div className="h-[300px] grid place-items-center text-center">
      <div>
        <h3 className="text-[14px] text-echo-text" style={{ fontWeight: 600 }}>{title}</h3>
        <p className="text-[12px] text-echo-text-muted mt-1 max-w-md">{body}</p>
      </div>
    </div>
  );
}

function formatDateTime(value: string) {
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return value || "unknown";
  return date.toLocaleString(undefined, { month: "short", day: "2-digit", hour: "2-digit", minute: "2-digit" });
}
