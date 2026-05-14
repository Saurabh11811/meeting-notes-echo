import { useEffect, useMemo, useState } from "react";
import { Link2, Upload, FileText, Sparkles, ChevronDown, Wand2, Mic, Clock, AlertTriangle, X, Plus, Layers, CheckCircle2 } from "lucide-react";
import { createJobs, createUploadJobs, getHomeSummary, type HomeSummary } from "../../api/echo-api";

const tabs = [
  { id: "links", label: "Meeting Links", icon: Link2 },
  { id: "upload", label: "Upload Recordings", icon: Upload },
  { id: "transcript", label: "Existing Transcript", icon: FileText },
];

const toneBar: Record<string, string> = {
  accent: "bg-echo-accent",
  info: "bg-echo-info",
  muted: "bg-echo-text-faint",
  warning: "bg-echo-warning",
  success: "bg-echo-success",
};
const toneText: Record<string, string> = {
  accent: "text-echo-accent",
  info: "text-echo-info",
  muted: "text-echo-text-muted",
  warning: "text-echo-warning",
  success: "text-echo-success",
};

export function HomePage({ onOpenReview }: { onOpenReview: (meetingId: string) => void }) {
  const [home, setHome] = useState<HomeSummary | null>(null);

  const refreshHome = () => {
    return getHomeSummary()
      .then(setHome)
      .catch(() => setHome(null));
  };

  useEffect(() => {
    refreshHome();
    const interval = window.setInterval(refreshHome, 5000);
    return () => {
      window.clearInterval(interval);
    };
  }, []);

  const jobs = useMemo(() => {
    if (!home?.active_jobs?.length) return [];
    return home.active_jobs.map((job) => ({
      id: job.id,
      title: job.title || "Untitled meeting",
      status: job.stage,
      progress: job.progress,
      icon: job.status === "failed" ? AlertTriangle : job.stage.toLowerCase().includes("transcrib") ? Mic : job.status === "queued" ? Clock : Wand2,
      tone: job.status === "failed" ? "warning" : job.status === "queued" ? "muted" : "accent",
      events: job.events || [],
      errorMessage: job.error_message || "",
      sourceType: job.source_type,
      templateName: job.template_name || "",
      updatedAt: job.updated_at,
    }));
  }, [home]);

  const recentRows = useMemo(() => {
    if (!home?.recent_meetings?.length) return [];
    return home.recent_meetings.map((meeting) => ({
      id: meeting.id,
      title: meeting.title,
      date: formatDate(meeting.updated_at || meeting.created_at),
      type: meeting.meeting_type,
      status: meeting.status,
      tone: meeting.status.toLowerCase().includes("attention") ? "warning" : meeting.status.toLowerCase().includes("review") ? "accent" : "success",
    }));
  }, [home]);

  return (
    <div className="space-y-5">
      <CreateBlock onCreated={refreshHome} />

      <div className="grid grid-cols-1 xl:grid-cols-5 gap-5">
        <InProgressCard className="xl:col-span-2" jobs={jobs} />
        <RecentCard className="xl:col-span-3" rows={recentRows} onOpen={onOpenReview} />
      </div>
    </div>
  );
}

function CreateBlock({ onCreated }: { onCreated: () => Promise<void> }) {
  const [active, setActive] = useState("links");
  const [bulk, setBulk] = useState(false);
  const [files, setFiles] = useState<File[]>([]);
  const [linkText, setLinkText] = useState("");
  const [transcriptText, setTranscriptText] = useState("");
  const [meetingType, setMeetingType] = useState("Executive");
  const [message, setMessage] = useState("");
  const [submitting, setSubmitting] = useState(false);

  const handleFiles = (selected: FileList | null) => {
    if (!selected?.length) return;
    setFiles((current) => [...current, ...Array.from(selected)]);
  };

  const submit = async (runNow: boolean) => {
    const sourceType = active === "links" ? "url" : active === "upload" ? "upload" : "transcript";
    const sources =
      sourceType === "url"
        ? linkText.split(/\n+/).map((line) => line.trim()).filter(Boolean)
        : sourceType === "upload"
          ? files.map((file) => file.name)
          : [transcriptText.trim()].filter(Boolean);

    if (!sources.length) {
      setMessage("Add at least one link, recording, or transcript first.");
      return;
    }

    setSubmitting(true);
    setMessage("");
    try {
      const result = sourceType === "upload"
        ? await createUploadJobs({
            files,
            meeting_type: meetingType,
            run_now: runNow,
          })
        : await createJobs({
            source_type: sourceType,
            sources,
            meeting_type: meetingType,
            run_now: runNow,
          });
      setMessage(
        runNow
          ? `${result.jobs.length} item${result.jobs.length === 1 ? "" : "s"} started. A browser window may open for transcript capture.`
          : `${result.jobs.length} item${result.jobs.length === 1 ? "" : "s"} added to the queue.`
      );
      if (sourceType === "url") setLinkText("");
      if (sourceType === "upload") setFiles([]);
      if (sourceType === "transcript") setTranscriptText("");
      await onCreated();
    } catch (error) {
      setMessage(error instanceof Error ? error.message : "Could not add item to the queue.");
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <section className="bg-echo-surface border border-echo-border rounded-lg shadow-[var(--echo-shadow)]">
      <div className="px-6 pt-5 pb-4 flex items-start justify-between gap-4 border-b border-echo-border">
        <div>
          <h2 className="text-[16px] text-echo-text" style={{ fontWeight: 600 }}>Create meeting notes</h2>
          <p className="text-[12px] text-echo-text-muted mt-1">Paste a meeting link, upload a recording, or reuse a transcript.</p>
        </div>
        <label className="flex items-center gap-2 text-[12px] text-echo-text-muted cursor-pointer select-none">
          <span>Bulk mode</span>
          <button onClick={() => setBulk(!bulk)} className={`h-5 w-9 rounded-full transition-colors relative ${bulk ? "bg-echo-accent" : "bg-echo-border-strong"}`}>
            <span className={`absolute top-0.5 h-4 w-4 rounded-full bg-white shadow transition-all ${bulk ? "left-4" : "left-0.5"}`} />
          </button>
        </label>
      </div>

      <div className="px-6 pt-4">
        <div className="inline-flex p-1 rounded-md bg-echo-surface-2 border border-echo-border">
          {tabs.map((t) => {
            const Icon = t.icon;
            const isActive = active === t.id;
            return (
              <button key={t.id} onClick={() => setActive(t.id)} className={`flex items-center gap-2 px-3 py-1.5 rounded text-[12px] transition-colors ${isActive ? "bg-echo-surface text-echo-text shadow-sm" : "text-echo-text-muted hover:text-echo-text"}`}>
                <Icon size={13} />{t.label}
              </button>
            );
          })}
        </div>
      </div>

      <div className="px-6 py-4">
        {active === "links" && (
          <textarea
            value={linkText}
            onChange={(event) => setLinkText(event.target.value)}
            placeholder={bulk ? "Paste multiple meeting links, one per line…" : "Paste a meeting link…"}
            className="w-full h-24 p-3 rounded-md border border-echo-border bg-echo-surface-2 text-[13px] font-mono text-echo-text placeholder:text-echo-text-faint focus:outline-none focus:bg-echo-surface focus:border-echo-accent"
          />
        )}

        {active === "upload" && (
          <div>
            <label className="block border-2 border-dashed border-echo-border rounded-md py-8 px-6 bg-echo-surface-2 text-center hover:border-echo-accent hover:bg-echo-accent-bg/40 cursor-pointer transition-colors">
              <input
                type="file"
                multiple={bulk}
                accept="audio/*,video/*,.mp3,.wav,.m4a,.aac,.flac,.ogg,.wma,.mp4,.mkv,.mov,.avi,.webm"
                className="hidden"
                onChange={(event) => handleFiles(event.target.files)}
              />
              <div className="mx-auto h-9 w-9 rounded-full bg-echo-surface border border-echo-border grid place-items-center mb-2"><Upload size={14} className="text-echo-accent" /></div>
              <div className="text-[13px] text-echo-text">Drop {bulk ? "multiple " : ""}audio or video files</div>
              <div className="text-[11px] text-echo-text-muted mt-1">MP4, MP3, M4A, WAV, MOV{bulk ? " · multiple supported" : ""}</div>
            </label>
            {files.length > 0 && (
              <ul className="mt-3 space-y-1.5">
                {files.map((f, i) => (
                  <li key={i} className="flex items-center gap-2 px-3 py-2 rounded-md border border-echo-border bg-echo-surface-2 text-[12px]">
                    <FileText size={12} className="text-echo-text-muted" />
                    <span className="flex-1 truncate text-echo-text">{f.name}</span>
                    <span className="text-[10px] text-echo-text-faint">Ready</span>
                    <button onClick={() => setFiles(files.filter((_, idx) => idx !== i))} className="text-echo-text-faint hover:text-echo-danger"><X size={12} /></button>
                  </li>
                ))}
              </ul>
            )}
          </div>
        )}

        {active === "transcript" && (
          <textarea
            value={transcriptText}
            onChange={(event) => setTranscriptText(event.target.value)}
            placeholder="Paste transcript text or select a saved transcript…"
            className="w-full h-24 p-3 rounded-md border border-echo-border bg-echo-surface-2 text-[13px] text-echo-text placeholder:text-echo-text-faint focus:outline-none focus:bg-echo-surface focus:border-echo-accent"
          />
        )}
      </div>

      <div className="px-6 py-3 border-t border-echo-border bg-echo-surface-2/50 rounded-b-lg flex items-center gap-3 flex-wrap">
        <OptionSelect label="Type" value={meetingType} options={["Executive", "Project Review", "Client Call", "Townhall", "Demo/UAT", "Incident"]} onChange={setMeetingType} />

        <div className="w-full flex items-center gap-2 pt-1">
          <button disabled={submitting} onClick={() => submit(true)} className="inline-flex items-center gap-2 h-9 px-4 rounded-md bg-echo-accent hover:bg-echo-accent-hover text-white text-[13px] shadow-sm disabled:opacity-60" style={{ color: "#fff" }}>
            <Sparkles size={14} />
            {submitting ? "Adding…" : bulk ? "Generate all MoMs" : "Generate MoM"}
          </button>
          <button disabled={submitting} onClick={() => submit(false)} className="inline-flex items-center gap-2 h-9 px-3 rounded-md border border-echo-border bg-echo-surface text-echo-text text-[13px] hover:bg-echo-surface-hover disabled:opacity-60">
            <Plus size={14} />
            Add to queue
          </button>
          {message && <span className="text-[12px] text-echo-text-muted ml-auto">{message}</span>}
        </div>
      </div>
    </section>
  );
}

function OptionSelect({ label, value, options, onChange }: { label: string; value: string; options: string[]; onChange: (value: string) => void }) {
  return (
    <label className="flex items-center gap-2 h-8 px-2.5 rounded-md border border-echo-border bg-echo-surface hover:border-echo-border-strong text-[12px]">
      <span className="text-echo-text-muted">{label}:</span>
      <select value={value} onChange={(event) => onChange(event.target.value)} className="bg-transparent text-echo-text focus:outline-none">
        {options.map((option) => <option key={option} value={option}>{option}</option>)}
      </select>
      <ChevronDown size={11} className="text-echo-text-faint" />
    </label>
  );
}

type HomeJobView = {
  id: string;
  title: string;
  status: string;
  progress: number;
  icon: typeof Wand2;
  tone: string;
  events: HomeSummary["active_jobs"][number]["events"];
  errorMessage: string;
  sourceType: string;
  templateName: string;
  updatedAt: string;
};

function InProgressCard({ className = "", jobs }: { className?: string; jobs: HomeJobView[] }) {
  return (
    <section className={`bg-echo-surface border border-echo-border rounded-lg ${className}`}>
      <SectionHeader title="In progress" subtitle={`${jobs.length} jobs`} action="View activity" />
      {jobs.length === 0 ? (
        <div className="px-5 py-8 text-center text-[12px] text-echo-text-muted">
          No active jobs. Paste a meeting link or transcript above to start.
        </div>
      ) : (
        <ul className="divide-y divide-echo-border">
          {jobs.map((j) => {
          const Icon = j.icon;
          return (
            <li key={j.id} className="px-5 py-3 hover:bg-echo-surface-hover">
              <details>
                <summary className="list-none cursor-pointer">
                  <div className="flex items-center gap-3">
                    <div className="h-7 w-7 rounded-md bg-echo-surface-2 grid place-items-center text-echo-text-muted shrink-0"><Icon size={13} /></div>
                    <div className="flex-1 min-w-0">
                      <div className="text-[12.5px] text-echo-text truncate">{j.title}</div>
                      <div className={`text-[10.5px] mt-0.5 ${toneText[j.tone]}`}>{j.status}</div>
                    </div>
                    <div className="w-12 text-right text-[10px] text-echo-text-faint">{j.progress}%</div>
                    <ChevronDown size={13} className="text-echo-text-faint" />
                  </div>
                  <div className="mt-2 h-1 rounded-full bg-echo-surface-2 overflow-hidden ml-10">
                    <div className={`h-full ${toneBar[j.tone]}`} style={{ width: `${j.progress}%` }} />
                  </div>
                </summary>
                <div className="ml-10 mt-3 rounded-md border border-echo-border bg-echo-surface-2 p-3">
                  <div className="grid grid-cols-2 gap-2 text-[11px] text-echo-text-muted mb-3">
                    <div>Source: <span className="text-echo-text">{j.sourceType}</span></div>
                    <div>Template: <span className="text-echo-text">{j.templateName || "Default"}</span></div>
                    <div>Updated: <span className="text-echo-text">{formatTime(j.updatedAt)}</span></div>
                    <div>Status: <span className={toneText[j.tone]}>{j.status}</span></div>
                  </div>
                  {j.errorMessage && <div className="mb-3 text-[11.5px] text-echo-danger">{j.errorMessage}</div>}
                  {j.events.length === 0 ? (
                    <div className="text-[11.5px] text-echo-text-muted">Waiting for worker activity.</div>
                  ) : (
                    <ol className="space-y-2">
                      {j.events.map((event, index) => (
                        <li key={`${event.created_at}-${index}`} className="flex items-start gap-2 text-[11.5px]">
                          <span className={`mt-0.5 ${event.level === "error" ? "text-echo-danger" : event.progress >= 100 ? "text-echo-success" : "text-echo-accent"}`}>
                            {event.level === "error" ? <AlertTriangle size={12} /> : event.progress >= 100 ? <CheckCircle2 size={12} /> : <Clock size={12} />}
                          </span>
                          <div className="flex-1">
                            <div className="text-echo-text">{event.message}</div>
                            <div className="text-[10px] text-echo-text-faint">{formatTime(event.created_at)} · {event.stage} · {event.progress}%</div>
                          </div>
                        </li>
                      ))}
                    </ol>
                  )}
                </div>
              </details>
            </li>
          );
          })}
        </ul>
      )}
    </section>
  );
}

type RecentRow = {
  id: string;
  title: string;
  date: string;
  type: string;
  status: string;
  tone: string;
};

function RecentCard({ className = "", rows, onOpen }: { className?: string; rows: RecentRow[]; onOpen: (meetingId: string) => void }) {
  return (
    <section className={`bg-echo-surface border border-echo-border rounded-lg overflow-hidden ${className}`}>
      <SectionHeader title="Recent meeting notes" subtitle="Last 7 days" action="Open archive" />
      <table className="w-full text-[12px]">
        <thead className="bg-echo-surface-2 text-echo-text-muted">
          <tr>
            <th className="text-left px-5 py-2.5" style={{ fontWeight: 500 }}>Meeting</th>
            <th className="text-left px-3 py-2.5" style={{ fontWeight: 500 }}>Date</th>
            <th className="text-left px-3 py-2.5" style={{ fontWeight: 500 }}>Type</th>
            <th className="text-left px-5 py-2.5" style={{ fontWeight: 500 }}>Status</th>
          </tr>
        </thead>
        <tbody className="divide-y divide-echo-border">
          {rows.length === 0 && (
            <tr>
              <td colSpan={4} className="px-5 py-10 text-center text-[12px] text-echo-text-muted">
                No meeting notes in the database yet.
              </td>
            </tr>
          )}
          {rows.map((n) => (
            <tr key={n.id} onClick={() => onOpen(n.id)} className="hover:bg-echo-surface-hover cursor-pointer">
              <td className="px-5 py-2.5 text-echo-text">{n.title}</td>
              <td className="px-3 py-2.5 text-echo-text-muted">{n.date}</td>
              <td className="px-3 py-2.5"><span className="px-1.5 py-0.5 rounded bg-echo-surface-2 text-echo-text-muted text-[11px]">{n.type}</span></td>
              <td className="px-5 py-2.5">
                <span className="inline-flex items-center gap-1.5 text-[11px] text-echo-text">
                  <span className={`h-1.5 w-1.5 rounded-full ${toneBar[n.tone]}`} />
                  {n.status}
                </span>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </section>
  );
}

function formatDate(value: string) {
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return value;
  return date.toLocaleDateString(undefined, { month: "short", day: "2-digit" });
}

function formatTime(value: string) {
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return value || "unknown";
  return date.toLocaleTimeString(undefined, { hour: "2-digit", minute: "2-digit" });
}

function SectionHeader({ title, subtitle, action }: { title: string; subtitle?: string; action?: string }) {
  return (
    <div className="px-5 py-3 flex items-center justify-between border-b border-echo-border">
      <div className="flex items-baseline gap-2">
        <h3 className="text-[13px] text-echo-text" style={{ fontWeight: 600 }}>{title}</h3>
        {subtitle && <span className="text-[11px] text-echo-text-muted">{subtitle}</span>}
      </div>
      {action && <button className="text-[11.5px] text-echo-accent hover:text-echo-accent-hover">{action} →</button>}
    </div>
  );
}
