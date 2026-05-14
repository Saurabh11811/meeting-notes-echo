import { useEffect, useMemo, useRef, useState } from "react";
import { Search, FileText, Sparkles, Quote, X, CornerDownLeft } from "lucide-react";
import { getMeetings } from "../api/echo-api";

type Result = {
  id: string;
  meetingId: string;
  kind: "meeting" | "decision" | "action" | "transcript";
  title: string;
  context: string;
  meeting: string;
  date: string;
  timestamp?: string;
};

const kindMeta: Record<Result["kind"], { label: string; icon: any; tint: string }> = {
  meeting: { label: "Meeting", icon: FileText, tint: "text-echo-info" },
  decision: { label: "Decision", icon: Sparkles, tint: "text-echo-accent" },
  action: { label: "Action item", icon: CornerDownLeft, tint: "text-echo-warning" },
  transcript: { label: "Transcript", icon: Quote, tint: "text-echo-text-muted" },
};

export function SearchPalette({ open, onClose, onOpenMeeting }: { open: boolean; onClose: () => void; onOpenMeeting: (meetingId?: string) => void }) {
  const [q, setQ] = useState("");
  const [filter, setFilter] = useState<"all" | Result["kind"]>("all");
  const [active, setActive] = useState(0);
  const [corpus, setCorpus] = useState<Result[]>([]);
  const inputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    if (open) {
      setQ("");
      setFilter("all");
      setActive(0);
      requestAnimationFrame(() => inputRef.current?.focus());
      getMeetings()
        .then((data) => {
          setCorpus((data.meetings || []).map((meeting) => ({
            id: `meeting-${meeting.id}`,
            meetingId: meeting.id,
            kind: "meeting",
            title: meeting.title,
            context: `${meeting.meeting_type} · ${meeting.status} · ${meeting.decisions_count} decisions · ${meeting.action_items_count} actions`,
            meeting: meeting.title,
            date: formatDate(meeting.updated_at || meeting.created_at),
          })));
        })
        .catch(() => setCorpus([]));
    }
  }, [open]);

  const results = useMemo(() => {
    const term = q.trim().toLowerCase();
    return corpus.filter((r) => {
      if (filter !== "all" && r.kind !== filter) return false;
      if (!term) return true;
      return r.title.toLowerCase().includes(term) || r.context.toLowerCase().includes(term) || r.meeting.toLowerCase().includes(term);
    });
  }, [q, filter]);

  useEffect(() => {
    if (!open) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
      else if (e.key === "ArrowDown") { e.preventDefault(); setActive((i) => Math.min(i + 1, results.length - 1)); }
      else if (e.key === "ArrowUp") { e.preventDefault(); setActive((i) => Math.max(i - 1, 0)); }
      else if (e.key === "Enter" && results[active]) { onOpenMeeting(results[active].meetingId); onClose(); }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [open, results, active, onClose, onOpenMeeting]);

  if (!open) return null;

  const filters: { id: typeof filter; label: string }[] = [
    { id: "all", label: "All" },
    { id: "meeting", label: "Meetings" },
    { id: "decision", label: "Decisions" },
    { id: "action", label: "Action items" },
    { id: "transcript", label: "Transcripts" },
  ];

  return (
    <div className="fixed inset-0 z-50 flex items-start justify-center pt-[10vh] px-4 bg-black/40 backdrop-blur-sm" onClick={onClose}>
      <div onClick={(e) => e.stopPropagation()} className="w-full max-w-2xl bg-echo-surface border border-echo-border rounded-xl shadow-2xl overflow-hidden">
        <div className="flex items-center gap-2 px-4 py-3 border-b border-echo-border">
          <Search size={16} className="text-echo-text-faint" />
          <input
            ref={inputRef}
            value={q}
            onChange={(e) => { setQ(e.target.value); setActive(0); }}
            placeholder="Search meetings, decisions, action items, or transcripts…"
            className="flex-1 h-8 bg-transparent text-[14px] text-echo-text focus:outline-none placeholder:text-echo-text-faint"
          />
          <button onClick={onClose} className="h-7 w-7 grid place-items-center rounded text-echo-text-faint hover:bg-echo-surface-hover"><X size={14} /></button>
        </div>

        <div className="flex items-center gap-1 px-3 py-2 border-b border-echo-border bg-echo-surface-2">
          {filters.map((f) => (
            <button
              key={f.id}
              onClick={() => { setFilter(f.id); setActive(0); }}
              className={`px-2.5 py-1 rounded text-[11px] ${filter === f.id ? "bg-echo-accent-bg text-echo-accent-fg" : "text-echo-text-muted hover:bg-echo-surface-hover"}`}
            >{f.label}</button>
          ))}
          <span className="ml-auto text-[11px] text-echo-text-faint">{results.length} results</span>
        </div>

        <ul className="max-h-[420px] overflow-y-auto py-1">
          {results.length === 0 && (
            <li className="px-6 py-10 text-center text-[12px] text-echo-text-muted">No saved meetings found.</li>
          )}
          {results.map((r, i) => {
            const meta = kindMeta[r.kind];
            const Icon = meta.icon;
            return (
              <li key={r.id}>
                <button
                  onMouseEnter={() => setActive(i)}
                  onClick={() => { onOpenMeeting(r.meetingId); onClose(); }}
                  className={`w-full text-left px-4 py-2.5 flex items-start gap-3 ${i === active ? "bg-echo-accent-bg" : "hover:bg-echo-surface-hover"}`}
                >
                  <div className={`h-7 w-7 rounded-md grid place-items-center bg-echo-surface-2 ${meta.tint} shrink-0 mt-0.5`}><Icon size={13} /></div>
                  <div className="flex-1 min-w-0">
                    <div className="flex items-center gap-2">
                      <span className="text-[10px] uppercase tracking-wide text-echo-text-faint">{meta.label}</span>
                      <span className="text-[11px] text-echo-text-faint">·</span>
                      <span className="text-[11px] text-echo-text-muted truncate">{r.meeting} · {r.date}{r.timestamp ? ` · ${r.timestamp}` : ""}</span>
                    </div>
                    <div className="text-[13px] text-echo-text truncate mt-0.5">{r.title}</div>
                    <div className="text-[11px] text-echo-text-muted truncate">{r.context}</div>
                  </div>
                </button>
              </li>
            );
          })}
        </ul>

        <div className="px-4 py-2 border-t border-echo-border bg-echo-surface-2 flex items-center gap-4 text-[10px] text-echo-text-faint">
          <span><kbd className="px-1 border border-echo-border rounded">↑↓</kbd> navigate</span>
          <span><kbd className="px-1 border border-echo-border rounded">↵</kbd> open</span>
          <span><kbd className="px-1 border border-echo-border rounded">esc</kbd> close</span>
          <span className="ml-auto">Searching saved meetings from the local database</span>
        </div>
      </div>
    </div>
  );
}

function formatDate(value: string) {
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return value;
  return date.toLocaleDateString(undefined, { month: "short", day: "2-digit" });
}
