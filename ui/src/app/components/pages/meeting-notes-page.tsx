import { useEffect, useMemo, useState } from "react";
import { PageHeader } from "../page-header";
import { Search, Download, Archive, ChevronDown, ChevronLeft, ChevronRight, ExternalLink, RefreshCw, X } from "lucide-react";
import { getMeetings, type MeetingSummary } from "../../api/echo-api";

const TYPES = ["All types", "Executive", "Project Review", "Client Call", "Townhall", "Demo/UAT", "Incident"];
const DATES = ["All time", "Last 7 days", "Last 30 days", "This quarter"];
const PAGE_SIZE = 10;

export function MeetingNotesPage({ onOpen }: { onOpen: (meetingId: string) => void }) {
  const [rows, setRows] = useState<MeetingSummary[]>([]);
  const [q, setQ] = useState("");
  const [type, setType] = useState("All types");
  const [date, setDate] = useState("All time");
  const [page, setPage] = useState(1);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");

  useEffect(() => {
    setLoading(true);
    getMeetings()
      .then((data) => {
        setRows(data.meetings || []);
        setError("");
      })
      .catch((err) => setError(err instanceof Error ? err.message : "Could not load meetings."))
      .finally(() => setLoading(false));
  }, []);

  const filtered = useMemo(() => {
    return rows.filter((r) => {
      if (type !== "All types" && r.meeting_type !== type) return false;
      if (!matchesDateFilter(r.updated_at || r.created_at, date)) return false;
      const term = q.trim().toLowerCase();
      if (!term) return true;
      return [r.title, r.meeting_type, r.project, r.host, r.source_label, r.status]
        .some((value) => (value || "").toLowerCase().includes(term));
    });
  }, [rows, q, type, date]);

  const totalPages = Math.max(1, Math.ceil(filtered.length / PAGE_SIZE));
  const safePage = Math.min(page, totalPages);
  const slice = filtered.slice((safePage - 1) * PAGE_SIZE, safePage * PAGE_SIZE);
  const start = (safePage - 1) * PAGE_SIZE + 1;
  const end = (safePage - 1) * PAGE_SIZE + slice.length;

  return (
    <div className="space-y-5">
      <PageHeader
        title="Meeting Notes"
        subtitle="Every generated MoM, searchable and filterable."
        actions={
          <>
            <button disabled className="h-9 px-3 rounded-md border border-echo-border bg-echo-surface text-[12px] text-echo-text-muted inline-flex items-center gap-1.5 cursor-not-allowed"><Download size={13} />Export</button>
            <button disabled className="h-9 px-3 rounded-md border border-echo-border bg-echo-surface text-[12px] text-echo-text-muted inline-flex items-center gap-1.5 cursor-not-allowed"><Archive size={13} />Archive</button>
          </>
        }
      />

      <section className="bg-echo-surface border border-echo-border rounded-lg overflow-hidden">
        <div className="px-3 py-2.5 border-b border-echo-border flex items-center gap-2 flex-wrap">
          <div className="flex items-center gap-2 flex-1 min-w-[260px] relative">
            <Search size={14} className="absolute left-3 text-echo-text-faint" />
            <input
              value={q}
              onChange={(e) => { setQ(e.target.value); setPage(1); }}
              placeholder="Search by meeting title, source, type, or status..."
              className="w-full h-9 pl-9 pr-8 rounded-md border border-echo-border bg-echo-surface-2 text-[13px] text-echo-text focus:outline-none focus:bg-echo-surface focus:border-echo-accent placeholder:text-echo-text-faint"
            />
            {q && (
              <button onClick={() => setQ("")} className="absolute right-2 text-echo-text-faint hover:text-echo-text"><X size={13} /></button>
            )}
          </div>

          <FilterDropdown label="Type" value={type} options={TYPES} onChange={(v) => { setType(v); setPage(1); }} />
          <FilterDropdown label="Date" value={date} options={DATES} onChange={(v) => { setDate(v); setPage(1); }} />

          <span className="text-[11px] text-echo-text-faint ml-auto">{filtered.length} results</span>
        </div>

        <table className="w-full text-[12px]">
          <thead className="bg-echo-surface-2 text-echo-text-muted">
            <tr>
              <th className="text-left px-5 py-2.5" style={{ fontWeight: 500 }}>Meeting</th>
              <th className="text-left px-3 py-2.5" style={{ fontWeight: 500 }}>Date</th>
              <th className="text-left px-3 py-2.5" style={{ fontWeight: 500 }}>Type</th>
              <th className="text-left px-3 py-2.5" style={{ fontWeight: 500 }}>Source</th>
              <th className="text-left px-3 py-2.5" style={{ fontWeight: 500 }}>Decisions</th>
              <th className="text-left px-3 py-2.5" style={{ fontWeight: 500 }}>Actions</th>
              <th className="text-left px-3 py-2.5" style={{ fontWeight: 500 }}>Version</th>
              <th className="text-right px-5 py-2.5" style={{ fontWeight: 500 }}></th>
            </tr>
          </thead>
          <tbody className="divide-y divide-echo-border">
            {(loading || error || slice.length === 0) && (
              <tr>
                <td colSpan={8} className="px-5 py-12 text-center text-[12px] text-echo-text-muted">
                  {loading ? "Loading meetings..." : error || "No meeting notes in the database yet."}
                </td>
              </tr>
            )}
            {slice.map((r) => (
              <tr key={r.id} onClick={() => onOpen(r.id)} className="hover:bg-echo-surface-hover cursor-pointer">
                <td className="px-5 py-3 text-echo-text">{r.title}</td>
                <td className="px-3 py-3 text-echo-text-muted">{formatDate(r.updated_at || r.created_at)}</td>
                <td className="px-3 py-3"><span className="px-1.5 py-0.5 rounded bg-echo-surface-2 text-echo-text-muted text-[11px]">{r.meeting_type}</span></td>
                <td className="px-3 py-3 text-echo-text-muted max-w-[220px] truncate">{r.source_label || r.source_type}</td>
                <td className="px-3 py-3 text-echo-text">{r.decisions_count}</td>
                <td className="px-3 py-3 text-echo-text">{r.action_items_count}</td>
                <td className="px-3 py-3 text-echo-text-muted">v{r.mom_version || 0}</td>
                <td className="px-5 py-3" onClick={(e) => e.stopPropagation()}>
                  <div className="flex items-center justify-end gap-1 text-echo-text-faint">
                    <button onClick={() => onOpen(r.id)} title="Open" className="h-7 w-7 grid place-items-center rounded hover:bg-echo-surface-hover hover:text-echo-text"><ExternalLink size={13} /></button>
                    <button disabled title="Export" className="h-7 w-7 grid place-items-center rounded opacity-40 cursor-not-allowed"><Download size={13} /></button>
                    <button disabled title="Regenerate" className="h-7 w-7 grid place-items-center rounded opacity-40 cursor-not-allowed"><RefreshCw size={13} /></button>
                  </div>
                </td>
              </tr>
            ))}
          </tbody>
        </table>

        <div className="px-5 py-2.5 border-t border-echo-border flex items-center justify-between text-[11.5px] text-echo-text-muted">
          <div>{filtered.length === 0 ? "No results" : `Showing ${start}-${end} of ${filtered.length}`}</div>
          <div className="flex items-center gap-1">
            <button onClick={() => setPage(Math.max(1, safePage - 1))} disabled={safePage === 1} className="h-7 w-7 grid place-items-center rounded border border-echo-border bg-echo-surface hover:bg-echo-surface-hover disabled:opacity-40 disabled:cursor-not-allowed"><ChevronLeft size={13} /></button>
            <span className="px-2">Page {safePage} of {totalPages}</span>
            <button onClick={() => setPage(Math.min(totalPages, safePage + 1))} disabled={safePage === totalPages} className="h-7 w-7 grid place-items-center rounded border border-echo-border bg-echo-surface hover:bg-echo-surface-hover disabled:opacity-40 disabled:cursor-not-allowed"><ChevronRight size={13} /></button>
          </div>
        </div>
      </section>
    </div>
  );
}

function FilterDropdown({ label, value, options, onChange }: { label: string; value: string; options: string[]; onChange: (v: string) => void }) {
  const [open, setOpen] = useState(false);
  return (
    <div className="relative">
      <button onClick={() => setOpen(!open)} className="h-9 px-2.5 rounded-md border border-echo-border bg-echo-surface text-[12px] text-echo-text inline-flex items-center gap-1.5 hover:bg-echo-surface-hover">
        <span className="text-echo-text-muted">{label}:</span>
        <span>{value}</span>
        <ChevronDown size={11} className="text-echo-text-faint" />
      </button>
      {open && (
        <>
          <div className="fixed inset-0 z-10" onClick={() => setOpen(false)} />
          <ul className="absolute right-0 mt-1 z-20 min-w-[180px] bg-echo-surface border border-echo-border rounded-md shadow-lg py-1">
            {options.map((o) => (
              <li key={o}>
                <button onClick={() => { onChange(o); setOpen(false); }} className={`w-full text-left px-3 py-1.5 text-[12px] ${o === value ? "bg-echo-accent-bg text-echo-accent-fg" : "text-echo-text hover:bg-echo-surface-hover"}`}>
                  {o}
                </button>
              </li>
            ))}
          </ul>
        </>
      )}
    </div>
  );
}

function matchesDateFilter(value: string, filter: string) {
  if (filter === "All time") return true;
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return true;
  const now = new Date();
  const diffDays = (now.getTime() - date.getTime()) / 86400000;
  if (filter === "Last 7 days") return diffDays <= 7;
  if (filter === "Last 30 days") return diffDays <= 30;
  if (filter === "This quarter") {
    return date.getFullYear() === now.getFullYear() && Math.floor(date.getMonth() / 3) === Math.floor(now.getMonth() / 3);
  }
  return true;
}

function formatDate(value: string) {
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return value;
  return date.toLocaleDateString(undefined, { month: "short", day: "2-digit" });
}
