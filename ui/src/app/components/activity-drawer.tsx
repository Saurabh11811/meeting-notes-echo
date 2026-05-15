import { useEffect, useMemo, useState } from "react";
import { X, Pause, RefreshCcw, AlertTriangle, CheckCircle2, Clock, FileText, Mic, Wand2, Play } from "lucide-react";
import { getHomeSummary, processAvailableJobs, retryFailedJobs, type HomeSummary } from "../api/echo-api";

const toneBar: Record<string, string> = {
  accent: "bg-echo-accent",
  info: "bg-echo-info",
  muted: "bg-echo-text-faint",
  warning: "bg-echo-warning",
  success: "bg-echo-success",
};

export function ActivityDrawer({ open, onClose }: { open: boolean; onClose: () => void }) {
  const [home, setHome] = useState<HomeSummary | null>(null);
  const [message, setMessage] = useState("");
  const [busy, setBusy] = useState(false);

  const refresh = () => getHomeSummary().then(setHome).catch(() => setHome(null));

  useEffect(() => {
    if (!open) return;
    refresh();
    const interval = window.setInterval(refresh, 5000);
    return () => window.clearInterval(interval);
  }, [open]);

  const jobs = useMemo(() => {
    return (home?.active_jobs || []).map((job) => ({
      id: job.id,
      title: job.title || "Untitled meeting",
      status: job.stage,
      jobStatus: job.status,
      progress: job.progress,
      updated: formatTime(job.updated_at),
      icon: job.status === "failed" ? AlertTriangle : job.stage.toLowerCase().includes("transcrib") ? Mic : job.source_type === "upload" ? FileText : ["queued", "scheduled"].includes(job.status) ? Clock : Wand2,
      tone: job.status === "failed" ? "warning" : job.status === "completed" ? "success" : ["queued", "scheduled"].includes(job.status) ? "muted" : "accent",
      errorMessage: job.error_message || "",
      events: job.events || [],
    }));
  }, [home]);

  const failed = jobs.find((job) => job.tone === "warning");
  const failedCount = jobs.filter((job) => job.tone === "warning").length;
  const queuedCount = jobs.filter((job) => job.jobStatus === "queued").length;

  const handleStartAll = async () => {
    setBusy(true);
    setMessage("");
    try {
      const result = await processAvailableJobs();
      setMessage(result.message);
      await refresh();
    } catch (error) {
      setMessage(error instanceof Error ? error.message : "Could not start the queue.");
    } finally {
      setBusy(false);
    }
  };

  const handleRetryFailed = async () => {
    setBusy(true);
    setMessage("");
    try {
      const result = await retryFailedJobs();
      setMessage(result.message);
      await refresh();
    } catch (error) {
      setMessage(error instanceof Error ? error.message : "Could not retry failed jobs.");
    } finally {
      setBusy(false);
    }
  };

  return (
    <>
      <div onClick={onClose} className={`fixed inset-0 z-40 bg-black/30 transition-opacity ${open ? "opacity-100" : "opacity-0 pointer-events-none"}`} />
      <aside className={`fixed right-0 top-0 bottom-0 z-50 w-[420px] bg-echo-surface border-l border-echo-border shadow-2xl flex flex-col transition-transform ${open ? "translate-x-0" : "translate-x-full"}`}>
        <div className="h-[60px] px-5 flex items-center justify-between border-b border-echo-border">
          <div>
            <div className="text-[17px] text-echo-text" style={{ fontWeight: 700 }}>Activity</div>
            <div className="text-[13px] text-echo-text-faint mt-0.5">{jobs.length} active jobs from local database</div>
          </div>
          <button onClick={onClose} className="h-8 w-8 grid place-items-center rounded-md text-echo-text-muted hover:bg-echo-surface-hover"><X size={15} /></button>
        </div>

        <div className="px-5 py-3 border-b border-echo-border bg-echo-surface-2 flex items-center gap-2">
          <button
            disabled={busy || queuedCount === 0}
            onClick={handleStartAll}
            className="h-9 px-3 rounded-md border border-echo-border bg-echo-surface text-[14px] text-echo-text inline-flex items-center gap-1.5 hover:bg-echo-surface-hover disabled:opacity-50 disabled:cursor-not-allowed"
          >
            <Play size={13} />Start all
          </button>
          <button disabled className="h-9 px-3 rounded-md border border-echo-border bg-echo-surface text-[14px] text-echo-text-muted inline-flex items-center gap-1.5 cursor-not-allowed"><Pause size={13} />Pause</button>
          <button
            disabled={busy || failedCount === 0}
            onClick={handleRetryFailed}
            className="h-9 px-3 rounded-md border border-echo-border bg-echo-surface text-[14px] text-echo-text inline-flex items-center gap-1.5 hover:bg-echo-surface-hover disabled:opacity-50 disabled:cursor-not-allowed"
          >
            <RefreshCcw size={13} />Retry failed
          </button>
          {message && <span className="ml-auto text-[14px] text-echo-text-muted truncate">{message}</span>}
        </div>

        {failed && (
          <div className="bg-amber-500/[0.08] border-b border-echo-border px-5 py-3 flex items-start gap-3">
            <div className="h-7 w-7 rounded-md bg-echo-surface border border-echo-border grid place-items-center text-echo-warning shrink-0"><AlertTriangle size={13} /></div>
            <div className="flex-1 text-[14px]">
              <div className="text-echo-text" style={{ fontWeight: 500 }}>{failed.title}</div>
              <div className="text-echo-text-muted mt-0.5">{failed.errorMessage || failed.status}</div>
            </div>
          </div>
        )}

        <ul className="flex-1 overflow-y-auto divide-y divide-echo-border">
          {jobs.length === 0 && (
            <li className="px-5 py-10 text-center text-[14px] text-echo-text-muted">
              No active jobs. Start a meeting from Home to see live activity here.
            </li>
          )}
          {jobs.map((j) => {
            const Icon = j.icon;
            return (
              <li key={j.id} className="px-5 py-4 hover:bg-echo-surface-hover">
                <div className="flex items-start gap-3">
                  <div className="h-8 w-8 rounded-md bg-echo-surface-2 grid place-items-center text-echo-text-muted shrink-0"><Icon size={15} /></div>
                  <div className="flex-1 min-w-0">
                    <div className="text-[15px] text-echo-text truncate" style={{ fontWeight: 600 }}>{j.title}</div>
                    <div className="text-[13px] text-echo-text-muted mt-0.5">{j.status} - updated {j.updated}</div>
                    <div className="mt-1.5 flex items-center gap-2">
                      <div className="flex-1 h-1 rounded-full bg-echo-surface-2 overflow-hidden">
                        <div className={`h-full ${toneBar[j.tone]}`} style={{ width: `${j.progress}%` }} />
                      </div>
                      <span className="text-[13px] text-echo-text-faint w-7 text-right">{j.progress}%</span>
                    </div>
                    {j.events.length > 0 && (
                      <div className="mt-2 text-[13px] text-echo-text-muted truncate">
                        {j.events[j.events.length - 1].message}
                      </div>
                    )}
                  </div>
                  {j.progress >= 100 && <CheckCircle2 size={13} className="text-echo-success mt-1" />}
                </div>
              </li>
            );
          })}
        </ul>
      </aside>
    </>
  );
}

function formatTime(value: string) {
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return value || "unknown";
  return date.toLocaleTimeString(undefined, { hour: "2-digit", minute: "2-digit" });
}
