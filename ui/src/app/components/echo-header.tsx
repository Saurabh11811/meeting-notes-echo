import { useEffect } from "react";
import { Search, Activity, ChevronDown } from "lucide-react";
import { ThemeToggle } from "./theme-toggle";

export function EchoHeader({
  breadcrumb = "Home",
  onOpenSearch,
  onOpenActivity,
  attentionCount = 0,
}: {
  breadcrumb?: string;
  onOpenSearch: () => void;
  onOpenActivity: () => void;
  attentionCount?: number;
}) {
  const parts = breadcrumb.split(" / ");

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === "k") {
        e.preventDefault();
        onOpenSearch();
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onOpenSearch]);

  return (
    <header className="h-[60px] shrink-0 px-6 border-b border-echo-border bg-echo-surface flex items-center gap-4">
      <div className="flex items-center gap-2 text-[12px] text-echo-text-muted">
        <span>Workspace</span>
        {parts.map((p, i) => (
          <span key={i} className="flex items-center gap-2">
            <span className="text-echo-text-faint">/</span>
            <span className={i === parts.length - 1 ? "text-echo-text" : "text-echo-text-muted"}>{p}</span>
          </span>
        ))}
      </div>

      <div className="flex-1 max-w-xl mx-auto">
        <button
          onClick={onOpenSearch}
          className="w-full h-9 pl-9 pr-16 rounded-md border border-echo-border bg-echo-surface-2 text-[12.5px] text-echo-text-faint hover:border-echo-border-strong hover:bg-echo-surface-hover relative flex items-center text-left"
        >
          <Search size={14} className="absolute left-3 text-echo-text-faint" />
          <span>Search meetings, decisions, action items, transcripts…</span>
          <kbd className="absolute right-3 text-[10px] text-echo-text-faint border border-echo-border rounded px-1.5 py-0.5 bg-echo-surface">⌘K</kbd>
        </button>
      </div>

      <button
        onClick={onOpenActivity}
        title="Activity"
        className="relative h-9 px-2.5 grid place-items-center rounded-md hover:bg-echo-surface-hover text-echo-text-muted inline-flex items-center gap-1.5 text-[12px]"
      >
        <Activity size={15} />
        <span>5</span>
        {attentionCount > 0 && (
          <span className="absolute top-1 right-1 h-1.5 w-1.5 rounded-full bg-echo-warning" />
        )}
      </button>

      <ThemeToggle />

      <button className="flex items-center gap-2 pl-1 pr-2 py-1 rounded-md hover:bg-echo-surface-hover">
        <div className="h-7 w-7 rounded-full bg-echo-accent grid place-items-center text-white text-[11px]" style={{ background: "linear-gradient(135deg, var(--echo-accent), var(--echo-accent-hover))" }}>PS</div>
        <div className="text-left leading-tight">
          <div className="text-[12px] text-echo-text">Priya Sharma</div>
          <div className="text-[10px] text-echo-text-muted">Chief of Staff</div>
        </div>
        <ChevronDown size={13} className="text-echo-text-faint" />
      </button>
    </header>
  );
}
