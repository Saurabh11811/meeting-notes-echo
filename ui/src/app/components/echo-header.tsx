import { useEffect } from "react";
import { Search } from "lucide-react";
import { ThemeToggle } from "./theme-toggle";

export function EchoHeader({
  breadcrumb = "Home",
  onOpenSearch,
}: {
  breadcrumb?: string;
  onOpenSearch: () => void;
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
      <div className="flex items-center gap-2 text-[14px] text-echo-text-muted">
        {parts.map((p, i) => (
          <span key={i} className="flex items-center gap-2">
            {i > 0 && <span className="text-echo-text-faint">/</span>}
            <span className={i === parts.length - 1 ? "text-echo-text" : "text-echo-text-muted"}>{p}</span>
          </span>
        ))}
      </div>

      <div className="flex-1 max-w-xl mx-auto">
        <button
          onClick={onOpenSearch}
          className="w-full h-10 pl-10 pr-16 rounded-md border border-echo-border bg-echo-surface-2 text-[14px] text-echo-text-faint hover:border-echo-border-strong hover:bg-echo-surface-hover relative flex items-center text-left"
        >
          <Search size={16} className="absolute left-3 text-echo-text-faint" />
          <span>Search meetings, decisions, action items, transcripts…</span>
          <kbd className="absolute right-3 text-[12px] text-echo-text-faint border border-echo-border rounded px-1.5 py-0.5 bg-echo-surface">⌘K</kbd>
        </button>
      </div>

      <ThemeToggle />
    </header>
  );
}
