import { Home, NotebookText, LayoutTemplate, Settings, HelpCircle } from "lucide-react";
import { EchoLogoLockup } from "./echo-logo-v2";
import type { PageId } from "../App";

const navItems: { id: PageId; label: string; icon: any }[] = [
  { id: "home", label: "Home", icon: Home },
  { id: "notes", label: "Meeting Notes", icon: NotebookText },
  { id: "templates", label: "Templates", icon: LayoutTemplate },
  { id: "settings", label: "Settings", icon: Settings },
];

export function EchoSidebar({ current, onNavigate }: { current: PageId; onNavigate: (id: PageId) => void }) {
  return (
    <aside className="w-[232px] shrink-0 border-r border-echo-border bg-echo-surface flex flex-col">
      <div className="h-[60px] px-5 flex items-center border-b border-echo-border">
        <EchoLogoLockup />
      </div>

      <nav className="flex-1 px-3 py-4 space-y-0.5 overflow-y-auto">
        {navItems.map((item) => {
          const Icon = item.icon;
          const active = current === item.id || (item.id === "notes" && current === "detail");
          return (
            <button
              key={item.id}
              onClick={() => onNavigate(item.id)}
              className={`w-full flex items-center gap-3 px-3 py-2 rounded-md text-[13px] transition-colors ${
                active ? "bg-echo-accent-bg text-echo-accent-fg" : "text-echo-text-muted hover:bg-echo-surface-hover hover:text-echo-text"
              }`}
            >
              <Icon size={15} className={active ? "text-echo-accent" : ""} />
              <span className="flex-1 text-left">{item.label}</span>
            </button>
          );
        })}
      </nav>

      <div className="px-3 py-3 border-t border-echo-border">
        <button className="w-full flex items-center gap-3 px-3 py-2 rounded-md text-[12px] text-echo-text-muted hover:bg-echo-surface-hover">
          <HelpCircle size={14} />
          <span>Help & Shortcuts</span>
          <kbd className="ml-auto text-[10px] text-echo-text-faint">?</kbd>
        </button>
        <div className="mt-2 px-3 py-2.5 rounded-md bg-echo-surface-2 border border-echo-border">
          <div className="text-[10px] text-echo-text-faint mb-1 uppercase tracking-wider">Workspace</div>
          <div className="text-[12px] text-echo-text">Executive Office</div>
          <div className="text-[10px] text-echo-text-faint mt-1 flex items-center gap-1.5">
            <span className="h-1.5 w-1.5 rounded-full bg-echo-success" />
            v2.4.1 · all systems healthy
          </div>
        </div>
      </div>
    </aside>
  );
}
