import { Home, NotebookText, LayoutTemplate, Settings, Info } from "lucide-react";
import { EchoLogoLockup } from "./echo-logo-v2";
import type { PageId } from "../App";

const navItems: { id: PageId; label: string; icon: any }[] = [
  { id: "home", label: "Home", icon: Home },
  { id: "notes", label: "Meeting Notes", icon: NotebookText },
  { id: "templates", label: "Template Studio", icon: LayoutTemplate },
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
              className={`w-full flex items-center gap-3 px-3 py-2.5 rounded-md text-[14px] transition-colors ${
                active ? "bg-echo-accent-bg text-echo-accent-fg" : "text-echo-text-muted hover:bg-echo-surface-hover hover:text-echo-text"
              }`}
            >
              <Icon size={16} className={active ? "text-echo-accent" : ""} />
              <span className="flex-1 text-left">{item.label}</span>
            </button>
          );
        })}
      </nav>

      <div className="px-3 py-4 border-t border-echo-border">
        <div className="px-3">
          <div className="flex items-center gap-2 text-[15px] text-echo-text mb-2">
            <Info size={16} className="text-echo-accent" />
            <span style={{ fontWeight: 600 }}>About ECHO</span>
          </div>
          <p className="text-[14px] text-echo-text-muted leading-relaxed">
            ECHO is an AI-powered assistant that converts your meeting recordings and transcripts into professional, board-ready minutes.
          </p>
        </div>
      </div>
    </aside>
  );
}
