import { useState } from "react";
import { PageHeader } from "../page-header";
import { Briefcase, ClipboardCheck, Phone, Megaphone, MonitorPlay, ShieldAlert, Lock, Star, Copy, Pencil, FlaskConical, Plus } from "lucide-react";

const templates = [
  { name: "Executive MoM", icon: Briefcase, desc: "Board-ready summary with decisions, risks, next steps.", updated: "Apr 28", isDefault: true, locked: true },
  { name: "Project Review", icon: ClipboardCheck, desc: "Status update with milestones, blockers, and owners.", updated: "Apr 22", locked: false },
  { name: "Client Call", icon: Phone, desc: "External-facing recap with commitments and follow-ups.", updated: "Apr 18", locked: false },
  { name: "Townhall", icon: Megaphone, desc: "Wide-audience summary, key announcements, Q&A.", updated: "Mar 30", locked: false },
  { name: "Demo / UAT", icon: MonitorPlay, desc: "Acceptance criteria coverage, defects, sign-off readiness.", updated: "Apr 02", locked: false },
  { name: "Incident Review", icon: ShieldAlert, desc: "Timeline, root cause, mitigations, follow-up actions.", updated: "Apr 12", locked: true },
];

export function TemplatesPage() {
  const [active, setActive] = useState(templates[0].name);
  const current = templates.find((t) => t.name === active)!;
  const Icon = current.icon;

  return (
    <div className="space-y-5">
      <PageHeader
        title="Templates"
        subtitle="Pick the structure that fits each meeting."
        actions={<button className="h-9 px-3 rounded-md bg-echo-text text-echo-surface text-[12px] inline-flex items-center gap-1.5"><Plus size={13} />New template</button>}
      />

      <div className="grid grid-cols-1 xl:grid-cols-[1fr_400px] gap-5">
        <div className="grid sm:grid-cols-2 gap-3">
          {templates.map((t) => {
            const TIcon = t.icon;
            const isActive = active === t.name;
            return (
              <button key={t.name} onClick={() => setActive(t.name)} className={`text-left bg-echo-surface border rounded-lg p-4 transition-colors ${isActive ? "border-echo-accent ring-2 ring-echo-accent/20" : "border-echo-border hover:border-echo-border-strong"}`}>
                <div className="flex items-start gap-3">
                  <div className="h-9 w-9 rounded-md bg-echo-accent-bg text-echo-accent grid place-items-center"><TIcon size={16} /></div>
                  <div className="flex-1">
                    <div className="flex items-center gap-2">
                      <div className="text-[13px] text-echo-text" style={{ fontWeight: 600 }}>{t.name}</div>
                      {t.isDefault && <span className="text-[10px] px-1.5 py-0.5 rounded bg-echo-accent-bg text-echo-accent-fg">Default</span>}
                      {t.locked && <Lock size={11} className="text-echo-text-faint" />}
                    </div>
                    <p className="text-[12px] text-echo-text-muted mt-1">{t.desc}</p>
                    <div className="text-[11px] text-echo-text-faint mt-2">Updated {t.updated}</div>
                  </div>
                </div>
              </button>
            );
          })}
        </div>

        <aside className="bg-echo-surface border border-echo-border rounded-lg overflow-hidden h-fit">
          <div className="px-5 py-4 border-b border-echo-border flex items-start gap-3">
            <div className="h-10 w-10 rounded-md bg-echo-accent-bg text-echo-accent grid place-items-center"><Icon size={18} /></div>
            <div className="flex-1">
              <h2 className="text-[15px] text-echo-text" style={{ fontWeight: 600 }}>{current.name}</h2>
              <p className="text-[12px] text-echo-text-muted">{current.desc}</p>
            </div>
          </div>

          <div className="p-5 space-y-4 text-[12px]">
            <div>
              <h3 className="text-[11px] text-echo-text-muted uppercase tracking-wide mb-2">Sections included</h3>
              <div className="flex flex-wrap gap-1.5">
                {["Executive Summary", "Decisions", "Action Items", "Risks", "Discussion Points", "Next Steps"].map((s) => (
                  <span key={s} className="px-2 py-0.5 rounded bg-echo-surface-2 text-echo-text-muted text-[11px]">{s}</span>
                ))}
              </div>
            </div>

            <div>
              <h3 className="text-[11px] text-echo-text-muted uppercase tracking-wide mb-2">Output preview</h3>
              <div className="border border-echo-border rounded-md p-3 bg-echo-surface-2 text-echo-text leading-relaxed">
                <div className="text-echo-text" style={{ fontWeight: 600 }}>Summary</div>
                <p className="mt-1 text-echo-text-muted">Brief, board-ready synthesis of the meeting outcomes and strategic implications.</p>
                <div className="text-echo-text mt-3" style={{ fontWeight: 600 }}>Decisions</div>
                <p className="mt-1 text-echo-text-muted">Numbered decisions with context, owner, and timestamp reference.</p>
                <div className="text-echo-text mt-3" style={{ fontWeight: 600 }}>Action items</div>
                <p className="mt-1 text-echo-text-muted">Owner · Task · Due date · Status · Source reference.</p>
              </div>
            </div>

            <div className="grid grid-cols-2 gap-2">
              <button className="h-9 rounded-md bg-echo-accent text-white text-[12px] inline-flex items-center justify-center gap-1.5"><Star size={12} />Use as default</button>
              <button className="h-9 rounded-md border border-echo-border bg-echo-surface text-echo-text text-[12px] inline-flex items-center justify-center gap-1.5"><Copy size={12} />Duplicate</button>
              <button className="h-9 rounded-md border border-echo-border bg-echo-surface text-echo-text text-[12px] inline-flex items-center justify-center gap-1.5"><Pencil size={12} />Edit</button>
              <button className="h-9 rounded-md border border-echo-border bg-echo-surface text-echo-text text-[12px] inline-flex items-center justify-center gap-1.5"><FlaskConical size={12} />Test with transcript</button>
            </div>

            <div className="text-[11px] text-echo-text-muted pt-2 border-t border-echo-border">
              Prompt and model details are managed in <span className="text-echo-accent hover:underline cursor-pointer">Prompt Studio</span> (admin).
            </div>
          </div>
        </aside>
      </div>
    </div>
  );
}
