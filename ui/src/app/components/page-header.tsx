import { ReactNode } from "react";

export function PageHeader({ title, subtitle, actions }: { title: string; subtitle?: string; actions?: ReactNode }) {
  return (
    <div className="flex items-end justify-between gap-4 pb-2">
      <div>
        <h1 className="text-[26px] text-echo-text" style={{ fontWeight: 700 }}>{title}</h1>
        {subtitle && <p className="text-[15px] text-echo-text-muted mt-1.5">{subtitle}</p>}
      </div>
      {actions && <div className="flex items-center gap-2">{actions}</div>}
    </div>
  );
}
