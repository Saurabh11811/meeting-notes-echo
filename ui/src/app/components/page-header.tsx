import { ReactNode } from "react";

export function PageHeader({ title, subtitle, actions }: { title: string; subtitle?: string; actions?: ReactNode }) {
  return (
    <div className="flex items-end justify-between gap-4 pb-1">
      <div>
        <h1 className="text-[22px] text-echo-text" style={{ fontWeight: 600 }}>{title}</h1>
        {subtitle && <p className="text-[13px] text-echo-text-muted mt-1">{subtitle}</p>}
      </div>
      {actions && <div className="flex items-center gap-2">{actions}</div>}
    </div>
  );
}
