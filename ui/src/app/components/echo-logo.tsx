import { motion } from "motion/react";

export function EchoLogoMark({ size = 26 }: { size?: number }) {
  return (
    <svg width={size} height={size} viewBox="0 0 32 32" fill="none" xmlns="http://www.w3.org/2000/svg">
      <motion.circle
        cx="16" cy="16" r="3"
        fill="var(--echo-accent)"
        initial={{ scale: 0.6, opacity: 0.6 }}
        animate={{ scale: [0.7, 1, 0.7], opacity: [0.6, 1, 0.6] }}
        transition={{ duration: 2.4, repeat: Infinity, ease: "easeInOut" }}
      />
      <motion.circle
        cx="16" cy="16" r="7"
        stroke="var(--echo-accent)" strokeWidth="1.5" fill="none" opacity="0.6"
        initial={{ scale: 0.8, opacity: 0 }}
        animate={{ scale: [0.8, 1.05, 0.8], opacity: [0, 0.6, 0] }}
        transition={{ duration: 2.4, repeat: Infinity, ease: "easeInOut", delay: 0.2 }}
      />
      <motion.circle
        cx="16" cy="16" r="12"
        stroke="var(--echo-accent)" strokeWidth="1.25" fill="none" opacity="0.3"
        initial={{ scale: 0.85, opacity: 0 }}
        animate={{ scale: [0.85, 1.05, 0.85], opacity: [0, 0.35, 0] }}
        transition={{ duration: 2.4, repeat: Infinity, ease: "easeInOut", delay: 0.4 }}
      />
    </svg>
  );
}

export function EchoLogoLockup() {
  return (
    <div className="flex items-center gap-2.5">
      <EchoLogoMark size={24} />
      <div className="flex flex-col leading-none">
        <span className="tracking-[0.18em] text-[15px] text-echo-text" style={{ fontWeight: 600 }}>ECHO</span>
        <span className="text-[11px] text-echo-text-faint mt-0.5 tracking-wide">Executive Calls, Highlights & Outcomes</span>
      </div>
    </div>
  );
}
