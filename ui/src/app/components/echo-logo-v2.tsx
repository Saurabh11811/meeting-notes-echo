import { motion } from "motion/react";

/**
 * ECHO mark — concept:
 *  - Left: three nested arcs opening right (the echo / sound emanating).
 *  - Right: three horizontal bars of decreasing length (structured outcomes —
 *    decisions, highlights, action items materialising from the signal).
 *  - A single accent dot at the source — the "speaker" / origin of the call.
 *
 *  Animation loop (~3.4s, restrained):
 *    1. Source dot pulses.
 *    2. Arcs ripple outward in sequence (opacity + scale).
 *    3. Outcome bars draw in left-to-right, one after another.
 *    4. Soft pause, then loop.
 */

const ARC_DASH = 28; // approx length of each arc path for stroke-dash animation
const LOOP = 3.6;

export function EchoLogoMark({ size = 32 }: { size?: number }) {
  return (
    <svg width={size} height={size} viewBox="0 0 40 40" fill="none" xmlns="http://www.w3.org/2000/svg" aria-label="ECHO">
      {/* origin dot */}
      <motion.circle
        cx="9" cy="20" r="2.2"
        fill="var(--echo-accent)"
        initial={{ scale: 0.7, opacity: 0.6 }}
        animate={{ scale: [0.7, 1, 0.7], opacity: [0.6, 1, 0.6] }}
        transition={{ duration: LOOP, repeat: Infinity, ease: "easeInOut" }}
      />

      {/* arc 1 — inner */}
      <motion.path
        d="M 13 14 A 7 7 0 0 1 13 26"
        stroke="var(--echo-accent)"
        strokeWidth="2"
        strokeLinecap="round"
        fill="none"
        initial={{ pathLength: 0, opacity: 0 }}
        animate={{ pathLength: [0, 1, 1, 1], opacity: [0, 1, 1, 0.4] }}
        transition={{ duration: LOOP, repeat: Infinity, times: [0, 0.18, 0.85, 1], ease: "easeOut" }}
      />

      {/* arc 2 — middle */}
      <motion.path
        d="M 11 9 A 12 12 0 0 1 11 31"
        stroke="var(--echo-accent)"
        strokeWidth="1.6"
        strokeLinecap="round"
        fill="none"
        opacity="0.75"
        initial={{ pathLength: 0, opacity: 0 }}
        animate={{ pathLength: [0, 0, 1, 1], opacity: [0, 0, 0.75, 0.3] }}
        transition={{ duration: LOOP, repeat: Infinity, times: [0, 0.1, 0.32, 1], ease: "easeOut" }}
      />

      {/* arc 3 — outer */}
      <motion.path
        d="M 9 4 A 17 17 0 0 1 9 36"
        stroke="var(--echo-accent)"
        strokeWidth="1.3"
        strokeLinecap="round"
        fill="none"
        opacity="0.4"
        initial={{ pathLength: 0, opacity: 0 }}
        animate={{ pathLength: [0, 0, 0, 1, 1], opacity: [0, 0, 0, 0.45, 0.15] }}
        transition={{ duration: LOOP, repeat: Infinity, times: [0, 0.15, 0.25, 0.45, 1], ease: "easeOut" }}
      />

      {/* outcome bars — three structured lines */}
      <motion.line
        x1="22" y1="14" x2="34" y2="14"
        stroke="var(--echo-text)"
        strokeWidth="2"
        strokeLinecap="round"
        initial={{ pathLength: 0, opacity: 0 }}
        animate={{ pathLength: [0, 0, 1, 1], opacity: [0, 0, 1, 1] }}
        transition={{ duration: LOOP, repeat: Infinity, times: [0, 0.45, 0.62, 1], ease: "easeOut" }}
      />
      <motion.line
        x1="22" y1="20" x2="32" y2="20"
        stroke="var(--echo-text)"
        strokeWidth="2"
        strokeLinecap="round"
        opacity="0.75"
        initial={{ pathLength: 0, opacity: 0 }}
        animate={{ pathLength: [0, 0, 0, 1, 1], opacity: [0, 0, 0, 0.85, 0.85] }}
        transition={{ duration: LOOP, repeat: Infinity, times: [0, 0.55, 0.65, 0.78, 1], ease: "easeOut" }}
      />
      <motion.line
        x1="22" y1="26" x2="29" y2="26"
        stroke="var(--echo-text)"
        strokeWidth="2"
        strokeLinecap="round"
        opacity="0.55"
        initial={{ pathLength: 0, opacity: 0 }}
        animate={{ pathLength: [0, 0, 0, 0, 1, 1], opacity: [0, 0, 0, 0, 0.65, 0.65] }}
        transition={{ duration: LOOP, repeat: Infinity, times: [0, 0.6, 0.7, 0.8, 0.92, 1], ease: "easeOut" }}
      />
    </svg>
  );
}

export function EchoLogoLockup() {
  return (
    <div className="flex items-center gap-2.5">
      <EchoLogoMark size={28} />
      <div className="flex flex-col leading-none">
        <span className="tracking-[0.22em] text-[15px] text-echo-text" style={{ fontWeight: 700 }}>ECHO</span>
        <span className="text-[11px] text-echo-text-faint mt-0.5 tracking-[0.08em] uppercase">Calls · Highlights · Outcomes</span>
      </div>
    </div>
  );
}

/** Larger lockup for loading screens / splash. */
export function EchoLogoHero() {
  return (
    <div className="flex items-center gap-4">
      <EchoLogoMark size={56} />
      <div className="flex flex-col leading-none">
        <span className="tracking-[0.24em] text-[28px] text-echo-text" style={{ fontWeight: 700 }}>ECHO</span>
        <span className="text-[13px] text-echo-text-muted mt-1 tracking-[0.12em] uppercase">Executive Calls, Highlights & Outcomes</span>
      </div>
    </div>
  );
}
