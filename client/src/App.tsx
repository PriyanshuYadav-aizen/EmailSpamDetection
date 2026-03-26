import { useState } from "react";
import api from "./services/api";
import "./index.css";

/* ─── SVG Icons ──────────────────────────────────────────────── */
const ShieldCheckIcon = ({ size = 22 }: { size?: number }) => (
  <svg width={size} height={size} viewBox="0 0 24 24" fill="none" aria-hidden>
    <path d="M12 2L3 6.5V12c0 4.97 3.73 9.63 9 10.93C17.27 21.63 21 16.97 21 12V6.5L12 2Z"
      stroke="currentColor" strokeWidth="1.5" strokeLinejoin="round" />
    <path d="M8.5 12l2.5 2.5 4.5-4.5"
      stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" />
  </svg>
);

const ShieldXIcon = ({ size = 22 }: { size?: number }) => (
  <svg width={size} height={size} viewBox="0 0 24 24" fill="none" aria-hidden>
    <path d="M12 2L3 6.5V12c0 4.97 3.73 9.63 9 10.93C17.27 21.63 21 16.97 21 12V6.5L12 2Z"
      stroke="currentColor" strokeWidth="1.5" strokeLinejoin="round" />
    <path d="M9.5 9.5l5 5M14.5 9.5l-5 5"
      stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" />
  </svg>
);

const ScanIcon = () => (
  <svg width={15} height={15} viewBox="0 0 24 24" fill="none" aria-hidden>
    <path d="M3 7V5a2 2 0 012-2h2" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" />
    <path d="M3 17v2a2 2 0 002 2h2" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" />
    <path d="M17 3h2a2 2 0 012 2v2" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" />
    <path d="M17 21h2a2 2 0 002-2v-2" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" />
    <circle cx="12" cy="12" r="3" stroke="currentColor" strokeWidth="1.5" />
    <path d="M12 9V7M12 17v-2M9 12H7M17 12h-2" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" />
  </svg>
);

const AlertIcon = () => (
  <svg width={14} height={14} viewBox="0 0 24 24" fill="none" aria-hidden>
    <circle cx="12" cy="12" r="9" stroke="currentColor" strokeWidth="1.6" />
    <path d="M12 8v5M12 16.5h.01" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" />
  </svg>
);

const ChipIcon = () => (
  <svg width={10} height={10} viewBox="0 0 24 24" fill="none" aria-hidden>
    <rect x="7" y="7" width="10" height="10" rx="1" stroke="currentColor" strokeWidth="1.8" />
    <path d="M9 7V4M12 7V4M15 7V4M9 20v-3M12 20v-3M15 20v-3M7 9H4M7 12H4M7 15H4M20 9h-3M20 12h-3M20 15h-3"
      stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" />
  </svg>
);

const ActivityIcon = () => (
  <svg width={10} height={10} viewBox="0 0 24 24" fill="none" aria-hidden>
    <polyline points="22 12 18 12 15 21 9 3 6 12 2 12"
      stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" />
  </svg>
);

const LayersIcon = () => (
  <svg width={10} height={10} viewBox="0 0 24 24" fill="none" aria-hidden>
    <path d="M12 2L2 7l10 5 10-5-10-5Z" stroke="currentColor" strokeWidth="1.8" strokeLinejoin="round" />
    <path d="M2 17l10 5 10-5M2 12l10 5 10-5"
      stroke="currentColor" strokeWidth="1.8" strokeLinejoin="round" />
  </svg>
);

/* ─── Helpers ────────────────────────────────────────────────── */
const getRiskLevel = (isSpam: boolean, pct: number): number => {
  if (isSpam) {
    if (pct >= 85) return 5;
    if (pct >= 70) return 4;
    if (pct >= 55) return 3;
    if (pct >= 40) return 2;
    return 1;
  }
  if (pct >= 85) return 1;
  if (pct >= 65) return 2;
  return 3;
};

const RISK_LABELS = ["", "MINIMAL", "LOW", "MODERATE", "HIGH", "CRITICAL"];

/* ─── App ────────────────────────────────────────────────────── */
export default function App() {
  const [content, setContent] = useState("");
  const [prediction, setPrediction] = useState<string | null>(null);
  const [confidence, setConfidence] = useState<number | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [scanTime, setScanTime] = useState<string>("");

  const handleAnalyze = async () => {
    if (!content.trim()) return;
    setError(null);
    setPrediction(null);
    try {
      setLoading(true);
      const res = await api.post("/analyze-email", { content });
      setPrediction(res.data.prediction);
      setConfidence(res.data.confidence);
      setScanTime(new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" }));
    } catch {
      setError("Could not reach the server. Make sure it's running and try again.");
    } finally {
      setLoading(false);
    }
  };

  const isSpam = prediction === "spam";
  const confidencePct = confidence !== null ? Math.round(confidence * 100) : 0;
  const riskLevel = prediction !== null ? getRiskLevel(isSpam, confidencePct) : 0;
  const riskLabel = RISK_LABELS[riskLevel] ?? "";

  /* scan status */
  const scanStatus = loading ? "SCANNING" : prediction !== null ? "COMPLETE" : "READY";
  const statusStyles: Record<string, { bg: string; border: string; text: string; dot: string }> = {
    READY:    { bg: "rgba(59,130,246,0.08)",  border: "rgba(59,130,246,0.2)",  text: "#60a5fa", dot: "#3b82f6" },
    SCANNING: { bg: "rgba(59,130,246,0.12)",  border: "rgba(59,130,246,0.3)",  text: "#93c5fd", dot: "#60a5fa" },
    COMPLETE: {
      bg:     isSpam ? "rgba(220,38,38,0.08)"   : "rgba(20,184,166,0.08)",
      border: isSpam ? "rgba(220,38,38,0.25)"   : "rgba(20,184,166,0.25)",
      text:   isSpam ? "#fca5a5"                : "#5eead4",
      dot:    isSpam ? "#f87171"                : "#2dd4bf",
    },
  };
  const ss = statusStyles[scanStatus];

  /* result card color shortcuts */
  const rc = { label: isSpam ? "var(--spam-label)" : "var(--safe-label)" };

  return (
    <div style={{
      minHeight: "100vh",
      display: "flex",
      alignItems: "center",
      justifyContent: "center",
      padding: "28px 20px",
      /* multi-layer radial background */
      background: `
        radial-gradient(ellipse 70% 50% at 30% -5%,  rgba(37,99,235,0.13)  0%, transparent 55%),
        radial-gradient(ellipse 60% 45% at 80% 110%, rgba(20,184,166,0.08) 0%, transparent 55%),
        var(--bg-root)
      `,
      position: "relative",
    }}>

      {/* ── Gradient card border wrapper ───────────────────── */}
      <div className="gradient-card-border" style={{ width: "100%", maxWidth: 980 }}>
        <div className="card-surface main-grid" style={{
          display: "grid",
          gridTemplateColumns: "clamp(210px, 37%, 330px) 1fr",
        }}>

          {/* Dot grid overlay */}
          <div className="dot-grid" />

          {/* ══ LEFT PANEL ════════════════════════════════════ */}
          <aside className="left-panel" style={{
            padding: "44px 36px",
            borderRight: "1px solid var(--border-subtle)",
            display: "flex",
            flexDirection: "column",
            justifyContent: "space-between",
            background: "linear-gradient(165deg, rgba(37,99,235,0.07) 0%, transparent 65%)",
            position: "relative",
          }}>

            {/* Version badge */}
            <div style={{ display: "flex", flexDirection: "column", gap: 0 }}>
              <div style={{
                display: "inline-flex", alignItems: "center", gap: 7,
                padding: "4px 11px",
                background: "rgba(255,255,255,0.035)",
                border: "1px solid rgba(255,255,255,0.07)",
                borderRadius: 999,
                marginBottom: 28,
                alignSelf: "flex-start",
              }}>
                <div className="live-dot" style={{ background: "#3b82f6" }} />
                <span style={{ fontSize: "0.62rem", fontWeight: 600, letterSpacing: "0.08em", textTransform: "uppercase", color: "var(--text-muted)" }}>
                  v1.0 · Active
                </span>
              </div>

              {/* Shield with pulsing rings */}
              <div style={{ position: "relative", width: 54, height: 54, marginBottom: 22, display: "flex", alignItems: "center", justifyContent: "center" }}>
                <div className="pulse-ring" style={{ width: 54, height: 54 }} />
                <div className="pulse-ring" style={{ width: 54, height: 54 }} />
                <div style={{
                  display: "flex", alignItems: "center", justifyContent: "center",
                  width: 48, height: 48,
                  borderRadius: 14,
                  background: "rgba(59,130,246,0.1)",
                  border: "1px solid rgba(59,130,246,0.22)",
                  color: "var(--accent-bright)",
                  position: "relative",
                  zIndex: 1,
                }}>
                  <ShieldCheckIcon size={24} />
                </div>
              </div>

              {/* Title */}
              <h1 style={{ margin: "0 0 4px", fontSize: "1.55rem", fontWeight: 800, color: "var(--text-primary)", letterSpacing: "-0.03em", lineHeight: 1.1 }}>
                Spam
                <br />
                <span style={{ color: "var(--accent-bright)" }}>Detector</span>
              </h1>
              <div className="glow-divider" />

              <p style={{ margin: "0 0 24px", fontSize: "0.8rem", color: "var(--text-secondary)", lineHeight: 1.75 }}>
                Analyse any email body with a machine-learning model trained on thousands of real messages.
              </p>

              {/* Stat chips — inline horizontal */}
              <div style={{ display: "flex", flexWrap: "wrap", gap: 7 }}>
                <div className="stat-chip"><ChipIcon />ML Powered</div>
                <div className="stat-chip"><LayersIcon />TF-IDF</div>
                <div className="stat-chip"><ActivityIcon />Real-time</div>
              </div>
            </div>

            {/* Bottom disclaimer */}
            <p style={{ margin: "36px 0 0", fontSize: "0.7rem", color: "var(--text-muted)", lineHeight: 1.65 }}>
              Model predictions may not be 100% accurate. Exercise judgement alongside results.
            </p>
          </aside>

          {/* ══ RIGHT PANEL ═══════════════════════════════════ */}
          <main className="right-panel" style={{ padding: "44px 42px", display: "flex", flexDirection: "column" }}>

            {/* Header row */}
            <div style={{ display: "flex", alignItems: "flex-start", justifyContent: "space-between", gap: 12, marginBottom: 24 }}>
              <div>
                <h2 style={{ margin: "0 0 4px", fontSize: "1.05rem", fontWeight: 700, color: "var(--text-primary)", letterSpacing: "-0.015em" }}>
                  Analyse an Email
                </h2>
                <p style={{ margin: 0, fontSize: "0.8rem", color: "var(--text-secondary)" }}>
                  Paste the full email body and click <strong style={{ color: "var(--text-primary)", fontWeight: 600 }}>Analyse Email</strong>.
                </p>
              </div>
              {/* Scan status chip */}
              <div className="scan-chip" style={{ background: ss.bg, borderColor: ss.border, color: ss.text, flexShrink: 0, marginTop: 2 }}>
                <div className="live-dot" style={{ background: ss.dot, animationPlayState: scanStatus === "SCANNING" ? "running" : "paused", opacity: scanStatus === "SCANNING" ? undefined : 1 }} />
                {scanStatus}
              </div>
            </div>

            {/* Label */}
            <label htmlFor="email-body" style={{
              display: "block", fontSize: "0.7rem", fontWeight: 700,
              color: "var(--text-muted)", letterSpacing: "0.1em",
              textTransform: "uppercase", marginBottom: 0,
            }}>
              Email Body
            </label>

            {/* Textarea with chrome bar */}
            <div className="textarea-wrapper" style={{ marginTop: 8 }}>
              <div className="email-chrome-bar">
                <div className="chrome-dot" style={{ background: "rgba(255,255,255,0.1)" }} />
                <div className="chrome-dot" style={{ background: "rgba(255,255,255,0.07)" }} />
                <div className="chrome-dot" style={{ background: "rgba(255,255,255,0.04)" }} />
                <span style={{ fontSize: "0.68rem", color: "var(--text-muted)", marginLeft: 8, letterSpacing: "0.04em" }}>
                  email_input.txt
                </span>
                <span style={{ marginLeft: "auto", fontSize: "0.66rem", color: "var(--text-muted)", fontVariantNumeric: "tabular-nums" }}>
                  {content.length} chars
                </span>
              </div>
              <textarea
                id="email-body"
                className="email-textarea"
                rows={7}
                placeholder="Paste the email content here…"
                value={content}
                onChange={(e) => {
                  setContent(e.target.value);
                  setPrediction(null);
                  setError(null);
                }}
              />
            </div>

            {/* Error banner */}
            {error && (
              <div className="error-banner" style={{ marginTop: 12 }}>
                <AlertIcon />
                {error}
              </div>
            )}

            {/* Analyse button */}
            <button
              id="analyse-btn"
              className="analyze-btn"
              onClick={handleAnalyze}
              disabled={loading || !content.trim()}
              style={{ marginTop: 14 }}
            >
              {loading ? (
                <><span className="spinner" /> Analysing…</>
              ) : (
                <><ScanIcon /> Analyse Email</>
              )}
            </button>

            {/* ── Result Card ───────────────────────────────── */}
            {prediction && (
              <div className={`result-card ${isSpam ? "spam" : "safe"}`} style={{ marginTop: 18 }}>

                {/* Scan complete header */}
                <div style={{
                  display: "flex", justifyContent: "space-between", alignItems: "center",
                  marginBottom: 14, paddingBottom: 13,
                  borderBottom: "1px solid rgba(255,255,255,0.055)",
                }}>
                  <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
                    <div style={{ width: 7, height: 7, borderRadius: "50%", background: rc.label, flexShrink: 0 }} />
                    <span style={{ fontSize: "0.67rem", fontWeight: 700, letterSpacing: "0.1em", textTransform: "uppercase", color: rc.label }}>
                      Scan Complete
                    </span>
                  </div>
                  <span style={{ fontSize: "0.67rem", color: "var(--text-muted)", fontFamily: "monospace", letterSpacing: "0.03em" }}>
                    {scanTime}
                  </span>
                </div>

                {/* Three metrics */}
                <div className="metrics-grid">

                  {/* Verdict */}
                  <div className="metric-cell">
                    <div className="metric-label">Verdict</div>
                    <div className="metric-value" style={{ color: rc.label }}>
                      {isSpam ? "Spam" : "Legit"}
                    </div>
                    <div style={{ marginTop: 6, display: "flex", alignItems: "center", gap: 5 }}>
                      <div style={{ color: rc.label, display: "flex" }}>
                        {isSpam ? <ShieldXIcon size={14} /> : <ShieldCheckIcon size={14} />}
                      </div>
                      <span style={{ fontSize: "0.68rem", color: "var(--text-muted)" }}>
                        {isSpam ? "Flagged" : "Clean"}
                      </span>
                    </div>
                  </div>

                  {/* Confidence */}
                  <div className="metric-cell">
                    <div className="metric-label">Confidence</div>
                    <div className="metric-value" style={{ color: rc.label }}>
                      {confidencePct}
                      <span style={{ fontSize: "0.75rem", fontWeight: 600 }}>%</span>
                    </div>
                    <div style={{ marginTop: 8 }}>
                      <div className="confidence-track">
                        <div className={`confidence-fill ${isSpam ? "spam" : "safe"}`} style={{ width: `${confidencePct}%` }} />
                      </div>
                    </div>
                  </div>

                  {/* Risk */}
                  <div className="metric-cell">
                    <div className="metric-label">Risk Level</div>
                    <div style={{ fontSize: "0.8rem", fontWeight: 700, color: rc.label, letterSpacing: "-0.01em" }}>
                      {riskLabel}
                    </div>
                    <div className="risk-blocks">
                      {[1, 2, 3, 4, 5].map((i) => (
                        <div key={i} className="risk-block" style={{
                          height: 6 + i * 3,
                          background: i <= riskLevel ? rc.label : "rgba(255,255,255,0.06)",
                          animationDelay: `${i * 60}ms`,
                        }} />
                      ))}
                    </div>
                  </div>

                </div>

                {/* Contextual message */}
                <p style={{ margin: "12px 0 0", fontSize: "0.78rem", color: "var(--text-secondary)", lineHeight: 1.65 }}>
                  {isSpam
                    ? "This email exhibits characteristics commonly associated with spam. Avoid clicking any links or sharing personal information."
                    : "The model considers this email legitimate. As always, verify the sender identity if the message seems unexpected."}
                </p>
              </div>
            )}
          </main>

        </div>
      </div>
    </div>
  );
}
