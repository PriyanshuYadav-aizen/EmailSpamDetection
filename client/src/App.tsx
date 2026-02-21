import { useState } from "react";
import api from "./services/api";

function App() {
  const [content, setContent] = useState("");
  const [prediction, setPrediction] = useState<string | null>(null);
  const [confidence, setConfidence] = useState<number | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleAnalyze = async () => {
    if (!content.trim()) return;

    setError(null);
    try {
      setLoading(true);

      const res = await api.post("/analyze-email", { content });

      setPrediction(res.data.prediction);
      setConfidence(res.data.confidence);
    } catch {
      setError("Could not analyze email. Check the server and try again.");
    } finally {
      setLoading(false);
    }
  };

  const isSpam = prediction === "spam";
  const confidencePct = confidence !== null ? Math.round(confidence * 100) : 0;

  return (
    <div className="min-h-screen flex items-center justify-center p-6 bg-gradient-to-br from-slate-50 via-slate-100 to-indigo-50">
      <div className="w-full max-w-lg">
        <div className="bg-white/90 backdrop-blur-sm shadow-xl shadow-slate-200/50 rounded-2xl border border-slate-200/80 overflow-hidden">
          <div className="px-8 pt-8 pb-6">
            <div className="text-center mb-6">
              <h1 className="text-2xl font-bold text-slate-800 tracking-tight">
                Email Spam Detection
              </h1>
              <p className="text-slate-500 text-sm mt-1">
                Paste the email body below to check if it’s spam
              </p>
            </div>

            <label className="block text-sm font-medium text-slate-700 mb-2">
              Email content
            </label>
            <textarea
              className="w-full border border-slate-200 rounded-xl p-3.5 text-slate-800 placeholder-slate-400 focus:outline-none focus:ring-2 focus:ring-indigo-500/50 focus:border-indigo-400 resize-y min-h-[140px] transition shadow-sm"
              rows={6}
              placeholder="Paste email content here..."
              value={content}
              onChange={(e) => {
                setContent(e.target.value);
                setPrediction(null);
                setError(null);
              }}
            />

            {error && (
              <div className="mt-3 px-3 py-2 rounded-lg bg-red-50 border border-red-100 text-red-700 text-sm">
                {error}
              </div>
            )}

            <button
              onClick={handleAnalyze}
              disabled={loading}
              className="w-full mt-4 bg-indigo-600 text-white font-medium py-3 rounded-xl hover:bg-indigo-700 focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:ring-offset-2 disabled:opacity-60 disabled:cursor-not-allowed transition shadow-sm"
            >
              {loading ? (
                <span className="inline-flex items-center gap-2">
                  <svg
                    className="animate-spin h-4 w-4"
                    xmlns="http://www.w3.org/2000/svg"
                    fill="none"
                    viewBox="0 0 24 24"
                    aria-hidden
                  >
                    <circle
                      className="opacity-25"
                      cx="12"
                      cy="12"
                      r="10"
                      stroke="currentColor"
                      strokeWidth="4"
                    />
                    <path
                      className="opacity-75"
                      fill="currentColor"
                      d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"
                    />
                  </svg>
                  Analyzing…
                </span>
              ) : (
                "Analyze email"
              )}
            </button>
          </div>

          {prediction && (
            <div className="px-8 pb-8 pt-2">
              <div
                className={`rounded-xl border-2 p-4 ${
                  isSpam
                    ? "bg-amber-50 border-amber-200"
                    : "bg-emerald-50 border-emerald-200"
                }`}
              >
                <div className="flex items-center justify-between gap-3">
                  <span
                    className={`font-semibold uppercase tracking-wide text-sm ${
                      isSpam ? "text-amber-800" : "text-emerald-800"
                    }`}
                  >
                    {isSpam ? "Spam" : "Not spam"}
                  </span>
                  <span
                    className={`text-lg font-bold tabular-nums ${
                      isSpam ? "text-amber-700" : "text-emerald-700"
                    }`}
                  >
                    {confidencePct}%
                  </span>
                </div>
                <div className="mt-2 h-2 rounded-full bg-white/80 overflow-hidden">
                  <div
                    className={`h-full rounded-full transition-all duration-500 ${
                      isSpam ? "bg-amber-500" : "bg-emerald-500"
                    }`}
                    style={{ width: `${confidencePct}%` }}
                  />
                </div>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

export default App;
