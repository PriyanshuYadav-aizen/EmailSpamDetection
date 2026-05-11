import express from "express";
import cors from "cors";
import dotenv from "dotenv";
import emailRoutes from "./routes/emailRoutes";
import connectDB from "./config/db";

dotenv.config();

const app = express();
const PORT = process.env.PORT || 5000;
const CLIENT_ORIGIN = process.env.CLIENT_ORIGIN;
const allowedOrigins = [
  CLIENT_ORIGIN,
  "http://localhost:5173",
  "http://127.0.0.1:5173",
  "https://spam-client.onrender.com",
].filter((origin): origin is string => Boolean(origin));

app.use(
  cors({
    origin: (origin, callback) => {
      if (!origin || allowedOrigins.includes(origin)) {
        callback(null, true);
        return;
      }

      callback(new Error(`CORS blocked for origin: ${origin}`));
    },
  })
);
app.use(express.json());

app.get("/health", (_req, res) => {
  res.status(200).json({ status: "ok" });
});

app.use("/api", emailRoutes);

// ── Keep-alive pinger ────────────────────────────────────────────────────
// Render free-tier spins down services after ~15 min of inactivity.
// We self-ping every 5 min to stay awake, and also ping the ML API.
const KEEP_ALIVE_INTERVAL_MS = 5 * 60 * 1000; // 5 minutes

function startKeepAlive() {
  const selfUrl = process.env.RENDER_EXTERNAL_URL || `http://localhost:${PORT}`;
  const mlApiUrl = process.env.ML_API_URL;

  setInterval(async () => {
    try {
      await fetch(`${selfUrl}/health`);
      console.log("[keep-alive] Self-ping OK");
    } catch (err) {
      console.warn("[keep-alive] Self-ping failed:", (err as Error).message);
    }

    if (mlApiUrl) {
      try {
        await fetch(`${mlApiUrl}/health`);
        console.log("[keep-alive] ML API ping OK");
      } catch (err) {
        console.warn("[keep-alive] ML API ping failed:", (err as Error).message);
      }
    }
  }, KEEP_ALIVE_INTERVAL_MS);

  console.log(
    `[keep-alive] Started — pinging every ${KEEP_ALIVE_INTERVAL_MS / 60000} min`
  );
}

const startServer = async () => {
  await connectDB();
  app.listen(PORT, () => {
    console.log(`Server running on port ${PORT}`);
    startKeepAlive();
  });
};

startServer().catch((error) => {
  console.error("Failed to start server:", error);
  process.exit(1);
});