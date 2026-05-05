"use strict";
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
const express_1 = __importDefault(require("express"));
const cors_1 = __importDefault(require("cors"));
const dotenv_1 = __importDefault(require("dotenv"));
const emailRoutes_1 = __importDefault(require("./routes/emailRoutes"));
const db_1 = __importDefault(require("./config/db"));
dotenv_1.default.config();
const app = (0, express_1.default)();
const PORT = process.env.PORT || 5000;
const CLIENT_ORIGIN = process.env.CLIENT_ORIGIN;
const allowedOrigins = [
    CLIENT_ORIGIN,
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "https://spam-client.onrender.com",
].filter((origin) => Boolean(origin));
app.use((0, cors_1.default)({
    origin: (origin, callback) => {
        if (!origin || allowedOrigins.includes(origin)) {
            callback(null, true);
            return;
        }
        callback(new Error(`CORS blocked for origin: ${origin}`));
    },
}));
app.use(express_1.default.json());
app.get("/health", (_req, res) => {
    res.status(200).json({ status: "ok" });
});
app.use("/api", emailRoutes_1.default);
const startServer = async () => {
    await (0, db_1.default)();
    app.listen(PORT, () => {
        console.log(`Server running on port ${PORT}`);
    });
};
startServer().catch((error) => {
    console.error("Failed to start server:", error);
    process.exit(1);
});
