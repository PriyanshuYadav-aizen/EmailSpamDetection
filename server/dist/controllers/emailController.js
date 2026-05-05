"use strict";
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
exports.getEmails = exports.analyzeEmail = void 0;
const axios_1 = __importDefault(require("axios"));
const crypto_1 = require("crypto");
const mongoose_1 = __importDefault(require("mongoose"));
const Email_1 = __importDefault(require("../models/Email"));
const fallbackEmails = [];
const isMongoConnected = () => mongoose_1.default.connection.readyState === 1;
const analyzeEmail = async (req, res) => {
    try {
        const content = typeof req.body?.content === "string" ? req.body.content : "";
        if (!content || content.trim() === "") {
            return res.status(400).json({ message: "Email content is required" });
        }
        const mlApiUrl = process.env.ML_API_URL;
        if (!mlApiUrl) {
            return res.status(500).json({ message: "ML service URL is not configured" });
        }
        const modelResponse = await axios_1.default.post(`${mlApiUrl.replace(/\/$/, "")}/predict`, { content }, { timeout: 30000 });
        const rawVerdict = String(modelResponse.data?.verdict ?? modelResponse.data?.label ?? "").toUpperCase();
        const rawScore = Number(modelResponse.data?.confidence ?? modelResponse.data?.score);
        const score = Number.isFinite(rawScore)
            ? Math.max(0, Math.min(1, rawScore))
            : 0;
        const prediction = rawVerdict === "SPAM" ? "spam" : "not_spam";
        const confidence = prediction === "spam" ? score : 1 - score;
        const savedEmail = isMongoConnected()
            ? await Email_1.default.create({
                content,
                prediction,
                confidence,
            })
            : (() => {
                const now = new Date();
                const email = {
                    _id: (0, crypto_1.randomUUID)(),
                    content,
                    prediction,
                    confidence,
                    createdAt: now,
                    updatedAt: now,
                };
                fallbackEmails.unshift(email);
                return email;
            })();
        return res.status(200).json({
            id: savedEmail._id,
            prediction,
            confidence,
        });
    }
    catch (err) {
        if (axios_1.default.isAxiosError(err)) {
            console.error("Model API Error:", err.response?.data ?? err.message);
            return res.status(503).json({ message: "ML service unavailable" });
        }
        console.error("Analyze Email Error:", err);
        return res.status(500).json({ message: "Server error" });
    }
};
exports.analyzeEmail = analyzeEmail;
const getEmails = async (_req, res) => {
    try {
        const emails = isMongoConnected()
            ? await Email_1.default.find().sort({ createdAt: -1 })
            : [...fallbackEmails].sort((left, right) => right.createdAt.getTime() - left.createdAt.getTime());
        return res.status(200).json(emails);
    }
    catch (err) {
        console.error("Get Emails Error:", err);
        return res.status(500).json({ message: "Server error" });
    }
};
exports.getEmails = getEmails;
