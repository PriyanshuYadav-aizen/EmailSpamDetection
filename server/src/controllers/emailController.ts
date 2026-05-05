import { Request, Response } from "express";
import axios from "axios";
import { randomUUID } from "crypto";
import mongoose from "mongoose";
import Email from "../models/Email";

type StoredEmail = {
  _id: string;
  content: string;
  prediction: string;
  confidence: number;
  createdAt: Date;
  updatedAt: Date;
};

const fallbackEmails: StoredEmail[] = [];

const isMongoConnected = () => mongoose.connection.readyState === 1;

export const analyzeEmail = async (req: Request, res: Response) => {
  try {
    const content =
      typeof req.body?.content === "string" ? req.body.content : "";

    if (!content || content.trim() === "") {
      return res.status(400).json({ message: "Email content is required" });
    }

    const mlApiUrl = process.env.ML_API_URL;

    if (!mlApiUrl) {
      return res.status(500).json({ message: "ML service URL is not configured" });
    }

    const modelResponse = await axios.post(
      `${mlApiUrl.replace(/\/$/, "")}/predict`,
      { content },
      { timeout: 30000 }
    );

    const rawVerdict = String(
      modelResponse.data?.verdict ?? modelResponse.data?.label ?? ""
    ).toUpperCase();
    const rawScore = Number(
      modelResponse.data?.confidence ?? modelResponse.data?.score
    );
    const score = Number.isFinite(rawScore)
      ? Math.max(0, Math.min(1, rawScore))
      : 0;

    const prediction = rawVerdict === "SPAM" ? "spam" : "not_spam";
    const confidence = prediction === "spam" ? score : 1 - score;

    const savedEmail = isMongoConnected()
      ? await Email.create({
          content,
          prediction,
          confidence,
        })
      : (() => {
          const now = new Date();
          const email = {
            _id: randomUUID(),
            content,
            prediction,
            confidence,
            createdAt: now,
            updatedAt: now,
          } satisfies StoredEmail;

          fallbackEmails.unshift(email);
          return email;
        })();

    return res.status(200).json({
      id: savedEmail._id,
      prediction,
      confidence,
    });

  } catch (err: unknown) {
    if (axios.isAxiosError(err)) {
      console.error("Model API Error:", err.response?.data ?? err.message);
      return res.status(503).json({ message: "ML service unavailable" });
    }

    console.error("Analyze Email Error:", err);
    return res.status(500).json({ message: "Server error" });
  }
};

export const getEmails = async (_req: Request, res: Response) => {
  try {
    const emails = isMongoConnected()
      ? await Email.find().sort({ createdAt: -1 })
      : [...fallbackEmails].sort(
          (left, right) => right.createdAt.getTime() - left.createdAt.getTime()
        );
    return res.status(200).json(emails);
  } catch (err: unknown) {
    console.error("Get Emails Error:", err);
    return res.status(500).json({ message: "Server error" });
  }
};