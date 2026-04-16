import { Request, Response } from "express";
import axios from "axios";
import Email from "../models/Email";

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

    const savedEmail = await Email.create({
      content,
      prediction,
      confidence,
    });

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
    const emails = await Email.find().sort({ createdAt: -1 });
    return res.status(200).json(emails);
  } catch (err: unknown) {
    console.error("Get Emails Error:", err);
    return res.status(500).json({ message: "Server error" });
  }
};