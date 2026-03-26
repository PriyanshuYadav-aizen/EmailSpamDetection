import { Request, Response } from "express";
import Email from "../models/Email";

export const analyzeEmail = async (req: Request, res: Response) => {
  try {
    const content =
      typeof req.body?.content === "string" ? req.body.content : "";

    if (!content || content.trim() === "") {
      return res.status(400).json({ message: "Email content is required" });
    }

    // 🔥 Dummy prediction (we will replace with ML later)
    const isSpam = content.toLowerCase().includes("win");
    const prediction = isSpam ? "spam" : "not_spam";
    const confidence = isSpam ? 0.85 : 0.15;

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