import { Request, Response } from "express";
import pool from "../config/db";

export const analyzeEmail = async (req: Request, res: Response) => {
  try {
    const { content } = req.body;

    if (!content || content.trim() === "") {
      return res.status(400).json({ message: "Email content is required" });
    }

    // 🔥 Dummy prediction (we will replace with ML later)
    const isSpam = content.toLowerCase().includes("win");
    const prediction = isSpam ? "spam" : "not_spam";
    const confidence = isSpam ? 0.85 : 0.15;

    // Save to database
    const [result]: any = await pool.query(
      "INSERT INTO emails (content, prediction, confidence) VALUES (?, ?, ?)",
      [content, prediction, confidence]
    );

    return res.status(200).json({
      id: result.insertId,
      prediction,
      confidence,
    });

  } catch (err: unknown) {
    console.error("Analyze Email Error:", err);
    return res.status(500).json({ message: "Server error" });
  }
};