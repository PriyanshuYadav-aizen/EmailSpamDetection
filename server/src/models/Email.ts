import mongoose from "mongoose";

const emailSchema = new mongoose.Schema(
  {
    content: { type: String, required: true },
    prediction: { type: String, required: true },
    confidence: { type: Number, required: true },
  },
  { timestamps: true }
);

export default mongoose.model("Email", emailSchema);
