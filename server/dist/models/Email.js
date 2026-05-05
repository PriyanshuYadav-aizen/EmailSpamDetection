"use strict";
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
const mongoose_1 = __importDefault(require("mongoose"));
const emailSchema = new mongoose_1.default.Schema({
    content: { type: String, required: true },
    prediction: { type: String, required: true },
    confidence: { type: Number, required: true },
}, { timestamps: true });
exports.default = mongoose_1.default.model("Email", emailSchema);
