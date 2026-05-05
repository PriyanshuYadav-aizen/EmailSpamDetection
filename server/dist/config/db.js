"use strict";
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
const mongoose_1 = __importDefault(require("mongoose"));
const dotenv_1 = __importDefault(require("dotenv"));
dotenv_1.default.config();
const connectDB = async () => {
    const uri = process.env.MONGO_URI;
    if (!uri) {
        console.warn("MONGO_URI is not set; starting server without MongoDB");
        return false;
    }
    try {
        await mongoose_1.default.connect(uri);
        console.log("MongoDB Connected");
        return true;
    }
    catch (error) {
        console.warn("MongoDB connection failed; starting server without MongoDB");
        console.warn(error);
        return false;
    }
};
exports.default = connectDB;
