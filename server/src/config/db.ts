import mongoose from "mongoose";
import dotenv from "dotenv";

dotenv.config();

const connectDB = async () => {
  const uri = process.env.MONGO_URI;

  if (!uri) {
    console.warn("MONGO_URI is not set; starting server without MongoDB");
    return false;
  }

  try {
    await mongoose.connect(uri);
    console.log("MongoDB Connected");
    return true;
  } catch (error) {
    console.warn("MongoDB connection failed; starting server without MongoDB");
    console.warn(error);
    return false;
  }
};

export default connectDB;