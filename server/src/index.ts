import express from "express";
import cors from "cors";
import dotenv from "dotenv";
import emailRoutes from "./routes/emailRoutes";
import connectDB from "./config/db";

dotenv.config();

const app = express();
const PORT = process.env.PORT || 5000;
const CLIENT_ORIGIN = process.env.CLIENT_ORIGIN;

app.use(cors(CLIENT_ORIGIN ? { origin: CLIENT_ORIGIN } : undefined));
app.use(express.json());

app.use("/api", emailRoutes);

const startServer = async () => {
  await connectDB();
  app.listen(PORT, () => {
    console.log(`Server running on port ${PORT}`);
  });
};

startServer().catch((error) => {
  console.error("Failed to start server:", error);
  process.exit(1);
});