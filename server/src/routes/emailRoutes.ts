import { Router } from "express";
import { analyzeEmail } from "../controllers/emailController";

const router = Router();

router.post("/analyze-email", analyzeEmail);

export default router;