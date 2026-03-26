import { Router } from "express";
import { analyzeEmail, getEmails } from "../controllers/emailController";

const router = Router();

router.post("/analyze-email", analyzeEmail);
router.get("/emails", getEmails);

export default router;