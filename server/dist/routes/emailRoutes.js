"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
const express_1 = require("express");
const emailController_1 = require("../controllers/emailController");
const router = (0, express_1.Router)();
router.post("/analyze-email", emailController_1.analyzeEmail);
router.get("/emails", emailController_1.getEmails);
exports.default = router;
