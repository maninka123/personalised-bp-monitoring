#!/usr/bin/env node

import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import readline from "node:readline/promises";
import { stdin as input, stdout as output } from "node:process";

const HF_CHAT_URL = "https://router.huggingface.co/v1/chat/completions";
const DEFAULT_HF_MODEL = "google/gemma-4-31B-it:fastest";
const CONFIG_DIR = path.join(os.homedir(), ".sleep-aware-bp-report-assistant");
const CONFIG_PATH = path.join(CONFIG_DIR, "tokens.json");

const SYSTEM_INSTRUCTION = [
  "You are a BP report explanation assistant.",
  "Explain the current ABPM report in simple language.",
  "Use only the provided report data.",
  "Do not diagnose, prescribe, or recommend medication changes.",
  "If asked about changing medication, say that only the treating clinician can decide.",
  "For urgent symptoms, advise urgent medical care."
].join(" ");

const EXAMPLE_CONTEXT = {
  profile: "Non-dipper with morning surge and high variability",
  priority: "Review soon",
  awake_bp: "140/86",
  sleep_bp: "138/82",
  dipping_percentage: "1.4%",
  morning_surge: "24 mmHg",
  bp_variability: "High",
  review_points: [
    "Review night BP and sleep quality",
    "Review morning BP control",
    "Check stress, caffeine, adherence and measurement quality"
  ],
  clinical_boundary: "This explains the calculated report only. It is not medication advice."
};

function parseArgs(argv) {
  const args = {};
  for (let index = 2; index < argv.length; index += 1) {
    const item = argv[index];
    if (item.startsWith("--")) {
      const key = item.slice(2);
      const next = argv[index + 1];
      if (!next || next.startsWith("--")) {
        args[key] = true;
      } else {
        args[key] = next;
        index += 1;
      }
    }
  }
  return args;
}

function loadContext(filePath) {
  if (!filePath) return EXAMPLE_CONTEXT;
  return JSON.parse(fs.readFileSync(filePath, "utf8"));
}

function loadSavedToken() {
  if (!fs.existsSync(CONFIG_PATH)) return "";
  try {
    const config = JSON.parse(fs.readFileSync(CONFIG_PATH, "utf8"));
    return config.hugging_face || "";
  } catch {
    return "";
  }
}

function resolveHfToken(explicitToken) {
  return explicitToken || process.env.HF_TOKEN || loadSavedToken();
}

async function saveHfToken(token) {
  fs.mkdirSync(CONFIG_DIR, { recursive: true });
  fs.writeFileSync(CONFIG_PATH, JSON.stringify({ hugging_face: token }, null, 2), "utf8");
  console.log(`Saved Hugging Face token to ${CONFIG_PATH}`);
  console.log("The token is stored locally. Do not commit or share this file.");
}

function tokenStatus() {
  if (process.env.HF_TOKEN) return "Hugging Face Gemma 4: available from HF_TOKEN";
  if (loadSavedToken()) return `Hugging Face Gemma 4: available from ${CONFIG_PATH}`;
  return "Hugging Face Gemma 4: not configured";
}

function unsafeMedicationRequest(question) {
  const lower = question.toLowerCase();
  const medication = ["medicine", "medication", "drug", "dose", "dosage", "tablet", "pill"];
  const actions = ["change", "increase", "decrease", "stop", "start", "adjust", "switch"];
  return medication.some((word) => lower.includes(word)) && actions.some((word) => lower.includes(word));
}

function ruleBasedAnswer(question, ctx) {
  const lower = question.toLowerCase();
  const review = (ctx.review_points || []).join("; ");
  if (unsafeMedicationRequest(question)) {
    return "Only the treating clinician can decide medication changes. This report can support review, but it should not be used to change dose or timing automatically.";
  }
  if (lower.includes("why") || lower.includes("flag")) {
    return `The patient is flagged because sleep BP did not fall enough (${ctx.dipping_percentage}), morning surge was ${ctx.morning_surge}, and BP variability was ${ctx.bp_variability}. Monitoring priority: ${ctx.priority}. Review points: ${review}.`;
  }
  if (lower.includes("patient") || lower.includes("simple")) {
    return `Simple explanation: this report shows ${String(ctx.profile).toLowerCase()}. Awake BP was ${ctx.awake_bp}, sleep BP was ${ctx.sleep_bp}, sleep BP fall was ${ctx.dipping_percentage}, and morning rise was ${ctx.morning_surge}. Do not change medication without the treating clinician.`;
  }
  if (lower.includes("review") || lower.includes("next")) {
    return `Doctor review points: ${review}. Use this as clinical review support, not automatic treatment advice.`;
  }
  if (lower.includes("non-dipper") || lower.includes("non dipper")) {
    return `Non-dipper means BP did not fall enough during sleep. In this report, the sleep BP fall is ${ctx.dipping_percentage}.`;
  }
  return `This report shows ${ctx.profile}. Awake BP was ${ctx.awake_bp}, sleep BP was ${ctx.sleep_bp}, morning surge was ${ctx.morning_surge}, and variability was ${ctx.bp_variability}. Review points: ${review}.`;
}

async function askGemma(question, ctx, token, model) {
  if (!token) {
    return [
      "Hugging Face Gemma 4 is available when HF_TOKEN is set or a token is saved.",
      "",
      ruleBasedAnswer(question, ctx)
    ].join("\n");
  }
  const response = await fetch(HF_CHAT_URL, {
    method: "POST",
    headers: {
      Authorization: `Bearer ${token}`,
      "Content-Type": "application/json"
    },
    body: JSON.stringify({
      model: model || process.env.HF_MODEL || DEFAULT_HF_MODEL,
      stream: false,
      temperature: 0.2,
      max_tokens: 500,
      messages: [
        { role: "system", content: SYSTEM_INSTRUCTION },
        {
          role: "user",
          content: [
            "Current report summary JSON:",
            JSON.stringify(ctx, null, 2),
            "",
            `User question: ${question}`,
            "",
            "Answer using only the report summary JSON. Keep it concise and clinically safe."
          ].join("\n")
        }
      ]
    })
  });
  const data = await response.json();
  if (!response.ok) {
    throw new Error(data?.error?.message || data?.message || `HTTP ${response.status}`);
  }
  return data.choices?.[0]?.message?.content?.trim() || "No answer returned.";
}

async function main() {
  const args = parseArgs(process.argv);
  if (args["token-status"]) {
    console.log(tokenStatus());
    return;
  }
  if (args["save-token"]) {
    const rl = readline.createInterface({ input, output });
    const token = await rl.question("Paste Hugging Face token: ");
    rl.close();
    await saveHfToken(token.trim());
    return;
  }

  const ctx = loadContext(args.context);
  const question = args.question || "Why is this patient flagged?";
  const provider = args.provider || "Rule-based";
  const token = resolveHfToken(args.token);
  const answer = provider.toLowerCase().includes("gemma")
    ? await askGemma(question, ctx, token, args.model)
    : ruleBasedAnswer(question, ctx);
  console.log(`\n[${provider}]\n${answer}\n`);
}

main().catch((error) => {
  console.error(`Error: ${error.message}`);
  process.exit(1);
});
