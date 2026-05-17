from __future__ import annotations

import json
import os
import textwrap
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any

import pandas as pd

from clinical_report_utils import feature_table, profile_label, review_points
from sleep_aware_bp_framework import is_true


DEFAULT_HF_MODEL = "google/gemma-4-31B-it:fastest"
HF_CHAT_URL = "https://router.huggingface.co/v1/chat/completions"
GEMINI_URL_TEMPLATE = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
GROQ_CHAT_URL = "https://api.groq.com/openai/v1/chat/completions"


SYSTEM_INSTRUCTION = """
You are a BP report explanation assistant. Explain the current ABPM report in simple language.
Use only the provided report data. Do not diagnose, prescribe, or recommend medication changes.
If asked about changing medication, say that only the treating clinician can decide.
For urgent symptoms, advise urgent medical care.
""".strip()


@dataclass(frozen=True)
class AssistantResponse:
    answer: str
    source: str


def build_report_context(profile: dict[str, Any]) -> dict[str, Any]:
    table = feature_table(profile)
    return {
        "profile": profile_label(profile),
        "priority": profile.get("priority"),
        "data_quality": profile.get("data_quality"),
        "valid_readings": int(profile.get("valid_readings", 0) or 0),
        "sleep_readings": int(profile.get("sleep_valid_readings", 0) or 0),
        "awake_bp": _bp_value(profile.get("awake_mean_sbp"), profile.get("awake_mean_dbp")),
        "sleep_bp": _bp_value(profile.get("sleep_mean_sbp"), profile.get("sleep_mean_dbp")),
        "mean_24h_bp": _bp_value(profile.get("mean_24h_sbp"), profile.get("mean_24h_dbp")),
        "dipping_percentage": _fmt(profile.get("dipping_pct_sbp"), "{:.1f}%"),
        "dipping_category": str(profile.get("dipping_category", "")).replace("_", " "),
        "morning_surge": _fmt(profile.get("morning_surge_sbp"), "{:.0f} mmHg"),
        "bp_variability": "High" if profile.get("high_variability") else "Not flagged",
        "sustained_high_bp": "Yes" if is_true(profile.get("sustained_high_bp")) else "No",
        "mean_pulse_pressure": _fmt(profile.get("mean_pp"), "{:.0f} mmHg"),
        "mean_map": _fmt(profile.get("mean_map"), "{:.0f} mmHg"),
        "mean_hr": _fmt(profile.get("mean_hr"), "{:.0f} bpm"),
        "review_points": [item["Doctor review point"] for item in review_points(profile)],
        "feature_table": table.to_dict(orient="records"),
        "clinical_boundary": (
            "This is clinician-review support only. It does not recommend automatic "
            "medication changes."
        ),
    }


def quick_questions() -> dict[str, str]:
    return {
        "Explain profile": "What does this BP profile mean?",
        "Why flagged?": "Why is this patient flagged?",
        "Explain to patient": "Explain this result to the patient in simple language.",
        "What to review next?": "What should the doctor review next?",
        "Is data quality enough?": "Is the data quality enough to interpret the report?",
    }


def answer_report_question(
    question: str,
    report_context: dict[str, Any],
    provider: str = "Rule-based",
    api_key: str | None = None,
    model: str | None = None,
) -> AssistantResponse:
    clean_question = (question or "").strip()
    if not clean_question:
        return AssistantResponse("Ask a question about the current BP report.", "Rule-based")

    if _unsafe_medication_request(clean_question):
        return AssistantResponse(
            "Only the treating clinician can decide medication changes. This report can support a review of night BP, morning BP control, adherence, triggers and medication timing, but it should not be used to change dose or timing automatically.",
            "Safety guardrail",
        )

    provider_key = provider.lower().strip()
    if provider_key == "hugging face gemma 4":
        return _cloud_answer_hugging_face(clean_question, report_context, api_key, model)
    if provider_key == "google gemini":
        return _cloud_answer_gemini(clean_question, report_context, api_key, model)
    if provider_key == "groq":
        return _cloud_answer_groq(clean_question, report_context, api_key, model)
    return AssistantResponse(rule_based_answer(clean_question, report_context), "Rule-based")


def rule_based_answer(question: str, ctx: dict[str, Any]) -> str:
    q = question.lower()
    profile = ctx.get("profile", "the current BP profile")
    review = "; ".join(ctx.get("review_points", []))

    if "why" in q or "flag" in q:
        reasons = []
        if "non dipper" in str(ctx.get("dipping_category")) or "reverse" in str(ctx.get("dipping_category")):
            reasons.append(f"sleep BP did not fall enough ({ctx['dipping_percentage']})")
        if ctx.get("morning_surge") not in {"N/A", "nan"}:
            reasons.append(f"morning surge was {ctx['morning_surge']}")
        if ctx.get("bp_variability") == "High":
            reasons.append("BP variability was high")
        if ctx.get("sustained_high_bp") == "Yes":
            reasons.append("BP burden stayed high across monitoring")
        if not reasons:
            reasons.append("no major rule-based warning flag was detected")
        return (
            "The patient is flagged because " + ", and ".join(reasons) + ". "
            f"Monitoring priority: {ctx.get('priority')}. Review points: {review}."
        )
    if "patient" in q or "simple" in q:
        return (
            f"Simple explanation: this report shows {profile.lower()}. "
            f"The average awake BP was {ctx['awake_bp']} and sleep BP was {ctx['sleep_bp']}. "
            f"The sleep BP fall was {ctx['dipping_percentage']} and the morning rise was {ctx['morning_surge']}. "
            "The doctor may use this to review night BP, sleep quality, caffeine or stress triggers, adherence and medication timing. "
            "Do not change medication without the treating clinician."
        )
    if "review" in q or "next" in q:
        return (
            f"Doctor review points: {review}. "
            "Use this as a checklist for clinical review, not as an automatic treatment instruction."
        )
    if "quality" in q or "enough" in q:
        return (
            f"Data quality: {ctx.get('data_quality')}. "
            f"There are {ctx.get('valid_readings')} valid readings and {ctx.get('sleep_readings')} sleep readings. "
            "If sleep readings are limited, the night-time pattern should be interpreted cautiously or ABPM may need repeating."
        )
    if "non-dipper" in q or "non dipper" in q:
        return (
            "Non-dipper means blood pressure did not fall by the expected amount during sleep. "
            f"In this report, the sleep BP fall is {ctx['dipping_percentage']}. "
            "This can prompt review of night BP, sleep quality and related triggers."
        )
    return (
        f"This report shows {profile}. Awake BP was {ctx['awake_bp']}, sleep BP was {ctx['sleep_bp']}, "
        f"dipping was {ctx['dipping_percentage']}, morning surge was {ctx['morning_surge']}, "
        f"and variability was {ctx['bp_variability']}. Review points: {review}."
    )


def format_context_for_prompt(ctx: dict[str, Any]) -> str:
    return json.dumps(ctx, indent=2, ensure_ascii=True)


def _cloud_answer_hugging_face(
    question: str,
    ctx: dict[str, Any],
    api_key: str | None,
    model: str | None,
) -> AssistantResponse:
    token = api_key or os.getenv("HF_TOKEN")
    if not token:
        return AssistantResponse(
            "Hugging Face Gemma 4 is available when HF_TOKEN is set or entered in the app. Using the built-in rule-based explanation instead.\n\n"
            + rule_based_answer(question, ctx),
            "Rule-based fallback",
        )
    payload = {
        "model": model or os.getenv("HF_MODEL", DEFAULT_HF_MODEL),
        "messages": _messages(question, ctx),
        "stream": False,
        "temperature": 0.2,
        "max_tokens": 500,
    }
    return AssistantResponse(_post_openai_compatible(HF_CHAT_URL, payload, token), "Hugging Face Gemma 4")


def _cloud_answer_gemini(
    question: str,
    ctx: dict[str, Any],
    api_key: str | None,
    model: str | None,
) -> AssistantResponse:
    token = api_key or os.getenv("GEMINI_API_KEY")
    if not token:
        return AssistantResponse(
            "Gemini API is available when GEMINI_API_KEY is set or entered in the app. Using the built-in rule-based explanation instead.\n\n"
            + rule_based_answer(question, ctx),
            "Rule-based fallback",
        )
    prompt = _prompt_text(question, ctx)
    payload = {
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": 0.2, "maxOutputTokens": 500},
    }
    url = GEMINI_URL_TEMPLATE.format(model=model or os.getenv("GEMINI_MODEL", "gemini-2.5-flash"), api_key=token)
    data = _post_json(url, payload, None)
    try:
        answer = data["candidates"][0]["content"]["parts"][0]["text"]
    except (KeyError, IndexError, TypeError) as exc:
        raise RuntimeError(f"Unexpected Gemini response: {data}") from exc
    return AssistantResponse(answer.strip(), "Google Gemini")


def _cloud_answer_groq(
    question: str,
    ctx: dict[str, Any],
    api_key: str | None,
    model: str | None,
) -> AssistantResponse:
    token = api_key or os.getenv("GROQ_API_KEY")
    if not token:
        return AssistantResponse(
            "Groq is available when GROQ_API_KEY is set or entered in the app. Using the built-in rule-based explanation instead.\n\n"
            + rule_based_answer(question, ctx),
            "Rule-based fallback",
        )
    payload = {
        "model": model or os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"),
        "messages": _messages(question, ctx),
        "stream": False,
        "temperature": 0.2,
        "max_tokens": 500,
    }
    return AssistantResponse(_post_openai_compatible(GROQ_CHAT_URL, payload, token), "Groq")


def _messages(question: str, ctx: dict[str, Any]) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_INSTRUCTION},
        {"role": "user", "content": _prompt_text(question, ctx)},
    ]


def _prompt_text(question: str, ctx: dict[str, Any]) -> str:
    return textwrap.dedent(
        f"""
        Current report summary JSON:
        {format_context_for_prompt(ctx)}

        User question:
        {question}

        Answer using only the report summary JSON. Keep it concise and clinically safe.
        """
    ).strip()


def _post_openai_compatible(url: str, payload: dict[str, Any], token: str) -> str:
    data = _post_json(url, payload, token)
    try:
        return data["choices"][0]["message"]["content"].strip()
    except (KeyError, IndexError, TypeError) as exc:
        raise RuntimeError(f"Unexpected chat response: {data}") from exc


def _post_json(url: str, payload: dict[str, Any], token: str | None) -> dict[str, Any]:
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=45) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Cloud assistant request failed ({exc.code}): {detail}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Cloud assistant request failed: {exc.reason}") from exc


def _unsafe_medication_request(question: str) -> bool:
    lowered = question.lower()
    medication_terms = ["medicine", "medication", "drug", "dose", "dosage", "tablet", "pill"]
    action_terms = ["change", "increase", "decrease", "stop", "start", "adjust", "switch"]
    return any(term in lowered for term in medication_terms) and any(term in lowered for term in action_terms)


def _bp_value(sbp: Any, dbp: Any) -> str:
    if pd.isna(sbp) or pd.isna(dbp):
        return "N/A"
    return f"{float(sbp):.0f}/{float(dbp):.0f}"


def _fmt(value: Any, pattern: str) -> str:
    if pd.isna(value):
        return "N/A"
    return pattern.format(float(value))
