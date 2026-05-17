from __future__ import annotations

import argparse
import json
from pathlib import Path

from bp_report_assistant import (
    answer_report_question,
    build_report_context,
    quick_questions,
)
from clinical_report_utils import build_patient_profile, example_patient_abpm


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ask safe questions about an already calculated BP report summary."
    )
    parser.add_argument("--context", type=Path, help="Optional report context JSON file.")
    parser.add_argument("--question", help="Ask one question and exit.")
    parser.add_argument(
        "--provider",
        default="Rule-based",
        choices=["Rule-based", "Hugging Face Gemma 4", "Google Gemini", "Groq"],
        help="Assistant backend. Cloud providers require an API key environment variable.",
    )
    parser.add_argument("--model", help="Optional cloud model override.")
    args = parser.parse_args()

    context = _load_context(args.context)
    if args.question:
        response = answer_report_question(args.question, context, provider=args.provider, model=args.model)
        print(f"\n[{response.source}]\n{response.answer}\n")
        return

    print("Ask About This BP Report")
    print("Type a question, a shortcut number, or 'exit'.")
    print("\nQuick questions:")
    questions = list(quick_questions().items())
    for idx, (label, question) in enumerate(questions, start=1):
        print(f"{idx}. {label}: {question}")

    while True:
        raw = input("\nQuestion> ").strip()
        if raw.lower() in {"exit", "quit", "q"}:
            break
        if raw.isdigit() and 1 <= int(raw) <= len(questions):
            raw = questions[int(raw) - 1][1]
        response = answer_report_question(raw, context, provider=args.provider, model=args.model)
        print(f"\n[{response.source}]\n{response.answer}")


def _load_context(path: Path | None) -> dict:
    if path:
        return json.loads(path.read_text(encoding="utf-8"))
    valid = example_patient_abpm()
    profile = build_patient_profile(valid)
    return build_report_context(profile)


if __name__ == "__main__":
    main()
