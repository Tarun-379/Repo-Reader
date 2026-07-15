"""
readme_generator.py
Project Outcome 3: draft a README section directly from the indexed codebase,
rather than asking the user to describe their own project to the LLM.

Strategy: build a lightweight structural summary (directory tree + per-file
top-level definitions) which is cheap and deterministic, then hand that
summary to Gemini to write prose around. We deliberately do NOT dump raw
file contents here -- that would blow the context window on anything but a
toy repo, and a directory + signature summary is what a human skimming the
repo for the first time would actually build a README from.
"""

import os
import re
from collections import defaultdict

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate

# Cheap, regex-based signature extraction per language. This is intentionally
# simple (no AST) so it works uniformly across languages without extra deps.
SIGNATURE_PATTERNS = {
    "py": [r"^\s*def\s+(\w+)\(", r"^\s*class\s+(\w+)"],
    "js": [r"^\s*(?:export\s+)?function\s+(\w+)\(", r"^\s*class\s+(\w+)"],
    "ts": [r"^\s*(?:export\s+)?function\s+(\w+)\(", r"^\s*class\s+(\w+)"],
    "java": [r"^\s*(?:public|private|protected)?\s*(?:static\s+)?\w[\w<>\[\]]*\s+(\w+)\s*\("],
    "go": [r"^\s*func\s+(?:\([^)]*\)\s*)?(\w+)\("],
}


def build_repo_summary(repo_path: str, max_files: int = 60) -> str:
    tree_lines = []
    signatures_by_file = defaultdict(list)

    file_count = 0
    for dirpath, dirnames, filenames in os.walk(repo_path):
        dirnames[:] = [
            d for d in dirnames
            if d not in {".git", "node_modules", "__pycache__", ".venv", "venv", "dist", "build"}
            and not d.startswith(".")
        ]
        rel_dir = os.path.relpath(dirpath, repo_path)
        depth = 0 if rel_dir == "." else rel_dir.count(os.sep) + 1
        if rel_dir != ".":
            tree_lines.append("  " * depth + os.path.basename(dirpath) + "/")

        for fname in sorted(filenames):
            if file_count >= max_files:
                break
            ext = fname.split(".")[-1].lower()
            tree_lines.append("  " * (depth + 1) + fname)

            if ext in SIGNATURE_PATTERNS:
                full_path = os.path.join(dirpath, fname)
                try:
                    with open(full_path, "r", encoding="utf-8", errors="ignore") as f:
                        text = f.read()
                except Exception:
                    continue
                found = []
                for pattern in SIGNATURE_PATTERNS[ext]:
                    found.extend(re.findall(pattern, text, re.MULTILINE))
                if found:
                    rel_path = os.path.relpath(full_path, repo_path)
                    signatures_by_file[rel_path] = found[:15]
            file_count += 1

    summary = "DIRECTORY STRUCTURE:\n" + "\n".join(tree_lines[:400])
    summary += "\n\nKEY DEFINITIONS PER FILE:\n"
    for path, sigs in signatures_by_file.items():
        summary += f"- {path}: {', '.join(sigs)}\n"
    return summary


README_PROMPT = """You are writing documentation for a software repository.
Based ONLY on the structural summary below (directory layout and detected
function/class names), draft a README.md section with these parts:

1. ## Overview -- one paragraph, plain language, on what this project likely does
   based on naming and structure. Be honest if the purpose is ambiguous from
   structure alone.
2. ## Project Structure -- a short annotated breakdown of the main folders/files.
3. ## Key Components -- bullet list of the most important classes/functions found,
   with the file each lives in.

Do not invent features, APIs, or behavior that isn't implied by the structure below.

STRUCTURAL SUMMARY:
{summary}
"""


def generate_readme_section(repo_path: str, google_api_key: str) -> str:
    summary = build_repo_summary(repo_path)
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash", google_api_key=google_api_key, temperature=0.2
    )
    prompt = ChatPromptTemplate.from_template(README_PROMPT)
    chain = prompt | llm
    response = chain.invoke({"summary": summary})
    return response.content
