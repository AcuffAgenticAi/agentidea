"""
================================================================================
AI Agent Suggestion Agent
================================================================================
Author  : Acuff Automation & Analytics
Version : 2.0.0

OVERVIEW
--------
This script automates personalized AI agent recommendations for businesses.
Given a company profile, it runs a multi-step AI pipeline to:

    1. Classify the company's industry
    2. Identify likely operational pain points
    3. Recommend 4 targeted AI agents with ROI estimates
    4. Compose a professional outbound email report
    5. Scan the report through an AI-powered content guardrail
    6. Save the report to disk and (if guardrail passes) send via SMTP

USAGE
-----
    # Interactive CLI (default):
    python agentidea.py

    # Load from CSV file:
    python agentidea.py --csv companies.csv

    # Load from JSON file:
    python agentidea.py --json companies.json

CSV FORMAT (required columns)
------------------------------
    company_name, email, services, employees

    Example:
        company_name,email,services,employees
        Acme Corp,owner@acme.com,Industrial supply distribution,120

JSON FORMAT
-----------
    Single object or list of objects with the same four keys.

    Example:
        [
          {
            "company_name": "Acme Corp",
            "email": "owner@acme.com",
            "services": "Industrial supply distribution",
            "employees": 120
          }
        ]

ENVIRONMENT VARIABLES
---------------------
    Required:
        ANTHROPIC_API_KEY   Your Anthropic API key (sk-ant-...)

    Optional (SMTP sending):
        SMTP_HOST           Mail server host     (default: smtp.gmail.com)
        SMTP_PORT           Mail server port     (default: 587)
        SMTP_USER           Login username / sender address
        SMTP_PASSWORD       Login password or app password
        SMTP_FROM           From address         (default: SMTP_USER)

    If SMTP_USER or SMTP_PASSWORD are not set, the script will generate
    and save reports but skip the email send step.

OUTPUT
------
    Reports are saved to ./reports/<company_name>_<timestamp>_report.txt
================================================================================
"""

# ── Standard library ──────────────────────────────────────────────────────────
import os
import re
import csv
import json
import time
import smtplib
import argparse
from dataclasses import dataclass
from datetime import datetime
from typing import List, Dict

# ── Email formatting ───────────────────────────────────────────────────────────
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# ── Third-party ────────────────────────────────────────────────────────────────
import anthropic  # pip install anthropic


# ==============================================================================
# CONFIGURATION
# ==============================================================================

# Fail loudly at startup if the API key is missing rather than crashing
# mid-workflow on the first API call.
_api_key = os.getenv("ANTHROPIC_API_KEY")
if not _api_key:
    raise EnvironmentError(
        "ANTHROPIC_API_KEY environment variable is not set.\n"
        "Set it before running:  export ANTHROPIC_API_KEY=sk-ant-..."
    )

# Anthropic client — shared across all API calls in the session
client = anthropic.Anthropic(api_key=_api_key)

# Model to use for all completions. Sonnet balances quality and speed.
MODEL = "claude-sonnet-4-20250514"

# Retry settings for transient API failures (network blips, rate limits, etc.)
MAX_RETRIES = 3    # number of attempts before giving up
RETRY_DELAY = 5    # seconds to wait between retries

# SMTP credentials pulled from environment so secrets never live in source code
SMTP_HOST     = os.getenv("SMTP_HOST", "smtp.gmail.com")
SMTP_PORT     = int(os.getenv("SMTP_PORT", 587))
SMTP_USER     = os.getenv("SMTP_USER", "")
SMTP_PASSWORD = os.getenv("SMTP_PASSWORD", "")
SMTP_FROM     = os.getenv("SMTP_FROM", SMTP_USER)  # falls back to SMTP_USER

# Directory where .txt report files are saved
OUTPUT_DIR = "reports"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Simple email regex used for input validation
EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


# ==============================================================================
# DATA MODELS
# ==============================================================================

@dataclass
class CompanyProfile:
    """
    Raw input data for a single company.

    Attributes:
        company_name : Display name of the business.
        email        : Recipient address for the generated report.
        services     : Plain-text description of what the company does.
        employees    : Approximate headcount (used for ROI scaling).
    """
    company_name: str
    email:        str
    services:     str
    employees:    int


@dataclass
class DiagnosisResult:
    """
    AI-generated analysis produced for a company.

    Attributes:
        industry            : Classified industry label.
        pain_points         : List of identified operational challenges.
        recommended_agents  : Formatted text blocks describing each AI agent.
        roi_summary         : 3-sentence ROI impact paragraph.
    """
    industry:           str
    pain_points:        List[str]
    recommended_agents: List[str]
    roi_summary:        str


# ==============================================================================
# CLAUDE API HELPER
# ==============================================================================

def ask_claude(prompt: str, temperature: float = 0.3) -> str:
    """
    Send a prompt to Claude and return the plain-text response.

    Wraps the API call in a retry loop so transient errors (timeouts,
    rate limits, brief service interruptions) don't abort the whole run.

    Args:
        prompt      : The full text prompt to send.
        temperature : Controls response randomness.
                      Lower  → more deterministic (good for classification).
                      Higher → more creative (good for recommendations).

    Returns:
        Stripped string response from Claude.

    Raises:
        RuntimeError: If all retry attempts are exhausted.
    """
    last_err = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = client.messages.create(
                model=MODEL,
                max_tokens=1024,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
            )
            return response.content[0].text.strip()

        except anthropic.APIError as e:
            last_err = e
            print(f"  [API] Attempt {attempt}/{MAX_RETRIES} failed: {e}")
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_DELAY)

    raise RuntimeError(
        f"Claude API failed after {MAX_RETRIES} attempts. Last error: {last_err}"
    )


# ==============================================================================
# STEP 1 — INDUSTRY CLASSIFICATION
# ==============================================================================

def classify_industry(profile: CompanyProfile) -> str:
    """
    Ask Claude to classify the company into one of the supported industries.

    Uses a low temperature (0.2) to keep the label deterministic — we want
    a consistent classification, not a creative interpretation.

    Args:
        profile: The company whose industry we are classifying.

    Returns:
        A single industry label string, e.g. "retail" or "logistics".
    """
    prompt = f"""
Classify the industry of the following company.

Company:   {profile.company_name}
Services:  {profile.services}
Employees: {profile.employees}

Choose the single closest match from this list:
  retail, manufacturing, nonprofit, medical,
  enterprise services, technology, logistics, finance

Return only the industry name — no explanation, no punctuation.
"""
    return ask_claude(prompt, temperature=0.2)


# ==============================================================================
# STEP 2 — PAIN POINT DIAGNOSIS
# ==============================================================================

def diagnose_pain_points(profile: CompanyProfile, industry: str) -> List[str]:
    """
    Identify the five most common operational pain points for this company.

    The model returns a numbered list; each non-empty line becomes a separate
    item so the numbering is preserved in the final report.

    Args:
        profile  : Company profile for context and sizing.
        industry : The industry label returned by classify_industry().

    Returns:
        A list of strings, one pain point per item.
    """
    prompt = f"""
A company operates in the {industry} industry.

Company description: {profile.services}
Employee count:      {profile.employees}

List the 5 most common operational pain points this type of company
typically experiences. Output as a numbered list, one item per line.
No preamble or closing remarks.
"""
    text = ask_claude(prompt, temperature=0.3)

    # Split on newlines; filter blanks so empty lines don't become list items
    return [line.strip() for line in text.split("\n") if line.strip()]


# ==============================================================================
# STEP 3 — AI AGENT RECOMMENDATIONS
# ==============================================================================

def recommend_agents(
    profile:     CompanyProfile,
    industry:    str,
    pain_points: List[str],
) -> Dict[str, str]:
    """
    Recommend four AI autonomous agents tailored to the company's pain points.

    A slightly higher temperature (0.4) allows for more varied and creative
    agent ideas across different company profiles.

    Args:
        profile     : Company profile for personalisation context.
        industry    : Classified industry label from Step 1.
        pain_points : Pain points list from Step 2.

    Returns:
        A dict with key "agents" containing the formatted recommendation text.
    """
    formatted_points = "\n".join(pain_points)

    prompt = f"""
A company in the {industry} industry has the following pain points:

{formatted_points}

Company details:
  Name:      {profile.company_name}
  Services:  {profile.services}
  Employees: {profile.employees}

Recommend exactly 4 AI autonomous agents that directly address these problems.
For each agent use this exact format, separated by a blank line:

Agent Name:
Description:
ROI:

No preamble or closing remarks.
"""
    return {"agents": ask_claude(prompt, temperature=0.4)}


# ==============================================================================
# STEP 4 — ROI SUMMARY
# ==============================================================================

def generate_roi_summary(industry: str, employees: int) -> str:
    """
    Generate a concise business ROI summary for this industry and company size.

    Args:
        industry  : The company's industry classification.
        employees : Headcount used to scale the ROI estimate.

    Returns:
        A 3-sentence plain-text ROI paragraph.
    """
    prompt = f"""
Estimate the business ROI impact of implementing AI automation in a
{industry} company with approximately {employees} employees.

Write exactly 3 sentences. Be specific and business-focused.
No preamble or closing remarks.
"""
    return ask_claude(prompt, temperature=0.3)


# ==============================================================================
# STEP 5 — EMAIL REPORT ASSEMBLY
# ==============================================================================

def generate_email_report(
    profile:   CompanyProfile,
    diagnosis: DiagnosisResult,
) -> str:
    """
    Compose the full outbound email report as a plain-text string.

    The "Subject:" line is embedded at the top so the saved .txt file is
    self-contained. The SMTP sender strips it before building the MIME message.

    Args:
        profile   : The company the report is written for.
        diagnosis : The AI-generated analysis from Steps 1–4.

    Returns:
        Complete email body as a multi-line string.
    """
    pain_list  = "\n".join(f"  - {p}" for p in diagnosis.pain_points)
    agent_text = "\n".join(diagnosis.recommended_agents)

    return f"""Subject: AI Automation Opportunities for {profile.company_name}

Hello,

Thank you for your interest in AI automation solutions.

Based on our analysis of your company information, we have identified the
following opportunities where autonomous AI agents could create operational
efficiencies and measurable ROI.

COMPANY OVERVIEW
----------------
  Company:   {profile.company_name}
  Employees: {profile.employees}
  Industry:  {diagnosis.industry}

COMMON OPERATIONAL CHALLENGES
------------------------------
{pain_list}

RECOMMENDED AI AGENTS
----------------------
{agent_text}

ESTIMATED ROI IMPACT
---------------------
{diagnosis.roi_summary}

NEXT STEPS
----------
We would be happy to schedule a consultation to discuss building custom
AI agents tailored specifically to your operations.

Best regards,

Acuff Automation & Analytics
AI Agentic Development
"""


# ==============================================================================
# STEP 6 — CONTENT GUARDRAIL
# ==============================================================================

def scan_report_guardrail(report: str) -> dict:
    """
    Use Claude as a second-pass reviewer to validate the report before sending.

    This AI-powered guardrail catches tone issues, incoherent content, or
    anything that should not reach a real business contact. The temperature is
    set very low (0.1) to make the pass/fail decision as consistent as possible.

    The guardrail is intentionally fail-safe: if the JSON response cannot be
    parsed for any reason, the email is blocked rather than sent through.

    Args:
        report: The assembled email text to review.

    Returns:
        A dict with:
            "pass"   (bool) — True if the report is safe to send.
            "reason" (str)  — Brief explanation of the decision.
    """
    prompt = f"""
You are a strict email content reviewer for a professional B2B automation company.

Review the following outbound email and assess whether it is:
  1. Professional and appropriate in tone
  2. Free of hallucinated or obviously false claims
  3. Free of sensitive or harmful content
  4. Coherent and complete

Email to review:
---
{report}
---

Respond with JSON only — no markdown fences, no text outside the JSON object.
Use exactly one of these formats:
  {{"pass": true,  "reason": "Brief explanation"}}
  {{"pass": false, "reason": "Brief explanation of what failed"}}
"""
    raw = ask_claude(prompt, temperature=0.1)

    # Strip accidental markdown code fences (```json ... ```) that can appear
    # even when the model is instructed not to include them.
    clean = re.sub(r"^```(?:json)?|```$", "", raw.strip(), flags=re.MULTILINE).strip()

    try:
        result = json.loads(clean)
        if "pass" not in result:
            raise ValueError("Guardrail JSON is missing the required 'pass' key.")
        return result

    except Exception as e:
        # Never assume pass if the response cannot be parsed — block the send.
        print(f"  [GUARDRAIL] Parse error: {e}\n  Raw response: {raw}")
        return {"pass": False, "reason": f"Could not parse guardrail response: {raw}"}


# ==============================================================================
# FILE OUTPUT
# ==============================================================================

def save_report(profile: CompanyProfile, report: str) -> str:
    """
    Write the email report to a timestamped .txt file in OUTPUT_DIR.

    The timestamp suffix prevents batch runs from silently overwriting earlier
    reports for the same company name.

    Args:
        profile : Used to derive the sanitized file base name.
        report  : Full report text to persist.

    Returns:
        The path of the saved file.
    """
    # Replace any non-alphanumeric characters (spaces, punctuation) with underscores
    safe_name = re.sub(r"[^\w\-]", "_", profile.company_name).lower()
    timestamp  = datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath   = os.path.join(OUTPUT_DIR, f"{safe_name}_{timestamp}_report.txt")

    with open(filepath, "w", encoding="utf-8") as f:
        f.write(report)

    print(f"  [SAVED] {filepath}")
    return filepath


# ==============================================================================
# SMTP EMAIL DELIVERY
# ==============================================================================

def send_email_smtp(to_address: str, report_text: str, company_name: str) -> None:
    """
    Send the report via SMTP using STARTTLS (port 587).

    Skips silently when SMTP credentials are absent so the script works in
    report-only mode without any mail server configuration.

    The embedded "Subject: ..." line is stripped from the body and used as the
    MIME Subject header so it doesn't appear in the email body itself.

    Args:
        to_address   : Recipient email address.
        report_text  : Full report string including the embedded Subject line.
        company_name : Used to construct the subject line.
    """
    if not SMTP_USER or not SMTP_PASSWORD:
        print("  [SMTP] Credentials not configured — skipping email send.")
        return

    subject = f"AI Automation Opportunities for {company_name}"

    # Separate the embedded Subject header from the deliverable body
    lines = report_text.strip().split("\n")
    if lines[0].startswith("Subject:"):
        body = "\n".join(lines[2:])  # skip the Subject line and the blank line after it
    else:
        body = "\n".join(lines)

    # Assemble the MIME message
    msg = MIMEMultipart()
    msg["From"]    = SMTP_FROM
    msg["To"]      = to_address
    msg["Subject"] = subject
    msg.attach(MIMEText(body, "plain"))

    try:
        with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as server:
            server.ehlo()
            server.starttls()   # Upgrade plain connection to TLS
            server.login(SMTP_USER, SMTP_PASSWORD)
            server.sendmail(SMTP_FROM, to_address, msg.as_string())
        print(f"  [SMTP] Email sent → {to_address}")

    except smtplib.SMTPException as e:
        print(f"  [SMTP ERROR] {e}")


# ==============================================================================
# MAIN PIPELINE ORCHESTRATION
# ==============================================================================

def run_agent(profile: CompanyProfile) -> str:
    """
    Execute the full AI analysis pipeline for one company profile.

    Runs Steps 1–5 in sequence and returns the composed email report.

    Args:
        profile: The company to analyse.

    Returns:
        The complete email report as a plain-text string.
    """
    print(f"  [1/5] Classifying industry ...")
    industry = classify_industry(profile)
    print(f"        → {industry}")

    print(f"  [2/5] Diagnosing pain points ...")
    pain_points = diagnose_pain_points(profile, industry)

    print(f"  [3/5] Recommending AI agents ...")
    agents = recommend_agents(profile, industry, pain_points)

    print(f"  [4/5] Generating ROI summary ...")
    roi_summary = generate_roi_summary(industry, profile.employees)

    diagnosis = DiagnosisResult(
        industry           = industry,
        pain_points        = pain_points,
        recommended_agents = [agents["agents"]],
        roi_summary        = roi_summary,
    )

    print(f"  [5/5] Composing email report ...")
    return generate_email_report(profile, diagnosis)


def process_profile(profile: CompanyProfile) -> None:
    """
    Orchestrate the full end-to-end workflow for one company:
        run_agent → print → save → guardrail → send (if approved).

    Args:
        profile: The company to process.
    """
    print(f"\n{'=' * 60}")
    print(f"  Processing: {profile.company_name}")
    print(f"{'=' * 60}")

    # Run the multi-step AI pipeline
    report = run_agent(profile)

    # Always print to terminal regardless of SMTP / guardrail outcome
    print(f"\n{'-' * 60}\n{report}\n{'-' * 60}")

    # Persist the report so it can be reviewed even if the send is blocked
    save_report(profile, report)

    # Content guardrail — must pass before the email is dispatched
    print("\n  [GUARDRAIL] Scanning report ...")
    result = scan_report_guardrail(report)
    status = "✅ PASS" if result["pass"] else "⛔ FAIL"
    print(f"  [GUARDRAIL] {status} — {result['reason']}")

    if result["pass"]:
        send_email_smtp(profile.email, report, profile.company_name)
    else:
        print("  [GUARDRAIL] Email blocked. Review the saved report and re-run if needed.")


# ==============================================================================
# INPUT HANDLERS
# ==============================================================================

def validate_email(addr: str) -> bool:
    """Return True if addr matches the basic email format pattern."""
    return bool(EMAIL_RE.match(addr))


def collect_from_cli() -> CompanyProfile:
    """
    Interactively collect a single company profile from the terminal.

    Loops on invalid inputs (bad email format, non-integer or zero employee
    count) so mistakes can be corrected without restarting the script.

    Returns:
        A fully populated CompanyProfile.
    """
    print("\n── Enter Company Details ───────────────────────────────────")

    name = input("  Company name:         ").strip()

    # Loop until a syntactically valid email is entered
    while True:
        email = input("  Contact email:        ").strip()
        if validate_email(email):
            break
        print("  ⚠  Invalid email address — please try again.")

    services = input("  Services/description: ").strip()

    # Loop until a positive integer is entered
    while True:
        emp_str = input("  Number of employees:  ").strip()
        try:
            employees = int(emp_str)
            if employees > 0:
                break
            print("  ⚠  Employee count must be greater than 0.")
        except ValueError:
            print("  ⚠  Please enter a whole number.")

    return CompanyProfile(
        company_name = name,
        email        = email,
        services     = services,
        employees    = employees,
    )


def load_from_csv(path: str) -> List[CompanyProfile]:
    """
    Load one or more company profiles from a CSV file.

    Required columns: company_name, email, services, employees

    Rows with invalid emails or non-positive employee counts are skipped with
    a warning rather than halting the entire batch.

    Args:
        path: Filesystem path to the CSV file.

    Returns:
        List of valid CompanyProfile objects.

    Raises:
        FileNotFoundError : If the file does not exist.
        ValueError        : If no valid profiles remain after filtering.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"CSV file not found: {path}")

    profiles = []

    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        for i, row in enumerate(reader, start=2):  # Row 1 is the header row
            email = row["email"].strip()

            if not validate_email(email):
                print(f"  [CSV] Row {i}: skipping — invalid email '{email}'")
                continue

            try:
                employees = int(row["employees"])
                if employees <= 0:
                    raise ValueError
            except ValueError:
                print(f"  [CSV] Row {i}: skipping — invalid employee count '{row['employees']}'")
                continue

            profiles.append(CompanyProfile(
                company_name = row["company_name"].strip(),
                email        = email,
                services     = row["services"].strip(),
                employees    = employees,
            ))

    if not profiles:
        raise ValueError("No valid company profiles found in the CSV file.")

    print(f"[CSV] Loaded {len(profiles)} profile(s) from {path}")
    return profiles


def load_from_json(path: str) -> List[CompanyProfile]:
    """
    Load one or more company profiles from a JSON file.

    Accepts a single JSON object or a list of objects, each with the keys:
    company_name, email, services, employees.

    Invalid entries are skipped with a warning.

    Args:
        path: Filesystem path to the JSON file.

    Returns:
        List of valid CompanyProfile objects.

    Raises:
        FileNotFoundError : If the file does not exist.
        ValueError        : If no valid profiles remain after filtering.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"JSON file not found: {path}")

    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    # Normalise: wrap a single object in a list for uniform processing
    if isinstance(data, dict):
        data = [data]

    profiles = []

    for i, d in enumerate(data):
        email = d.get("email", "").strip()

        if not validate_email(email):
            print(f"  [JSON] Entry {i}: skipping — invalid email '{email}'")
            continue

        try:
            employees = int(d["employees"])
            if employees <= 0:
                raise ValueError
        except (ValueError, KeyError):
            print(f"  [JSON] Entry {i}: skipping — invalid employee count")
            continue

        profiles.append(CompanyProfile(
            company_name = d["company_name"].strip(),
            email        = email,
            services     = d["services"].strip(),
            employees    = employees,
        ))

    if not profiles:
        raise ValueError("No valid company profiles found in the JSON file.")

    print(f"[JSON] Loaded {len(profiles)} profile(s) from {path}")
    return profiles


# ==============================================================================
# ENTRY POINT
# ==============================================================================

def main() -> None:
    """
    Parse command-line arguments, load company profiles, and run the pipeline.

    Supports three mutually exclusive input modes:
        --csv  FILE   Batch process companies from a CSV file
        --json FILE   Batch process companies from a JSON file
        (default)     Collect a single company profile interactively
    """
    parser = argparse.ArgumentParser(
        description="AI Agent Suggestion Agent — Acuff Automation & Analytics",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python agentidea.py                      # interactive CLI\n"
            "  python agentidea.py --csv companies.csv  # batch from CSV\n"
            "  python agentidea.py --json company.json  # batch from JSON\n"
        ),
    )

    group = parser.add_mutually_exclusive_group()
    group.add_argument("--csv",  metavar="FILE", help="Path to CSV file with company profiles")
    group.add_argument("--json", metavar="FILE", help="Path to JSON file with company profile(s)")

    args = parser.parse_args()

    # Route to the correct input loader
    if args.csv:
        profiles = load_from_csv(args.csv)
    elif args.json:
        profiles = load_from_json(args.json)
    else:
        profiles = [collect_from_cli()]

    # Run the full pipeline for every loaded profile
    for profile in profiles:
        process_profile(profile)


if __name__ == "__main__":
    main()
