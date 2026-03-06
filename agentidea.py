"""
AI Agent Suggestion Agent
Author: Acuff Automation & Analytics

This agent:
1. Accepts company data
2. Classifies the business industry
3. Diagnoses likely operational pain points
4. Suggests AI agents to improve ROI
5. Generates an email report
"""

import os
from dataclasses import dataclass
from typing import List, Dict
from openai import OpenAI


# -------------------------------
# Configuration
# -------------------------------

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


# -------------------------------
# Data Model
# -------------------------------

@dataclass
class CompanyProfile:
    company_name: str
    email: str
    services: str
    employees: int


@dataclass
class DiagnosisResult:
    industry: str
    pain_points: List[str]
    recommended_agents: List[str]
    roi_summary: str


# -------------------------------
# Industry Classification
# -------------------------------

def classify_industry(profile: CompanyProfile) -> str:

    prompt = f"""
    Classify the industry of the following company.

    Company: {profile.company_name}
    Services: {profile.services}
    Employees: {profile.employees}

    Choose the closest industry from:
    retail
    manufacturing
    nonprofit
    medical
    enterprise services
    technology
    logistics
    finance

    Return only the industry name.
    """

    response = client.chat.completions.create(
        model="gpt-4.1",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2
    )

    return response.choices[0].message.content.strip()


# -------------------------------
# Pain Point Diagnosis
# -------------------------------

def diagnose_pain_points(profile: CompanyProfile, industry: str) -> List[str]:

    prompt = f"""
    A company operates in the {industry} industry.

    Company description:
    {profile.services}

    Employee count: {profile.employees}

    List the 5 most common operational pain points this type of
    company typically experiences.

    Output as a simple numbered list.
    """

    response = client.chat.completions.create(
        model="gpt-4.1",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3
    )

    text = response.choices[0].message.content
    return [line.strip() for line in text.split("\n") if line.strip()]


# -------------------------------
# AI Agent Recommendation Engine
# -------------------------------

def recommend_agents(profile: CompanyProfile, industry: str, pain_points: List[str]) -> Dict:

    prompt = f"""
    A company in the {industry} industry has the following pain points:

    {pain_points}

    Company details:
    - Name: {profile.company_name}
    - Services: {profile.services}
    - Employees: {profile.employees}

    Recommend 4 AI autonomous agents that would solve these problems.

    For each agent provide:
    - Agent name
    - Description
    - Estimated ROI benefit

    Format:
    Agent Name:
    Description:
    ROI:
    """

    response = client.chat.completions.create(
        model="gpt-4.1",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.4
    )

    return {
        "agents": response.choices[0].message.content
    }


# -------------------------------
# ROI Summary
# -------------------------------

def generate_roi_summary(industry: str, employees: int) -> str:

    prompt = f"""
    Estimate the business ROI impact of implementing AI automation
    in a {industry} company with approximately {employees} employees.

    Provide a concise 3 sentence business summary.
    """

    response = client.chat.completions.create(
        model="gpt-4.1",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3
    )

    return response.choices[0].message.content


# -------------------------------
# Email Report Generator
# -------------------------------

def generate_email_report(profile: CompanyProfile, diagnosis: DiagnosisResult) -> str:

    email_report = f"""
Subject: AI Automation Opportunities for {profile.company_name}

Hello,

Thank you for your interest in AI automation solutions.

Based on our analysis of your company information, we identified the
following opportunities where autonomous AI agents could create
operational efficiencies and measurable ROI.

Company Overview
----------------
Company: {profile.company_name}
Employees: {profile.employees}
Industry Classification: {diagnosis.industry}

Common Operational Challenges
------------------------------
"""

    for p in diagnosis.pain_points:
        email_report += f"- {p}\n"

    email_report += f"""

Recommended AI Agents
---------------------

{chr(10).join(diagnosis.recommended_agents)}

Estimated ROI Impact
--------------------

{diagnosis.roi_summary}

Next Steps
----------

If you'd like, we can schedule a consultation to discuss building
custom AI agents tailored to your operations.

Best regards,

Acuff Automation & Analytics
AI Agentic Development
"""

    return email_report


# -------------------------------
# Agent Workflow
# -------------------------------

def run_agent(profile: CompanyProfile):

    industry = classify_industry(profile)

    pain_points = diagnose_pain_points(profile, industry)

    agents = recommend_agents(profile, industry, pain_points)

    roi_summary = generate_roi_summary(industry, profile.employees)

    diagnosis = DiagnosisResult(
        industry=industry,
        pain_points=pain_points,
        recommended_agents=[agents["agents"]],
        roi_summary=roi_summary
    )

    email = generate_email_report(profile, diagnosis)

    return email


# -------------------------------
# Example Execution
# -------------------------------

if __name__ == "__main__":

    company = CompanyProfile(
        company_name="Example Retail Co",
        email="owner@example.com",
        services="Local retail store selling home goods and decor",
        employees=25
    )

    report = run_agent(company)

    print(report)