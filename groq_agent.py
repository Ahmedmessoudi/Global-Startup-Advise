# groq_agent.py — Groq LLM Integration for Intent Parsing and Result Synthesis

import json
import logging
import re

from groq import Groq

from config import GROQ_MODEL, GROQ_MAX_TOKENS, SECTORS

logger = logging.getLogger(__name__)


def _get_client(api_key: str) -> Groq:
    return Groq(api_key=api_key)


# ─────────────────────────────────────────────────────────────────────────────
# INTENT PARSING
# ─────────────────────────────────────────────────────────────────────────────

PARSE_SYSTEM_PROMPT = f"""You are a startup advisor assistant. Your job is to extract structured information from the user's message.

Extract:
1. startup_description: A brief description of the startup idea (1-2 sentences)
2. target_country: The country name they want to launch in (normalize to standard English name, e.g. "United States" not "USA")
3. sector: The startup sector — must be one of: {', '.join(SECTORS)}
   - Choose the closest match based on the startup description
   - Default to "general" if unclear

Respond ONLY with valid JSON, no markdown, no extra text:
{{"startup_description": "...", "target_country": "...", "sector": "..."}}
"""


def parse_intent(user_message: str, api_key: str) -> dict:
    """
    Use Groq LLM to extract startup description, target country, and sector
    from a natural language user message.
    Returns a dict with keys: startup_description, target_country, sector.
    """
    client = _get_client(api_key)

    try:
        response = client.chat.completions.create(
            model=GROQ_MODEL,
            max_tokens=300,
            messages=[
                {"role": "system", "content": PARSE_SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
        )
        raw = response.choices[0].message.content.strip()

        # Robust JSON extraction to handle reasoning models that output extra text
        match = re.search(r'\{.*\}', raw, re.DOTALL)
        if match:
            raw = match.group(0)

        # Strip markdown fences if any
        raw = re.sub(r"```(?:json)?", "", raw).strip("`").strip()

        parsed = json.loads(raw)
        parsed["sector"] = parsed.get("sector", "general").lower()
        if parsed["sector"] not in SECTORS:
            parsed["sector"] = "general"

        logger.info(f"[PARSE] Intent: {parsed}")
        return parsed

    except json.JSONDecodeError as e:
        logger.error(f"[PARSE] JSON parse error: {e}")
        return {"startup_description": user_message, "target_country": "Unknown", "sector": "general"}
    except Exception as e:
        logger.error(f"[PARSE] Groq error: {e}")
        raise


# ─────────────────────────────────────────────────────────────────────────────
# RESULT SYNTHESIS
# ─────────────────────────────────────────────────────────────────────────────

SYNTHESIS_SYSTEM_PROMPT = """You are an expert startup advisor and data analyst with deep knowledge of global markets.

You have access to CIA World Factbook data covering {n_countries} countries. 
A user wants to launch a **{sector}** startup in **{country}**.

CRITICAL: Base your analysis ONLY on the structured data provided. Do not add information not present in the data.

Write a structured analysis using EXACTLY this format:

## 🎯 Opportunity Assessment for {country}
2-3 sentences. Reference the opportunity score, rank, and cluster. Be specific with numbers.

## ✅ Key Opportunities  
- [Bullet 1: based on strongest features from data]
- [Bullet 2]
- [Bullet 3]

## ⚠️ Key Risks  
- [Bullet 1: based on weakest features from data]
- [Bullet 2]
- [Bullet 3]

## 🌍 Recommended Alternative Countries
For each of the top 3 alternatives, explain in 1 sentence WHY it scores higher for this sector.

## 📊 Verdict
One of: **🟢 Recommended** | **🟡 Proceed with Caution** | **🔴 Not Recommended**
Followed by 1 sentence of reasoning.

Use the data. Be precise. Be actionable.
"""


def synthesize_results(context: dict, api_key: str) -> str:
    """
    Use Groq LLM to generate a natural language analysis from the data mining results.
    context: dict from results_engine.build_llm_context()
    Returns formatted markdown text.
    """
    if "error" in context:
        return f"❌ Error: {context['error']}"

    client = _get_client(api_key)

    system = SYNTHESIS_SYSTEM_PROMPT.format(
        n_countries=context.get("total_countries", "N"),
        sector=context.get("sector", "general"),
        country=context.get("target_country", "the target country"),
    )

    user_content = f"""
**Startup:** {context.get('startup_description', 'N/A')}
**Target Country:** {context.get('target_country')}
**Sector:** {context.get('sector')}
**Opportunity Score:** {context.get('opportunity_score')}/100
**Global Rank:** #{context.get('global_rank')} out of {context.get('total_countries')} countries
**Cluster:** {context.get('cluster')}

**Feature Analysis (normalized 0-1, with percentile):**
{json.dumps(context.get('feature_analysis', {}), indent=2)}

**Raw Economic Indicators:**
{json.dumps(context.get('raw_indicators', {}), indent=2)}

**Top Alternative Countries:**
{json.dumps(context.get('top_alternative_countries', []), indent=2)}

**Similar Countries in Same Cluster:**
{json.dumps(context.get('similar_cluster_countries', []), indent=2)}
"""

    try:
        response = client.chat.completions.create(
            model=GROQ_MODEL,
            max_tokens=GROQ_MAX_TOKENS,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user_content},
            ],
        )
        return response.choices[0].message.content.strip()

    except Exception as e:
        logger.error(f"[SYNTHESIS] Groq error: {e}")
        return f"❌ Error generating analysis: {str(e)}"


# ─────────────────────────────────────────────────────────────────────────────
# FOLLOW-UP CHAT
# ─────────────────────────────────────────────────────────────────────────────

FOLLOWUP_SYSTEM_PROMPT = """You are an expert startup advisor. You have already run a data mining analysis on the CIA World Factbook dataset.

The following context contains all the data you have available. Answer the user's follow-up question based ONLY on this data. If the answer is not in the data, say so clearly.

DATASET CONTEXT:
{context_json}

Guidelines:
- Be specific and cite numbers from the data
- If comparing countries, reference their scores and ranks
- Keep responses concise (3-5 sentences max unless a detailed comparison is requested)
- Do not invent data not in the context
"""


def followup_chat(
    user_message: str,
    chat_history: list,
    context: dict,
    api_key: str,
) -> str:
    """
    Handle follow-up questions in the chat.
    chat_history: list of {role: "user"|"assistant", content: str}
    context: the LLM context dict from the last analysis
    """
    client = _get_client(api_key)

    system = FOLLOWUP_SYSTEM_PROMPT.format(
        context_json=json.dumps(context, indent=2)
    )

    messages = [{"role": "system", "content": system}]
    # Include last 6 messages for context (avoid token overflow)
    messages.extend(chat_history[-6:])
    messages.append({"role": "user", "content": user_message})

    try:
        response = client.chat.completions.create(
            model=GROQ_MODEL,
            max_tokens=GROQ_MAX_TOKENS,
            messages=messages,
        )
        return response.choices[0].message.content.strip()

    except Exception as e:
        logger.error(f"[FOLLOWUP] Groq error: {e}")
        return f"❌ Error: {str(e)}"
