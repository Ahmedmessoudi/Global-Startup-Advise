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
        # Find the first '{' and the last '}'
        start_idx = raw.find('{')
        end_idx = raw.rfind('}')
        
        if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
            raw = raw[start_idx:end_idx+1]
        
        # Clean any remaining markdown backticks if they are somehow still there
        raw = raw.strip()

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

SYNTHESIS_SYSTEM_PROMPT = """Vous êtes un conseiller expert en startups et un analyste de données avec une connaissance approfondie des marchés mondiaux.

Vous avez accès aux données du CIA World Factbook couvrant {n_countries} pays.
Un utilisateur souhaite lancer une startup dans le secteur **{sector}** en **{country}**.

CRITIQUE : Basez votre analyse UNIQUEMENT sur les données structurées fournies. N'ajoutez pas d'informations non présentes dans les données.

Rédigez une analyse structurée (en Français) en utilisant EXACTEMENT ce format :

## Évaluation des Opportunités pour {country}
2-3 phrases. Faites référence au score d'opportunité, au rang et au cluster. Soyez précis avec les chiffres.

## Opportunités Clés
- [Point 1 : basé sur les caractéristiques les plus fortes des données]
- [Point 2]
- [Point 3]

## Risques Clés
- [Point 1 : basé sur les caractéristiques les plus faibles des données]
- [Point 2]
- [Point 3]

## Pays Alternatifs Recommandés
Pour chacune des 3 meilleures alternatives, expliquez en 1 phrase POURQUOI elle a un score plus élevé pour ce secteur.

## Verdict
Un parmi : **🟢 Recommandé** | **🟡 Procéder avec Prudence** | **🔴 Non Recommandé**
Suivi d'une phrase de justification.

Utilisez les données. Soyez précis. Soyez actionnable.
**RÈGLE DE FORMATAGE** : Aérez votre texte ! Sautez une ligne (retour à la ligne) après chaque ou toutes les deux phrases pour une meilleure lisibilité. Mettez toujours les pourcentages (ex: **45%**) en gras.
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

FOLLOWUP_SYSTEM_PROMPT = """Vous êtes un conseiller expert en startups. Vous avez déjà effectué une analyse d'exploration de données sur l'ensemble de données du CIA World Factbook.

Le contexte suivant contient toutes les données dont vous disposez. Répondez à la question de suivi de l'utilisateur en vous basant UNIQUEMENT sur ces données. Si la réponse n'est pas dans les données, dites-le clairement.

CONTEXTE DES DONNÉES :
{context_json}

Lignes directrices :
- Répondez toujours en Français.
- Soyez précis et citez des chiffres à partir des données.
- Si vous comparez des pays, faites référence à leurs scores et classements.
- Gardez des réponses concises (3-5 phrases maximum, à moins qu'une comparaison détaillée ne soit demandée).
- N'inventez pas de données non présentes dans le contexte.
- **RÈGLE DE FORMATAGE** : Aérez votre texte ! Sautez une ligne (retour à la ligne) après chaque ou toutes les deux phrases. Mettez toujours les pourcentages (ex: **45%**) en gras.
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
