import os
import sys
import json

import google.generativeai as genai
from dotenv import load_dotenv


PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from ai.build_fact_package import build_fact_package


load_dotenv()


MODEL_NAME = "models/gemini-2.5-flash"


GROUNDING_RULES = """
You are the explanation layer for a YouTube trend analytics system.

The Python/data layer has already calculated every metric supplied to you.
Your job is to interpret those facts clearly.

STRICT GROUNDING RULES:

1. Use ONLY facts contained in the supplied FACT PACKAGE.

2. Never invent a number, rank, percentage, category, region,
   channel statistic, velocity, lifecycle value, or prediction.

3. Do not recalculate or modify supplied metrics.

4. If the fact package does not contain enough evidence for a claim,
   explicitly say that the available data is insufficient.

5. Clearly distinguish:
   - observed facts
   - calculated historical metrics
   - model predictions
   - your interpretation

6. A model prediction is NOT an observed future outcome.
   Use wording such as:
   "the model estimates..."
   or
   "the model assigns a probability of..."

7. Do not claim causation.
   Correlation, rank movement, engagement, velocity, or channel size
   does not prove why a video is trending.

8. Do not use outside knowledge about creators, videos, events,
   countries, or YouTube to fill gaps.

9. Do not call something viral, successful, failing, or globally
   popular unless the supplied metrics support that description.
   Prefer precise descriptions such as:
   "rank increased sharply"
   or
   "view velocity is high relative to the supplied comparison."

10. Treat "cross_region" as meaning the video appeared in at least
    two regions collected by this pipeline. It does not mean worldwide.

11. Treat "region_specific" as meaning the video appeared in only one
    of the regions collected by this pipeline. It does not prove that
    the video was absent everywhere else in the world.

12. Do not describe historical videos as currently rising unless the
    supplied facts identify them as current observations.

13. When discussing engagement_rate, remember that the supplied value
    is a calculated ratio, not a causal measure of audience quality.

14. Keep explanations concise, analytical, and understandable.

15. When useful, quote supplied values to justify an interpretation.
"""


def configure_gemini():
    """
    Configure Gemini using the existing project environment variable.
    """

    api_key = os.getenv("GOOGLE_API_KEY")

    if not api_key:
        raise ValueError(
            "GOOGLE_API_KEY was not found in the environment."
        )

    genai.configure(api_key=api_key)

    return genai.GenerativeModel(MODEL_NAME)


def package_to_json(fact_package):
    """
    Serialize Python-calculated facts for Gemini.
    """

    return json.dumps(
        fact_package,
        indent=2,
        ensure_ascii=False,
    )


def ask_grounded_gemini(
    question,
    fact_package=None,
):
    """
    Ask Gemini to interpret the supplied Python-calculated facts.

    Gemini receives structured facts and grounding rules.
    It does not receive responsibility for calculating metrics.
    """

    if fact_package is None:
        fact_package = build_fact_package()

    model = configure_gemini()

    facts_json = package_to_json(fact_package)

    prompt = f"""
{GROUNDING_RULES}

FACT PACKAGE
============
{facts_json}

USER ANALYTICS QUESTION
=======================
{question}

RESPONSE REQUIREMENTS
=====================

Answer the question using only the FACT PACKAGE.

For every important conclusion:
- identify the supporting metric or observation;
- distinguish prediction from historical observation;
- avoid unsupported causal explanations.

If the requested conclusion cannot be supported by the supplied facts,
say that clearly rather than guessing.
"""

    response = model.generate_content(prompt)

    if not response.text:
        raise RuntimeError(
            "Gemini returned an empty response."
        )

    return response.text.strip()


def generate_trend_summary(fact_package=None):
    """
    Generate a grounded overall trend summary.
    """

    question = """
Create a concise trend intelligence summary.

Discuss:
- notable currently rising videos,
- important channel patterns,
- important category patterns,
- regional differences,
- cross-region versus region-specific patterns,
- prediction results where relevant.

Do not simply list every value.
Explain the strongest patterns supported by the facts.
"""

    return ask_grounded_gemini(
        question,
        fact_package,
    )


def explain_rising_videos(fact_package=None):
    """
    Explain unusual/currently rising videos using supplied facts.
    """

    question = """
Explain the most notable currently rising videos in the supplied data.

Use rank movement, view velocity, engagement metrics,
video lifecycle information, and prediction probabilities where
available.

Explain why each video is notable according to the supplied metrics.

Do not invent reasons for why the audience is responding to the video.
"""

    return ask_grounded_gemini(
        question,
        fact_package,
    )


def explain_regional_differences(fact_package=None):
    """
    Explain differences between collected regions.
    """

    question = """
Compare the regions represented in the fact package.

Explain meaningful differences in:
- trending-video counts,
- channel diversity,
- average rank,
- view velocity,
- engagement,
- category patterns,
- cross-region versus region-specific behavior.

Only describe differences supported by supplied values.
Do not speculate about cultural or demographic causes.
"""

    return ask_grounded_gemini(
        question,
        fact_package,
    )


if __name__ == "__main__":

    facts = build_fact_package()

    print(
        "\n"
        "============================================\n"
        "GROUNDED GEMINI TREND SUMMARY\n"
        "============================================\n"
    )

    summary = generate_trend_summary(facts)

    print(summary)