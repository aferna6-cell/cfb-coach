import os

# Prep must not download research/*.json from GitHub during tests; tests that cover the
# daily AI research call cfb_coach.ai_research directly with a fake fetch.
os.environ.setdefault("CFB_COACH_AI_RESEARCH", "off")
