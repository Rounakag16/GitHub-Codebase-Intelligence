"""
Phase 0, step 3: sanity-check that all three API keys actually work.
Run this after filling in your .env — don't move to Phase 1 until all three pass.
"""

import os
from dotenv import load_dotenv

load_dotenv()


def check_gemini():
    from google import genai

    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    resp = client.models.generate_content(
        model="gemini-flash-latest",
        contents="Say 'ok' and nothing else.",
    )
    print("Gemini:", resp.text.strip())

def check_github():
    from github import Github

    gh = Github(os.environ["GITHUB_TOKEN"])
    user = gh.get_user()
    print("GitHub: authenticated as", user.login)


def check_astra():
    from astrapy import DataAPIClient

    client = DataAPIClient(os.environ["ASTRA_DB_APPLICATION_TOKEN"])
    db = client.get_database(os.environ["ASTRA_DB_API_ENDPOINT"])
    print("Astra DB: connected, collections =", db.list_collection_names())


if __name__ == "__main__":
    for name, fn in [
        ("Gemini", check_gemini),
        ("GitHub", check_github),
        ("Astra DB", check_astra),
    ]:
        try:
            fn()
        except Exception as e:
            print(f"{name}: FAILED — {e}")