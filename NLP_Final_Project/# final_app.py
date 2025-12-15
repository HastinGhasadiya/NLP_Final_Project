# ============================================================
# FINAL APP — AI CONTENT FORENSICS (OPTION 3)
# ============================================================
# Uses:
#  - Sentiment (Neural MLP ONLY)
#  - Formality (LogReg)
#  - AI vs Human (LogReg)
#  - OpenAI GPT for explanation + rewrite + attack (OPTIONAL)
# ============================================================

import os
import json
import joblib
import numpy as np

from openai import OpenAI
from tensorflow.keras.models import load_model

# ============================================================
# 🔑 OPENAI CONFIG (SAFE FOR GITHUB)
# ============================================================
# Option A (recommended): environment variable OPENAI_API_KEY
# Option B (optional): paste key below for local testing ONLY
# ------------------------------------------------------------

PASTED_API_KEY = ""   # <-- LEAVE EMPTY for GitHub

if PASTED_API_KEY.strip():
    client = OpenAI(api_key=PASTED_API_KEY)
    USE_GPT = True
elif os.getenv("OPENAI_API_KEY"):
    client = OpenAI()  # reads from environment
    USE_GPT = True
else:
    client = None
    USE_GPT = False

# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ARTIFACT_DIR = os.path.join(BASE_DIR, "artifacts")

# ============================================================
# LOAD HELPERS
# ============================================================

def load_joblib(name):
    path = os.path.join(ARTIFACT_DIR, name)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing artifact: {path}")
    return joblib.load(path)

# ============================================================
# LOAD MODELS (NO SENTIMENT LOGREG)
# ============================================================

def load_models():
    formality_logreg, formality_vec = load_joblib("formality_logreg.joblib")
    ai_logreg, ai_vec = load_joblib("ai_vs_human_logreg.joblib")

    sentiment_mlp = load_model(os.path.join(ARTIFACT_DIR, "sentiment_mlp.h5"))
    sentiment_mlp_vec = load_joblib("sentiment_mlp_vectorizer.joblib")

    return {
        "sentiment_mlp": sentiment_mlp,
        "sentiment_mlp_vec": sentiment_mlp_vec,
        "formality_logreg": formality_logreg,
        "formality_vec": formality_vec,
        "ai_logreg": ai_logreg,
        "ai_vec": ai_vec
    }

# ============================================================
# CLASSIFIER HELPERS
# ============================================================

def predict_sklearn(text, clf, vec, labels):
    X = vec.transform([text])
    probs = clf.predict_proba(X)[0]
    idx = int(np.argmax(probs))
    return {
        "prediction": labels[idx],
        "proba": {labels[i]: float(probs[i]) for i in range(len(labels))}
    }

# ============================================================
# ANALYSIS
# ============================================================

def analyze_text(text, models):
    result = {}

    result["formality"] = predict_sklearn(
        text,
        models["formality_logreg"],
        models["formality_vec"],
        ["informal", "formal"]
    )

    result["ai_vs_human"] = predict_sklearn(
        text,
        models["ai_logreg"],
        models["ai_vec"],
        ["human", "ai"]
    )

    X_nn = models["sentiment_mlp_vec"].transform([text]).toarray()
    p_pos = float(models["sentiment_mlp"].predict(X_nn, verbose=0)[0][0])

    result["sentiment"] = {
        "prediction": "positive" if p_pos >= 0.5 else "negative",
        "proba": {
            "negative": 1 - p_pos,
            "positive": p_pos
        }
    }

    return result

# ============================================================
# GPT HELPERS (SAFE)
# ============================================================

def gpt(system, user):
    resp = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user}
        ],
        temperature=0.7
    )
    return resp.choices[0].message.content.strip()

def gpt_explain(text, analysis):
    return gpt(
        "You are an AI content forensics expert.",
        f"""
TEXT:
{text}

CLASSIFIER OUTPUT:
{json.dumps(analysis, indent=2)}

Explain:
- Why it looks AI or human
- Why sentiment + formality were predicted
- Mention style, structure, vocabulary
(≤200 words)
"""
    )

def gpt_rewrite_human(text, analysis):
    return gpt(
        "You rewrite AI text to look human.",
        f"""
Rewrite the following to sound MORE HUMAN,
less generic, slightly more natural.

TEXT:
{text}

Classifier context:
{json.dumps(analysis, indent=2)}

Return ONLY rewritten text.
"""
    )

def gpt_attack(text, analysis):
    return gpt(
        "You attempt to fool an AI detector.",
        f"""
Rewrite the text so classifiers think it is HUMAN.
Keep meaning, change style subtly.

TEXT:
{text}

Classifier context:
{json.dumps(analysis, indent=2)}

Return ONLY rewritten text.
"""
    )

# ============================================================
# DISPLAY
# ============================================================

def print_analysis(analysis):
    for k, v in analysis.items():
        print(f"\n{k.upper()}:")
        print(f"  Prediction: {v['prediction']}")
        for lbl, p in v["proba"].items():
            print(f"  P({lbl}) = {p:.3f}")

# ============================================================
# MAIN LOOP
# ============================================================

def main():
    print("=" * 60)
    print("AI CONTENT FORENSICS — OPTION 3")
    print("Type 'q' to quit")
    print("=" * 60)

    models = load_models()

    while True:
        text = input("\nEnter text:\n> ").strip()
        if text.lower() == "q":
            break

        analysis = analyze_text(text, models)

        print("\n=== CLASSIFIER RESULTS ===")
        print_analysis(analysis)

        print("\n=== GPT EXPLANATION ===")
        if USE_GPT:
            print(gpt_explain(text, analysis))
        else:
            print("GPT disabled (no API key set).")

        if input("\nRewrite to look HUMAN? (y/n): ").lower() == "y":
            if USE_GPT:
                rewritten = gpt_rewrite_human(text, analysis)
                print("\n--- REWRITTEN TEXT ---")
                print(rewritten)
                print("\nRe-analysis:")
                print_analysis(analyze_text(rewritten, models))
            else:
                print("GPT disabled (no API key set).")

        if input("\nAdversarial attack (fool detector)? (y/n): ").lower() == "y":
            if USE_GPT:
                adv = gpt_attack(text, analysis)
                print("\n--- ADVERSARIAL TEXT ---")
                print(adv)
                print("\nRe-analysis:")
                print_analysis(analyze_text(adv, models))
            else:
                print("GPT disabled (no API key set).")

    print("Done.")

# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()
