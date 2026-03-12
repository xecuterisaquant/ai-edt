import json
import requests
from transformers import pipeline

# 1. Initialize the Sieve (Runs on CPU/RAM - 110M parameters)
print("🚀 Loading FinBERT Sieve...")
sieve = pipeline("sentiment-analysis", model="ProsusAI/finbert")

def run_integrated_analysis(headline):
    # 2. Step 1: Sentiment Check (The Sieve)
    result = sieve(headline)[0]
    label = result['label']
    score = result['score']

    print(f"\n📰 Headline: {headline}")
    print(f"📊 Sieve Result: {label} ({score:.2f})")

    # If the news is neutral, don't waste 14B reasoning time
    if label == "neutral" and score > 0.5:
        print("⏭️ Skipping... No significant market-moving sentiment detected.")
        return

    # 3. Step 2: Deep Reasoning (The 14B Model)
    print("🧠 Sentiment confirmed. Triggering 14B Reasoning Engine...")
    
    url = "http://localhost:11434/api/generate"
    with open('market_knowledge.json', 'r') as f:
        knowledge = json.load(f)

    prompt = f"""
    ### KNOWLEDGE BASE
    {json.dumps(knowledge)}

    ### NEWS
    "{headline}"

    ### TASK
    Based on the {label} sentiment, find the 2nd-order winner. 
    Explain the logistical or technical link.
    """

    payload = {"model": "deepseek-r1:14b", "prompt": prompt, "stream": False}
    response = requests.post(url, json=payload).json()
    print(f"\n✅ SIGNAL DETECTED:\n{response['response']}")

# --- TEST IT ---
run_integrated_analysis("Saudi Arabia maintains steady oil production for April") # Should skip
run_integrated_analysis("US imposes new sanctions on Iranian tanker fleet") # Should trigger