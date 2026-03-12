import json
import requests
from transformers import pipeline

# 1. Initialize the Sieve (Runs on CPU/RAM)
print("🚀 Loading FinBERT Sieve...")
sieve = pipeline("sentiment-analysis", model="ProsusAI/finbert")

def run_integrated_analysis(headline):
    # 2. Step 1: Sentiment Check (The Sieve)
    result = sieve(headline)[0]
    label = result['label']
    score = result['score']

    print(f"\n📰 Headline: {headline}")
    print(f"📊 Sieve Result: {label} ({score:.2f})")

    # HARDENED LOGIC: Only proceed if it's a strong market move
    if label == "neutral" or score < 0.75:
        print(f"⏭️ Skipping... Signal too weak ({score:.2f}) or Neutral.")
        return

    # 3. Step 2: Deep Reasoning (Optimized 8B Model)
    print("🧠 High-Alpha detected. Triggering 8B Reasoning Engine...")
    
    url = "http://localhost:11434/api/generate"
    try:
        with open('market_knowledge.json', 'r') as f:
            knowledge = json.load(f)
    except FileNotFoundError:
        print("❌ Error: market_knowledge.json not found!")
        return

    prompt = f"""
    ### KNOWLEDGE BASE
    {json.dumps(knowledge)}

    ### NEWS
    "{headline}"

    ### TASK
    Perform a 2nd-order analysis. Identify the winner from the KNOWLEDGE BASE.
    Output Format:
    - Ticker: 
    - Signal: (LONG/SHORT)
    - Confidence: (0-100%)
    - Rationale: (Max 2 sentences)
    """

    # Switch to 8B for VRAM efficiency
    payload = {"model": "deepseek-r1:8b", "prompt": prompt, "stream": False}
    
    try:
        response = requests.post(url, json=payload, timeout=60)
        print(f"\n✅ SIGNAL DETECTED:\n{response.json()['response']}")
    except Exception as e:
        print(f"❌ Ollama Error: {e}")

# --- TEST SUITE ---
if __name__ == "__main__":
    run_integrated_analysis("US grants Chevron expanded license for Venezuelan heavy crude") # Trigger
    run_integrated_analysis("Oil prices flat in quiet Tuesday trading") # Skip
    run_integrated_analysis("New sanctions hit Iranian tanker fleet amid rising tensions") # Trigger