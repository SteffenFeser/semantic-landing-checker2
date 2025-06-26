import os
from flask import Flask, render_template, request
import pandas as pd
import requests
from bs4 import BeautifulSoup
from sklearn.metrics.pairwise import cosine_similarity
import openai
from tqdm import tqdm

app = Flask(__name__)
openai.api_key = os.getenv("OPENAI_API_KEY")

def get_embedding(text):
    response = openai.Embedding.create(
        input=text,
        model="text-embedding-3-small"
    )
    return response["data"][0]["embedding"]

def extract_text_from_url(url):
    try:
        r = requests.get(url, timeout=5)
        soup = BeautifulSoup(r.text, 'html.parser')
        return soup.get_text(separator=' ', strip=True)
    except:
        return ""

def evaluate_similarity(adtext, keyword, pagetext):
    input_1 = adtext + " " + keyword
    input_2 = pagetext
    emb1 = get_embedding(input_1)
    emb2 = get_embedding(input_2)
    score = cosine_similarity([emb1], [emb2])[0][0]
    return round(score, 2)

def get_feedback(adtext, keyword, pagetext):
    prompt = f"""
Du bist Conversion-Optimierer. Zwischen dem Anzeigen-Keyword '{keyword}', dem AdText '{adtext}' und der Landing Page besteht folgender Text:

--- Start Landing Page ---
{pagetext[:2000]}
--- Ende Landing Page ---

Was fehlt auf der Seite? Was kann verbessert werden, um die Anzeigen-Relevanz zu steigern?
Antwort in max. 2 Sätzen.
"""
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.6,
        max_tokens=100
    )
    return response['choices'][0]['message']['content']

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return "Keine Datei gefunden"
    
    file = request.files['file']
    df = pd.read_csv(file)

    results = []
    for _, row in tqdm(df.iterrows(), total=len(df)):
        adtext = str(row['AdText'])
        keyword = str(row['Keyword'])
        url = str(row['FinalURL'])

        pagetext = extract_text_from_url(url)
        if not pagetext:
            results.append({
                'Keyword': keyword,
                'AdText': adtext,
                'FinalURL': url,
                'Score': "Fehler",
                'Bewertung': "Seite nicht erreichbar",
                'Suggestion': "-"
            })
            continue

        score = evaluate_similarity(adtext, keyword, pagetext)
        bewertung = "🔴 Niedrig" if score < 0.65 else ("🟡 Mittel" if score < 0.85 else "🟢 Hoch")
        feedback = get_feedback(adtext, keyword, pagetext)

        results.append({
            'Keyword': keyword,
            'AdText': adtext,
            'FinalURL': url,
            'Score': score,
            'Bewertung': bewertung,
            'Suggestion': feedback
        })

    return render_template('index.html', results=results)

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
