from flask import Flask, render_template, request
import pandas as pd
import os

app = Flask(__name__)

def dummy_semantic_similarity(ad_text, landingpage):
    # Platzhalter: gibt Dummy-Score zurück
    return round(len(set(ad_text.lower().split()) & set(landingpage.lower().split())) / max(1, len(set(ad_text.lower().split()))), 2)

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    file = request.files['file']
    if not file:
        return "No file uploaded.", 400
    
    df = pd.read_csv(file)

    # Erwartete Spalten
    expected = ['Keyword', 'Ad Text', 'Landing Page URL']
    for col in expected:
        if col not in df.columns:
            return f"Fehlende Spalte: {col}", 400

    # Semantische Bewertung ergänzen
    df['Semantic Score'] = df.apply(
        lambda row: dummy_semantic_similarity(str(row['Ad Text']), str(row['Landing Page URL'])), axis=1
    )

    # Weitergabe an HTML
    return render_template('index.html', tables=[df.to_html(classes='data', header="true", index=False)])

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
