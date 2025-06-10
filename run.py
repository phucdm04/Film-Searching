from flask import Flask, request, jsonify, render_template
from pyngrok import ngrok
from pymongo import MongoClient
import os
from dotenv import load_dotenv
from search_engine.search_logic import search_query
from embedding.ppmi import PPMIEmbedder, TruncatedSVD 


load_dotenv()

app = Flask(__name__)
ngrok.set_auth_token(os.getenv("NGROK_TOKEN"))


# Kết nối MongoDB
uri = os.getenv("MONGO_URI")
client = MongoClient(uri, tls=True, tlsAllowInvalidCertificates=True)
db = client["Film"]
collection = db["Data"]


@app.route("/")
def home():
    return render_template("index.html")

@app.route("/about")
def about():
    return render_template("about.html")

@app.route("/search", methods=["POST"])
def search():
    data = request.get_json()
    query = data.get("query")
    model = data.get("model")
    hits = search_query(query, model)

    # query = data.get("query")
    # hits = search_query(query)
    # Chuyển đổi từng ScoredPoint thành dict
    results = [
        {
            "id": hit.id,
            "score": hit.score,
            "payload": hit.payload,
        }
        for hit in hits
    ]

    return jsonify(results)

if __name__ == "__main__":
    # Mở cổng 5000 bằng ngrok
    public_url = ngrok.connect(5000)
    print(" * Ngrok public URL:", public_url)
    
    app.run(port=5000)
