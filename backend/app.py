import os
import re
import pickle
import pandas as pd
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# ===== PATH SETUP =====
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
APP_DIR = os.path.join(BASE_DIR, "app")

TEXT_MODEL_PATH = os.path.join(BASE_DIR, "models", "phishing_model.pkl")
TEXT_VECTORIZER_PATH = os.path.join(BASE_DIR, "models", "vectorizer.pkl")
URL_MODEL_PATH = os.path.join(BASE_DIR, "models", "url_model.pkl")
URL_FEATURES_PATH = os.path.join(BASE_DIR, "models", "url_feature_names.pkl")

# ===== REQUIRED FOR PICKLE LOADING =====
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"[^\u0980-\u09FFa-zA-Z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def custom_tokenizer(text):
    text = clean_text(text)
    return re.findall(r'[\u0980-\u09FF]+|[a-zA-Z]+|\d+', text)

# ===== LOAD MODELS =====
with open(TEXT_MODEL_PATH, "rb") as f:
    text_model = pickle.load(f)

with open(TEXT_VECTORIZER_PATH, "rb") as f:
    text_vectorizer = pickle.load(f)

with open(URL_MODEL_PATH, "rb") as f:
    url_model = pickle.load(f)

with open(URL_FEATURES_PATH, "rb") as f:
    url_feature_names = pickle.load(f)

# ===== TEXT MODEL PREDICTION =====
def predict_text_model(text):
    cleaned = clean_text(text)
    vec = text_vectorizer.transform([cleaned])
    pred = int(text_model.predict(vec)[0])
    prob = float(max(text_model.predict_proba(vec)[0]))
    return pred, prob

# ===== REASONS FOR TEXT =====
def get_simple_reason(text):
    cleaned = clean_text(text)
    reasons = []

    suspicious_words = [
        "update", "verify", "bank", "account", "urgent", "click",
        "login", "otp", "free", "winner", "prize", "suspended",
        "password", "confirm", "security", "limited", "blocked",
        "আপডেট", "ব্যাংক", "অ্যাকাউন্ট", "জরুরি", "ক্লিক",
        "লগইন", "otp", "পাসওয়ার্ড", "ভেরিফাই", "ব্লক"
    ]

    for word in suspicious_words:
        if word in cleaned:
            reasons.append(f"Suspicious keyword detected: {word}")

    if "http" in text.lower() or "www" in text.lower():
        reasons.append("Contains a URL link")

    if "@" in text and "." in text:
        reasons.append("Contains email-like content")

    if not reasons:
        reasons.append("No major suspicious indicator found")

    return reasons[:3]

# ===== AWARENESS MESSAGES =====
def get_awareness_message(prediction, input_type):
    if prediction == "Phishing":
        return [
            "Do not click on suspicious links. | সন্দেহজনক লিংকে ক্লিক করবেন না।",
            "Do not share OTP, password, or banking information. | OTP, পাসওয়ার্ড বা ব্যাংক তথ্য শেয়ার করবেন না।",
            "Verify the sender or website before taking action. | কোনো পদক্ষেপ নেওয়ার আগে প্রেরক বা ওয়েবসাইট যাচাই করুন।",
            "Report suspicious content to the relevant authority. | সন্দেহজনক বিষয় যথাযথ কর্তৃপক্ষকে জানান।"
        ]
    else:
        return [
            "This content appears safe, but always stay cautious. | এটি নিরাপদ মনে হলেও সবসময় সতর্ক থাকুন।",
            "Avoid sharing sensitive information unnecessarily. | অপ্রয়োজনে সংবেদনশীল তথ্য শেয়ার করবেন না।",
            "Always verify unknown links and senders. | অপরিচিত লিংক ও প্রেরক সবসময় যাচাই করুন।"
        ]

# ===== URL FEATURE BUILDER =====
def build_url_features(url):
    url = str(url).lower()

    features = {
        "length_url": len(url),
        "nb_dots": url.count('.'),
        "nb_hyphens": url.count('-'),
        "nb_at": url.count('@'),
        "nb_qm": url.count('?'),
        "nb_and": url.count('&'),
        "nb_eq": url.count('='),
        "nb_slash": url.count('/'),
        "nb_www": 1 if "www" in url else 0,
        "nb_com": 1 if ".com" in url else 0,
        "https_token": 1 if "https" in url else 0,
        "ratio_digits_url": sum(c.isdigit() for c in url) / max(len(url), 1),
        "nb_subdomains": max(url.count('.') - 1, 0),
        "prefix_suffix": 1 if '-' in url else 0,
        "shortening_service": 1 if any(x in url for x in ["bit.ly", "tinyurl", "t.co", "cutt.ly"]) else 0,
        "phish_hints": sum(1 for w in ["login", "verify", "update", "secure", "bank", "account"] if w in url),
        "suspecious_tld": 1 if any(tld in url for tld in [".tk", ".ml", ".ga", ".cf"]) else 0,
    }

    return features

# ===== URL REASONS =====
def get_url_reasons(url):
    url_lower = str(url).lower()
    reasons = []

    if "login" in url_lower:
        reasons.append("Suspicious keyword detected: login")
    if "verify" in url_lower:
        reasons.append("Suspicious keyword detected: verify")
    if "update" in url_lower:
        reasons.append("Suspicious keyword detected: update")
    if "bank" in url_lower:
        reasons.append("Suspicious keyword detected: bank")
    if "account" in url_lower:
        reasons.append("Suspicious keyword detected: account")
    if any(x in url_lower for x in ["bit.ly", "tinyurl", "t.co", "cutt.ly"]):
        reasons.append("Uses URL shortening service")
    if any(tld in url_lower for tld in [".tk", ".ml", ".ga", ".cf"]):
        reasons.append("Contains suspicious top-level domain")
    if "http" in url_lower or "www" in url_lower:
        reasons.append("Contains a URL link")

    if not reasons:
        reasons.append("No major suspicious indicator found")

    return reasons[:3]

# ===== ROOT PAGE =====
@app.route("/")
def serve_index():
    return send_from_directory(APP_DIR, "index.html")

# ===== SMS ROUTE =====
@app.route("/predict_sms", methods=["POST"])
def predict_sms():
    data = request.get_json()

    if not data or "text" not in data:
        return jsonify({"error": "No SMS text provided"}), 400

    text = data["text"]
    pred, conf = predict_text_model(text)
    prediction = "Phishing" if pred == 1 else "Legitimate"

    return jsonify({
        "type": "SMS",
        "input": text,
        "prediction": prediction,
        "confidence": round(conf, 4),
        "reasons": get_simple_reason(text),
        "awareness": get_awareness_message(prediction, "SMS")
    })

# ===== EMAIL ROUTE =====
@app.route("/predict_email", methods=["POST"])
def predict_email():
    data = request.get_json()

    if not data or "text" not in data:
        return jsonify({"error": "No email text provided"}), 400

    text = data["text"]
    pred, conf = predict_text_model(text)
    prediction = "Phishing" if pred == 1 else "Legitimate"

    return jsonify({
        "type": "Email",
        "input": text,
        "prediction": prediction,
        "confidence": round(conf, 4),
        "reasons": get_simple_reason(text),
        "awareness": get_awareness_message(prediction, "Email")
    })

# ===== URL ROUTE =====
@app.route("/predict_url", methods=["POST"])
def predict_url():
    data = request.get_json()

    if not data or "url" not in data:
        return jsonify({"error": "No URL provided"}), 400

    url = data["url"]

    feature_dict = build_url_features(url)
    row = {col: feature_dict.get(col, 0) for col in url_feature_names}
    X_url = pd.DataFrame([row])

    pred = int(url_model.predict(X_url)[0])
    prob = float(max(url_model.predict_proba(X_url)[0]))
    prediction = "Phishing" if pred == 1 else "Legitimate"

    return jsonify({
        "type": "URL",
        "input": url,
        "prediction": prediction,
        "confidence": round(prob, 4),
        "reasons": get_url_reasons(url),
        "awareness": get_awareness_message(prediction, "URL")
    })

# ===== RUN APP =====
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port, debug=False)
