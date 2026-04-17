from flask import Flask, render_template, request, jsonify
from chatbot.engine import get_response
import os
import datetime

app = Flask(__name__)

# -------------------- HOME PAGE --------------------
@app.route("/")
def home():
    return render_template("index.html")

# -------------------- CHAT API --------------------
@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()

    # Validate input
    if not data or "message" not in data:
        return jsonify({"response": "Invalid input"}), 400

    user_input = data["message"].strip()

    if user_input == "":
        return jsonify({"response": "Please enter a message."})

    # Get AI response
    response = get_response(user_input)

    # -------------------- LOGGING --------------------
    try:
        base_dir = os.path.dirname(__file__)
        log_dir = os.path.join(base_dir, "logs")

        # Ensure logs folder exists
        os.makedirs(log_dir, exist_ok=True)

        log_path = os.path.join(log_dir, "chat_log.txt")

        with open(log_path, "a", encoding="utf-8") as f:
            time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            f.write(f"[{time}] USER: {user_input}\n")
            f.write(f"[{time}] BOT: {response}\n\n")

    except Exception as e:
        print("Logging error:", e)

    return jsonify({"response": response})

# -------------------- RUN SERVER --------------------
if __name__ == "__main__":
    app.run(debug=True)