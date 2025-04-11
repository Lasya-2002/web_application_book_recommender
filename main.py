from flask import Flask, request, jsonify, session
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import sqlite3
import secrets
import torch
import bitsandbytes as bnb
import re
import json
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime

app = Flask(__name__)
app.secret_key = secrets.token_hex(16)  # Secret key for session management

model_path = "sri-lasya/falcon-1B-book-recommendation"  # Change this to your local model directory
tokenizer = AutoTokenizer.from_pretrained(model_path)

# Load model with 8-bit quantization
model = AutoModelForCausalLM.from_pretrained(model_path, load_in_8bit=True, device_map="auto")

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device being used: {device}")

# Initialize database
def init_db():
    conn = sqlite3.connect('book_recommendations.db')
    c = conn.cursor()
    
    # Create users table
    c.execute('''CREATE TABLE IF NOT EXISTS users
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  username TEXT UNIQUE NOT NULL,
                  password TEXT NOT NULL,
                  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)''')
    
    # Create sessions table
    c.execute('''CREATE TABLE IF NOT EXISTS sessions
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  user_id INTEGER NOT NULL,
                  session_id TEXT NOT NULL UNIQUE,
                  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                  FOREIGN KEY(user_id) REFERENCES users(id))''')
    
    conn.commit()
    conn.close()

init_db()

# Database connection helper
def get_db_connection():
    conn = sqlite3.connect('book_recommendations.db')
    conn.row_factory = sqlite3.Row
    return conn

# User authentication endpoints
@app.route('/signup', methods=['POST'])
def signup():
    data = request.get_json()
    username = data.get('username')
    password = data.get('password')

    if not username or not password:
        return jsonify({'error': 'Username and password required'}), 400

    conn = get_db_connection()
    try:
        hashed_pw = generate_password_hash(password)
        conn.execute('INSERT INTO users (username, password) VALUES (?, ?)',
                     (username, hashed_pw))
        conn.commit()
        return jsonify({'message': 'User created successfully'}), 201
    except sqlite3.IntegrityError:
        return jsonify({'error': 'Username already exists'}), 409
    finally:
        conn.close()

@app.route('/login', methods=['POST'])
def login():
    data = request.get_json()
    username = data.get('username')
    password = data.get('password')

    conn = get_db_connection()
    user = conn.execute('SELECT * FROM users WHERE username = ?', (username,)).fetchone()
    conn.close()

    if not user or not check_password_hash(user['password'], password):
        return jsonify({'error': 'Invalid credentials'}), 401

    # Create new session
    session_id = secrets.token_urlsafe(32)
    conn = get_db_connection()
    conn.execute('INSERT INTO sessions (user_id, session_id) VALUES (?, ?)',
                 (user['id'], session_id))
    conn.commit()
    conn.close()

    return jsonify({'session_id': session_id}), 200

@app.route('/verify', methods=['POST'])
def verify_session():
    session_id = request.headers.get('Session-ID')
    if not session_id:
        return jsonify({'error': 'Session ID required'}), 401

    conn = get_db_connection()
    session_data = conn.execute('''SELECT users.username 
                                 FROM sessions 
                                 JOIN users ON sessions.user_id = users.id 
                                 WHERE session_id = ?''', (session_id,)).fetchone()
    conn.close()

    if not session_data:
        return jsonify({'error': 'Invalid session'}), 401

    return jsonify({'username': session_data['username']}), 200

# Existing recommendation endpoint (modified for authentication)
@app.route("/recommend", methods=["POST"])
def recommend_books():
    # Verify session first
    session_id = request.headers.get('Session-ID')
    if not session_id:
        return jsonify({"error": "Authentication required"}), 401

    conn = get_db_connection()
    valid_session = conn.execute('SELECT * FROM sessions WHERE session_id = ?', 
                               (session_id,)).fetchone()
    conn.close()
    
    if not valid_session:
        return jsonify({"error": "Invalid session"}), 401

    # Existing recommendation logic
    data = request.get_json()
    if not data or "input" not in data:
        return jsonify({"error": "Invalid input"}), 400

    user_input = data["input"]
    prompt = f"### Input: {user_input}\n### Output:"

    # Tokenize input and generate recommendations
    inputs = tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True).to(device)
    with torch.no_grad():
        output_ids = model.generate(**inputs, max_length=500, num_return_sequences=1, temperature=0.7)

    output_text = tokenizer.decode(output_ids[0].cpu(), skip_special_tokens=True)

    # Debugging - Check raw model output
    print(f"Raw output from model:\n{output_text}")

    # Extract only the first valid JSON object
    match = re.search(r"\{.*?\}", output_text, re.DOTALL)

    if match:
        first_json_str = match.group(0)  # Get the first JSON object as a string
        try:
            recommendation = json.loads(first_json_str)  # Convert to dictionary safely
            return jsonify({"recommendations": recommendation})
        except json.JSONDecodeError as e:
            print(f"Error parsing JSON: {e}")

    return jsonify({"error": "Failed to parse recommendations"}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)