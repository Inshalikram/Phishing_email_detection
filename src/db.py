from __future__ import annotations

import psycopg2
from psycopg2.extras import RealDictCursor

DB_CONFIG = {
    "dbname": "phishing_db",
    "user": "postgres",
    "password": "NewPassword123",
    "host": "localhost",
    "port": "5432",
}

def get_connection():
    return psycopg2.connect(**DB_CONFIG, cursor_factory=RealDictCursor)

def save_prediction(email_text, model_used, prediction, confidence, clean_text):
    print("DB SAVE CALLED")
    confidence = float(confidence) if confidence is not None else 0.0
    conn = get_connection()
    cur = conn.cursor()

    cur.execute(
        """
        INSERT INTO predictions (email_text, model_used, prediction, confidence, clean_text)
        VALUES (%s, %s, %s, %s, %s)
        """,
        (email_text, model_used, prediction, confidence, clean_text),
    )

    conn.commit()
    print("DB SAVE SUCCESS")

    cur.close()
    conn.close()

def get_all_predictions():
    conn = get_connection()
    cur = conn.cursor()

    cur.execute("SELECT * FROM predictions ORDER BY id DESC")
    rows = cur.fetchall()

    cur.close()
    conn.close()
    return rows