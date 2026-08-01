import sqlite3

conn = sqlite3.connect("churn.db")

cursor = conn.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS prediction_history(

    id INTEGER PRIMARY KEY AUTOINCREMENT,

    prediction TEXT,

    confidence REAL,

    risk TEXT,

    date_time TEXT

)
""")

conn.commit()
conn.close()

print("Database Created Successfully")