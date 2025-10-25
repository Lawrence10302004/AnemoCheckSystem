import sqlite3

# Path to your SQLite database
db_path = 'anemia_classification.db'

# Connect to the database
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

# List of new columns to add
new_columns = [
    ("neutrophils", "REAL"),
    ("lymphocytes", "REAL"),
    ("monocytes", "REAL"),
    ("eosinophils", "REAL"),
    ("basophil", "REAL"),
    ("immature_granulocytes", "REAL")
]

for column_name, column_type in new_columns:
    try:
        cursor.execute(f'ALTER TABLE classification_history ADD COLUMN {column_name} {column_type};')
        print(f"Added column: {column_name}")
    except sqlite3.OperationalError as e:
        print(f"Could not add column {column_name}: {e}")


# Commit changes and close connection
conn.commit()
conn.close()
