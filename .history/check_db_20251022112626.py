import sqlite3
import os

# Check database files
db_files = [f for f in os.listdir('.') if f.endswith('.db')]
print('Database files:', db_files)

for db_file in db_files:
    try:
        conn = sqlite3.connect(db_file)
        cursor = conn.cursor()
        
        # Check if imported_files table exists
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='imported_files'")
        result = cursor.fetchone()
        
        if result:
            print(f'{db_file}: imported_files table exists')
            # Get sample timestamp
            cursor.execute('SELECT imported_at FROM imported_files ORDER BY imported_at DESC LIMIT 1')
            sample = cursor.fetchone()
            if sample:
                print(f'Sample timestamp: {sample[0]}')
            else:
                print('No data in imported_files table')
        else:
            print(f'{db_file}: imported_files table does not exist')
        
        conn.close()
    except Exception as e:
        print(f'Error checking {db_file}: {e}')
