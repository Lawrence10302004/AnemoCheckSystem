import database

# Get database connection
conn = database.get_db_connection()
cursor = conn.cursor()

print("=== FIXING IMPORTED DATA ===")

# Check current state
cursor.execute('SELECT COUNT(*) FROM classification_import_data')
imported_count = cursor.fetchone()[0]
print(f"Total imported records: {imported_count}")

cursor.execute('SELECT COUNT(*) FROM imported_files')
files_count = cursor.fetchone()[0]
print(f"Total imported files: {files_count}")

if imported_count > 0 and files_count == 0:
    print("Creating missing file record for existing imported data...")
    
    # Create a file record for the existing imported data
    cursor.execute('''
        INSERT INTO imported_files (filename, original_filename, total_records, imported_by, is_applied)
        VALUES (?, ?, ?, ?, 1)
    ''', ('legacy_import.csv', 'legacy_import.csv', imported_count, 1))
    
    file_id = cursor.lastrowid
    print(f"Created file record with ID: {file_id}")
    
    # Update all existing imported data to link to this file
    cursor.execute('''
        UPDATE classification_import_data 
        SET file_id = ?
        WHERE file_id IS NULL
    ''', (file_id,))
    
    updated_rows = cursor.rowcount
    print(f"Updated {updated_rows} records with file_id")
    
    conn.commit()
    print("Data fixed successfully!")
    
    # Test the fix
    print("\n=== TESTING FIX ===")
    try:
        combined_data = database.get_applied_imported_data()
        print(f"Applied imported gender: {combined_data['gender_stats']}")
        print(f"Applied imported age: {combined_data['age_groups']}")
        print(f"Applied imported severity: {combined_data['severity_stats']}")
    except Exception as e:
        print(f"Error testing fix: {e}")
else:
    print("No fix needed - data is already properly linked")

conn.close()
