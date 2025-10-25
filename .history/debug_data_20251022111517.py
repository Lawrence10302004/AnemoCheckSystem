import database

# Get database connection
conn = database.get_db_connection()
cursor = conn.cursor()

print("=== ORIGINAL DATA ===")
# Check original gender stats
cursor.execute('SELECT gender, COUNT(*) FROM users WHERE is_admin = 0 GROUP BY gender')
gender_stats = dict(cursor.fetchall())
print(f"Original gender stats: {gender_stats}")

# Check original age groups
cursor.execute('''
    SELECT 
        CASE 
            WHEN (strftime('%Y', 'now') - strftime('%Y', date_of_birth)) < 18 THEN 'Under 18'
            WHEN (strftime('%Y', 'now') - strftime('%Y', date_of_birth)) BETWEEN 18 AND 30 THEN '18-30'
            WHEN (strftime('%Y', 'now') - strftime('%Y', date_of_birth)) BETWEEN 31 AND 45 THEN '31-45'
            WHEN (strftime('%Y', 'now') - strftime('%Y', date_of_birth)) BETWEEN 46 AND 60 THEN '46-60'
            ELSE 'Over 60'
        END as age_group,
        COUNT(*) as count
    FROM users 
    WHERE date_of_birth IS NOT NULL AND is_admin = 0
    GROUP BY age_group
''')
age_groups = dict(cursor.fetchall())
print(f"Original age groups: {age_groups}")

# Check original severity stats
cursor.execute('''
    SELECT 
        CASE 
            WHEN predicted_class LIKE '%Normal%' OR predicted_class LIKE '%normal%' THEN 'Normal'
            WHEN predicted_class LIKE '%Mild%' OR predicted_class LIKE '%mild%' THEN 'Mild Anemia'
            WHEN predicted_class LIKE '%Moderate%' OR predicted_class LIKE '%moderate%' THEN 'Moderate Anemia'
            WHEN predicted_class LIKE '%Severe%' OR predicted_class LIKE '%severe%' THEN 'Severe Anemia'
            ELSE 'Other'
        END as severity,
        COUNT(*) as count
    FROM classification_history
    GROUP BY severity
''')
severity_stats = dict(cursor.fetchall())
print(f"Original severity stats: {severity_stats}")

print("\n=== IMPORTED DATA ===")
# Check imported data
cursor.execute('SELECT COUNT(*) FROM classification_import_data')
imported_count = cursor.fetchone()[0]
print(f"Total imported records: {imported_count}")

if imported_count > 0:
    cursor.execute('SELECT gender, COUNT(*) FROM classification_import_data GROUP BY gender')
    imported_gender = dict(cursor.fetchall())
    print(f"Imported gender: {imported_gender}")
    
    cursor.execute('''
        SELECT 
            CASE 
                WHEN age < 18 THEN 'Under 18'
                WHEN age BETWEEN 18 AND 30 THEN '18-30'
                WHEN age BETWEEN 31 AND 45 THEN '31-45'
                WHEN age BETWEEN 46 AND 60 THEN '46-60'
                ELSE 'Over 60'
            END as age_group,
            COUNT(*) as count
        FROM classification_import_data
        GROUP BY age_group
    ''')
    imported_age = dict(cursor.fetchall())
    print(f"Imported age groups: {imported_age}")
    
    cursor.execute('SELECT category, COUNT(*) FROM classification_import_data GROUP BY category')
    imported_severity = dict(cursor.fetchall())
    print(f"Imported severity: {imported_severity}")

print("\n=== IMPORTED FILES STATUS ===")
# Check imported files status
cursor.execute('SELECT id, filename, is_applied FROM imported_files')
files = cursor.fetchall()
print(f"Imported files: {files}")

print("\n=== COMBINED DATA (what should be shown) ===")
# Test the combined function
try:
    combined_data = database.get_applied_imported_data()
    print(f"Applied imported gender: {combined_data['gender_stats']}")
    print(f"Applied imported age: {combined_data['age_groups']}")
    print(f"Applied imported severity: {combined_data['severity_stats']}")
except Exception as e:
    print(f"Error getting applied imported data: {e}")

conn.close()
