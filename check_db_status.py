#!/usr/bin/env python3
"""
Check database status after cleanup
"""

import sqlite3
import os

def check_database_status():
    """Check the current state of the database."""
    db_path = 'anemia_classification.db'
    
    if not os.path.exists(db_path):
        print("Database file not found!")
        return
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    try:
        # Check users
        cursor.execute("SELECT COUNT(*) as count FROM users")
        user_count = cursor.fetchone()[0]
        print(f"Users: {user_count}")
        
        cursor.execute("SELECT COUNT(*) as count FROM users WHERE is_admin = 0")
        regular_users = cursor.fetchone()[0]
        print(f"Regular users: {regular_users}")
        
        # Check classifications
        cursor.execute("SELECT COUNT(*) as count FROM classification_history")
        classification_count = cursor.fetchone()[0]
        print(f"Classification records: {classification_count}")
        
        # Check email settings
        cursor.execute("SELECT setting_name, setting_value FROM system_settings WHERE setting_name LIKE 'brevo%'")
        email_settings = cursor.fetchall()
        print("Email settings:")
        for setting_name, setting_value in email_settings:
            if 'key' in setting_name.lower():
                print(f"  {setting_name}: [HIDDEN]")
            else:
                print(f"  {setting_name}: {setting_value}")
        
        print("\nDatabase status:")
        if user_count == 1 and regular_users == 0 and classification_count == 0:
            print("✅ Database is clean and ready for deployment!")
        else:
            print("⚠️  Database may not be fully clean")
            
    except Exception as e:
        print(f"Error checking database: {e}")
    finally:
        conn.close()

if __name__ == "__main__":
    check_database_status()
