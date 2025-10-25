"""
Database cleanup script for AnemoCheck
Clears test data and sets up email configuration
"""
import sqlite3
import os
from datetime import datetime

def cleanup_database():
    """Clean up database and set up email settings."""
    # Use the same database path as the main app
    DB_PATH = os.environ.get('DATABASE_PATH', 'anemia_classification.db')
    
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    try:
        print("Starting database cleanup...")
        
        # 1. Delete all classification history
        cursor.execute("DELETE FROM classification_history")
        print("[OK] Cleared classification history")
        
        # 2. Delete all medical data
        cursor.execute("DELETE FROM medical_data")
        print("[OK] Cleared medical data")
        
        # 3. Delete all chat data
        cursor.execute("DELETE FROM chat_messages")
        cursor.execute("DELETE FROM chat_conversations")
        print("[OK] Cleared chat data")
        
        # 4. Delete all OTP data
        cursor.execute("DELETE FROM otp_verification")
        cursor.execute("DELETE FROM password_reset_otp")
        print("[OK] Cleared OTP data")
        
        # 5. Delete all imported data
        cursor.execute("DELETE FROM classification_import_data")
        cursor.execute("DELETE FROM imported_files")
        print("[OK] Cleared imported data")
        
        # 6. Delete all non-admin users
        cursor.execute("DELETE FROM users WHERE is_admin = 0")
        print("[OK] Deleted all non-admin users")
        
        # 7. Set up email settings (you need to configure these manually)
        current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        email_settings = [
            ('brevo_sender_email', 'lawrencetolentino103004@gmail.com'),
            ('brevo_sender_name', 'anemocheck'),
            ('enable_notifications', 'true')
        ]
        
        for setting_name, setting_value in email_settings:
            cursor.execute("""
                INSERT OR REPLACE INTO system_settings 
                (setting_name, setting_value, updated_at) 
                VALUES (?, ?, ?)
            """, (setting_name, setting_value, current_time))
        
        print("[OK] Email settings configured")
        
        # 8. Check admin user
        cursor.execute("SELECT * FROM users WHERE username = 'admin'")
        admin_user = cursor.fetchone()
        
        if admin_user:
            print("[OK] Admin user exists")
        else:
            print("[WARNING] Admin user not found - will be created on next app start")
        
        conn.commit()
        print("\n[SUCCESS] Database cleanup completed successfully!")
        print("\nNext steps:")
        print("1. Configure Brevo API key in Railway environment variables")
        print("2. Or use the admin panel to set up email settings")
        print("3. Redeploy your application")
        
    except Exception as e:
        print(f"[ERROR] Error during cleanup: {e}")
        conn.rollback()
    finally:
        conn.close()

if __name__ == "__main__":
    cleanup_database()
