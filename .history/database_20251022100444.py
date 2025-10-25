"""
Database Module for Anemia Classification System
-----------------------------------------------
This module handles database operations for storing user, admin, and classification data.
"""

import os
import datetime
from zoneinfo import ZoneInfo
import sqlite3
from werkzeug.security import generate_password_hash, check_password_hash

# Database setup
DB_PATH = 'anemia_classification.db'

def get_db_connection():
    """Get database connection."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    """Initialize the database with necessary tables."""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Create users table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE NOT NULL,
        password_hash TEXT NOT NULL,
        email TEXT UNIQUE NOT NULL,
        first_name TEXT,
        last_name TEXT,
        gender TEXT,
        date_of_birth TEXT,
        medical_id TEXT UNIQUE,
        is_admin INTEGER DEFAULT 0,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        last_login TIMESTAMP
    )
    ''')
    
    # Create classification_history table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS classification_history (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER,
        wbc REAL NOT NULL,
        rbc REAL NOT NULL,
        hgb REAL NOT NULL,
        hct REAL NOT NULL,
        mcv REAL NOT NULL,
        mch REAL NOT NULL,
        mchc REAL NOT NULL,
        plt REAL NOT NULL,
        predicted_class TEXT NOT NULL,
        confidence REAL NOT NULL,
        recommendation TEXT,
        notes TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (user_id) REFERENCES users(id)
    )
''')
    
    # Create medical_data table for additional patient information
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS medical_data (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER UNIQUE,
        height REAL,
        weight REAL,
        blood_type TEXT,
        medical_conditions TEXT,
        medications TEXT,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (user_id) REFERENCES users(id)
    )
    ''')
    
    # Create a table for system settings
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS system_settings (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        setting_name TEXT UNIQUE NOT NULL,
        setting_value TEXT,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_by INTEGER,
        FOREIGN KEY (updated_by) REFERENCES users(id)
    )
    ''')
    
    # Create classification_import_data table for imported statistics
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS classification_import_data (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        age INTEGER NOT NULL,
        gender TEXT NOT NULL,
        category TEXT NOT NULL,
        imported_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        file_id INTEGER,
        FOREIGN KEY (file_id) REFERENCES imported_files(id)
    )
    ''')
    
    # Create imported_files table to track imported files
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS imported_files (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        filename TEXT NOT NULL,
        original_filename TEXT NOT NULL,
        total_records INTEGER NOT NULL,
        imported_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        is_applied INTEGER DEFAULT 1,
        imported_by INTEGER,
        FOREIGN KEY (imported_by) REFERENCES users(id)
    )
    ''')
    
    # Insert default admin user if not exists
    cursor.execute("SELECT * FROM users WHERE username = 'admin'")
    if not cursor.fetchone():
        create_user(
            username='admin',
            password='admin123',  # This should be changed immediately in production
            email='admin@anemocheck.com',
            first_name='System',
            last_name='Administrator',
            is_admin=1
        )
        print("Default admin user created.")
    
    # Insert default system settings
    default_settings = [
        ('model_type', 'decision_tree'),
        ('visualization_enabled', 'true'),
        ('recommendation_enabled', 'true'),
        ('threshold_normal', '12.0'),
        ('threshold_mild', '10.0'),
        ('threshold_moderate', '8.0')
    ]
    
    for name, value in default_settings:
        cursor.execute("SELECT * FROM system_settings WHERE setting_name = ?", (name,))
        if not cursor.fetchone():
            cursor.execute(
                "INSERT INTO system_settings (setting_name, setting_value) VALUES (?, ?)",
                (name, value)
            )
    
    conn.commit()
    conn.close()
    
    print("Database initialized successfully.")


def create_user(username, password, email, first_name=None, last_name=None, 
                gender=None, date_of_birth=None, medical_id=None, is_admin=0):
    """Create a new user in the database."""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    try:
        # Normalize optional fields
        normalized_medical_id = None
        if medical_id is not None:
            mid = str(medical_id).strip()
            normalized_medical_id = mid if mid else None  # store NULL when empty
        
        password_hash = generate_password_hash(password)
        cursor.execute(
            """
            INSERT INTO users 
            (username, password_hash, email, first_name, last_name, gender, 
             date_of_birth, medical_id, is_admin)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (username, password_hash, email, first_name, last_name, gender, 
             date_of_birth, normalized_medical_id, is_admin)
        )
        conn.commit()
        user_id = cursor.lastrowid
        
        # Create empty medical_data entry for the user
        cursor.execute(
            "INSERT INTO medical_data (user_id) VALUES (?)",
            (user_id,)
        )
        conn.commit()
        return True, user_id
    except sqlite3.IntegrityError as e:
        if "UNIQUE constraint failed: users.username" in str(e):
            return False, "Username already exists."
        elif "UNIQUE constraint failed: users.email" in str(e):
            return False, "Email already exists."
        elif "UNIQUE constraint failed: users.medical_id" in str(e):
            return False, "Medical ID already exists."
        else:
            return False, str(e)
    finally:
        conn.close()


def verify_user(username, password):
    """Verify user credentials."""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    cursor.execute("SELECT * FROM users WHERE username = ?", (username,))
    user = cursor.fetchone()
    
    if user and check_password_hash(user['password_hash'], password):
        # Update last login time
        cursor.execute(
            "UPDATE users SET last_login = ? WHERE id = ?",
            (datetime.datetime.now(), user['id'])
        )
        conn.commit()
        conn.close()
        return True, dict(user)
    
    conn.close()
    return False, "Invalid username or password."


def get_user(user_id):
    """Get user by ID."""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    cursor.execute("SELECT * FROM users WHERE id = ?", (user_id,))
    user = cursor.fetchone()
    
    conn.close()
    return dict(user) if user else None


def update_user(user_id, **kwargs):
    """Update user information."""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Build the SET part of the SQL query
    set_clauses = []
    values = []
    
    for key, value in kwargs.items():
        if key == 'password':
            set_clauses.append('password_hash = ?')
            values.append(generate_password_hash(value))
        elif key in ['username', 'email', 'first_name', 'last_name', 'gender', 
                     'date_of_birth', 'medical_id', 'is_admin']:
            set_clauses.append(f'{key} = ?')
            if key == 'medical_id':
                # Normalize to NULL when empty so UNIQUE does not block blanks
                if value is None:
                    values.append(None)
                else:
                    mid = str(value).strip()
                    values.append(mid if mid else None)
            else:
                values.append(value)
    
    if not set_clauses:
        conn.close()
        return False, "No valid fields to update."
    
    sql = f"UPDATE users SET {', '.join(set_clauses)} WHERE id = ?"
    values.append(user_id)
    
    try:
        cursor.execute(sql, values)
        conn.commit()
        conn.close()
        return True, "User updated successfully."
    except sqlite3.IntegrityError as e:
        conn.close()
        if "UNIQUE constraint failed: users.username" in str(e):
            return False, "Username already exists."
        elif "UNIQUE constraint failed: users.email" in str(e):
            return False, "Email already exists."
        elif "UNIQUE constraint failed: users.medical_id" in str(e):
            return False, "Medical ID already exists."
        else:
            return False, str(e)


# def add_classification_record(user_id, hemoglobin, predicted_class, confidence, recommendation, notes=None):
#     """Add a classification record to history."""
#     conn = get_db_connection()
#     cursor = conn.cursor()
    
#     cursor.execute(
#         """
#         INSERT INTO classification_history 
#         (user_id, hemoglobin, predicted_class, confidence, recommendation, notes)
#         VALUES (?, ?, ?, ?, ?, ?)
#         """,
#         (user_id, hemoglobin, predicted_class, confidence, recommendation, notes)
#     )
    
#     conn.commit()
#     record_id = cursor.lastrowid
#     conn.close()
    
#     return record_id

def add_classification_record(*args, **kwargs):
    """Add a classification record. Supports two call styles:

    1) Minimal (legacy) form used by some routes:
       add_classification_record(user_id=..., hemoglobin=..., predicted_class=..., confidence=..., recommendation=..., notes=...)

    2) Full CBC form used by other routes:
       add_classification_record(user_id=..., wbc=..., rbc=..., hgb=..., hct=..., mcv=..., mch=..., mchc=..., plt=..., neutrophils=..., ..., predicted_class=..., confidence=..., recommendation=..., notes=...)

    This function always writes a created_at timestamp in Asia/Manila time.
    """
    # Accept both positional and keyword forms; normalize into kwargs
    if len(args) == 1 and not kwargs:
        # Called as add_classification_record(dict) - not expected, treat as error
        raise TypeError("add_classification_record requires keyword arguments")

    user_id = kwargs.get('user_id')
    if user_id is None:
        raise TypeError('user_id is required')

    # Philippines time string
    ph_now = datetime.datetime.now(ZoneInfo('Asia/Manila')).strftime('%Y-%m-%d %H:%M:%S')

    conn = get_db_connection()
    cursor = conn.cursor()

    # Legacy simple API: 'hemoglobin' key may be used instead of 'hgb'
    if 'hemoglobin' in kwargs or ('hgb' in kwargs and len(kwargs) <= 6):
        hgb = kwargs.get('hemoglobin') if 'hemoglobin' in kwargs else kwargs.get('hgb')
        predicted_class = kwargs.get('predicted_class')
        confidence = kwargs.get('confidence')
        recommendation = kwargs.get('recommendation')
        notes = kwargs.get('notes')

        cursor.execute(
            """
            INSERT INTO classification_history
            (user_id, hgb, predicted_class, confidence, recommendation, notes, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (user_id, hgb, predicted_class, confidence, recommendation, notes, ph_now)
        )

    else:
        # Full CBC form - pick values from kwargs, defaulting to 0 or None where sensible
        fields = [
            'wbc', 'rbc', 'hgb', 'hct', 'mcv', 'mch', 'mchc', 'plt',
            'neutrophils', 'lymphocytes', 'monocytes', 'eosinophils', 'basophil', 'immature_granulocytes'
        ]
        values = [kwargs.get(f, 0.0) for f in fields]
        predicted_class = kwargs.get('predicted_class')
        confidence = kwargs.get('confidence')
        recommendation = kwargs.get('recommendation')
        notes = kwargs.get('notes')

        cursor.execute(
            """
            INSERT INTO classification_history 
            (user_id, wbc, rbc, hgb, hct, mcv, mch, mchc, plt,
             neutrophils, lymphocytes, monocytes, eosinophils, basophil, immature_granulocytes,
             predicted_class, confidence, recommendation, notes, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            tuple([user_id] + values + [predicted_class, confidence, recommendation, notes, ph_now])
        )

    conn.commit()
    record_id = cursor.lastrowid
    conn.close()

    return record_id





def get_user_classification_history(user_id, limit=10):
    """Get classification history for a specific user."""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    cursor.execute(
        """
        SELECT * FROM classification_history 
        WHERE user_id = ? 
        ORDER BY created_at DESC
        LIMIT ?
        """,
        (user_id, limit)
    )
    
    history = [dict(row) for row in cursor.fetchall()]
    conn.close()
    
    return history


def get_all_classification_history(limit=100):
    """Get all classification history (legacy, limited). Prefer get_classification_history_paginated."""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    cursor.execute(
        """
        SELECT ch.*, u.username, u.first_name, u.last_name
        FROM classification_history ch
        LEFT JOIN users u ON ch.user_id = u.id
        ORDER BY ch.created_at DESC
        LIMIT ?
        """,
        (limit,)
    )
    
    history = [dict(row) for row in cursor.fetchall()]
    conn.close()
    
    return history

def get_classification_history_paginated(page=1, per_page=5):
    """Get classification history with server-side pagination."""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Calculate offset
    if page < 1:
        page = 1
    offset = (page - 1) * per_page
    
    # Get total count
    cursor.execute("SELECT COUNT(*) as total FROM classification_history")
    total = cursor.fetchone()['total']
    
    # Get paginated results
    cursor.execute(
        """
        SELECT ch.*, u.username, u.first_name, u.last_name
        FROM classification_history ch
        LEFT JOIN users u ON ch.user_id = u.id
        ORDER BY ch.created_at DESC
        LIMIT ? OFFSET ?
        """,
        (per_page, offset)
    )
    
    records = [dict(row) for row in cursor.fetchall()]
    conn.close()
    
    # Calculate pagination info
    total_pages = (total + per_page - 1) // per_page
    has_prev = page > 1
    has_next = page < total_pages
    
    return {
        'records': records,
        'total': total,
        'page': page,
        'per_page': per_page,
        'total_pages': total_pages,
        'has_prev': has_prev,
        'has_next': has_next,
        'prev_num': page - 1 if has_prev else None,
        'next_num': page + 1 if has_next else None
    }


def get_system_setting(setting_name):
    """Get a system setting value."""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    cursor.execute(
        "SELECT setting_value FROM system_settings WHERE setting_name = ?",
        (setting_name,)
    )
    
    row = cursor.fetchone()
    conn.close()
    
    return row['setting_value'] if row else None


def update_system_setting(setting_name, setting_value, updated_by=None):
    """Update a system setting."""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    try:
        # Use INSERT OR REPLACE to handle both new and existing settings
        cursor.execute(
            """
            INSERT OR REPLACE INTO system_settings 
            (setting_name, setting_value, updated_at, updated_by) 
            VALUES (?, ?, ?, ?)
            """,
            (setting_name, setting_value, datetime.datetime.now(), updated_by)
        )
        
        conn.commit()
        conn.close()
        return True
        
    except Exception as e:
        conn.rollback()
        conn.close()
        print(f"Error updating system setting {setting_name}: {str(e)}")
        return False


def get_all_users(limit=100):
    """Get all users (legacy, limited). Prefer get_users_paginated."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(
        """
        SELECT id, username, email, first_name, last_name, gender, 
               date_of_birth, medical_id, is_admin, created_at, last_login
        FROM users
        ORDER BY created_at DESC
        LIMIT ?
        """,
        (limit,)
    )
    users = [dict(row) for row in cursor.fetchall()]
    conn.close()
    return users


def get_users_paginated(page=1, per_page=5):
    """Get users with server-side pagination and counts for UI stats."""
    conn = get_db_connection()
    cursor = conn.cursor()

    # Counts for header cards
    cursor.execute("SELECT COUNT(*) AS total FROM users")
    total = cursor.fetchone()['total']

    cursor.execute("SELECT COUNT(*) AS admins FROM users WHERE is_admin = 1")
    admins = cursor.fetchone()['admins']

    cursor.execute("SELECT COUNT(*) AS regulars FROM users WHERE is_admin = 0")
    regulars = cursor.fetchone()['regulars']

    cursor.execute("SELECT COUNT(*) AS active FROM users WHERE last_login IS NOT NULL")
    active = cursor.fetchone()['active']

    # Pagination window
    if page < 1:
        page = 1
    offset = (page - 1) * per_page

    cursor.execute(
        """
        SELECT id, username, email, first_name, last_name, gender,
               date_of_birth, medical_id, is_admin, created_at, last_login
        FROM users
        ORDER BY created_at DESC
        LIMIT ? OFFSET ?
        """,
        (per_page, offset)
    )
    records = [dict(row) for row in cursor.fetchall()]

    conn.close()

    total_pages = (total + per_page - 1) // per_page
    has_prev = page > 1
    has_next = page < total_pages

    return {
        'records': records,
        'total': total,
        'page': page,
        'per_page': per_page,
        'total_pages': total_pages,
        'has_prev': has_prev,
        'has_next': has_next,
        'prev_num': page - 1 if has_prev else None,
        'next_num': page + 1 if has_next else None,
        # header stats
        'counts': {
            'total': total,
            'regulars': regulars,
            'admins': admins,
            'active': active
        }
    }


def update_medical_data(user_id, **kwargs):
    """Update user medical data."""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Build the SET part of the SQL query
    set_clauses = []
    values = []
    
    for key, value in kwargs.items():
        if key in ['height', 'weight', 'blood_type', 'medical_conditions', 'medications']:
            set_clauses.append(f'{key} = ?')
            values.append(value)
    
    set_clauses.append('updated_at = ?')
    values.append(datetime.datetime.now())
    
    if not set_clauses:
        conn.close()
        return False, "No valid fields to update."
    
    # Check if the record exists
    cursor.execute("SELECT id FROM medical_data WHERE user_id = ?", (user_id,))
    if cursor.fetchone():
        # Update existing record
        sql = f"UPDATE medical_data SET {', '.join(set_clauses)} WHERE user_id = ?"
        values.append(user_id)
        cursor.execute(sql, values)
    else:
        # Insert new record
        keys = [key for key, _ in kwargs.items() if key in ['height', 'weight', 'blood_type', 'medical_conditions', 'medications']]
        keys.append('user_id')
        values.append(user_id)
        
        sql = f"INSERT INTO medical_data ({', '.join(keys)}) VALUES ({', '.join(['?'] * len(keys))})"
        cursor.execute(sql, values)
    
    conn.commit()
    conn.close()
    
    return True, "Medical data updated successfully."


def get_medical_data(user_id):
    """Get user medical data."""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    cursor.execute("SELECT * FROM medical_data WHERE user_id = ?", (user_id,))
    data = cursor.fetchone()
    
    conn.close()
    return dict(data) if data else None


def get_statistics():
    """Get system statistics for admin dashboard."""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Get user count
    cursor.execute("SELECT COUNT(*) as user_count FROM users WHERE is_admin = 0")
    user_count = cursor.fetchone()['user_count']
    
    # Get total classifications
    cursor.execute("SELECT COUNT(*) as classification_count FROM classification_history")
    classification_count = cursor.fetchone()['classification_count']

    # Get class distribution (raw)
    cursor.execute("""
        SELECT predicted_class, COUNT(*) as count
        FROM classification_history
        GROUP BY predicted_class
    """)
    raw_distribution = { (row['predicted_class'] or '').strip(): row['count'] for row in cursor.fetchall() }

    # Normalize labels to canonical forms and aggregate anemic vs normal
    # Common variants in code: 'Normal', 'Mild', 'Moderate', 'Severe',
    # sometimes routes append ' Anemia' (e.g., 'Mild Anemia').
    class_distribution = {}
    anemic_count = 0
    normal_count = 0
    for label, cnt in raw_distribution.items():
        if not label:
            continue
        l = label.lower()
        if 'normal' in l:
            canonical = 'Normal'
            normal_count += cnt
        elif 'mild' in l:
            canonical = 'Mild'
            anemic_count += cnt
        elif 'moderate' in l:
            canonical = 'Moderate'
            anemic_count += cnt
        elif 'severe' in l:
            canonical = 'Severe'
            anemic_count += cnt
        elif 'anemia' in l:
            # fallback: if label mentions anemia but not severity, treat as anemic
            canonical = 'Anemia'
            anemic_count += cnt
        else:
            canonical = label
        class_distribution[canonical] = class_distribution.get(canonical, 0) + cnt
    
    # Get new users in the last 7 days
    cursor.execute("""
        SELECT COUNT(*) as new_user_count 
        FROM users 
        WHERE created_at > datetime('now', '-7 days') AND is_admin = 0
    """)
    new_user_count = cursor.fetchone()['new_user_count']
    
    # Get active users in the last 7 days
    cursor.execute("""
        SELECT COUNT(DISTINCT user_id) as active_user_count 
        FROM classification_history 
        WHERE created_at > datetime('now', '-7 days')
    """)
    active_user_count = cursor.fetchone()['active_user_count']
    
    conn.close()

    # Prepare stats keys expected by the admin template
    return {
        'total_users': user_count,
        'total_classifications': classification_count,
        'class_distribution': class_distribution,
        'anemic_cases': anemic_count,
        'normal_cases': normal_count,
        'new_user_count': new_user_count,
        'active_user_count': active_user_count
    }

def get_recent_classifications(page=1, per_page=5):
    """Get recent classifications with pagination."""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Calculate offset
    offset = (page - 1) * per_page
    
    # Get total count
    cursor.execute("SELECT COUNT(*) as total FROM classification_history")
    total = cursor.fetchone()['total']
    
    # Get paginated results
    cursor.execute("""
        SELECT ch.*, u.username
        FROM classification_history ch
        LEFT JOIN users u ON ch.user_id = u.id
        ORDER BY ch.created_at DESC
        LIMIT ? OFFSET ?
    """, (per_page, offset))
    
    records = [dict(row) for row in cursor.fetchall()]
    conn.close()
    
    # Calculate pagination info
    total_pages = (total + per_page - 1) // per_page
    has_prev = page > 1
    has_next = page < total_pages
    
    return {
        'records': records,
        'total': total,
        'page': page,
        'per_page': per_page,
        'total_pages': total_pages,
        'has_prev': has_prev,
        'has_next': has_next,
        'prev_num': page - 1 if has_prev else None,
        'next_num': page + 1 if has_next else None
    }

def delete_user(user_id):
    """Delete a user and all associated data."""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    try:
        # Check if user exists
        cursor.execute("SELECT id, username, is_admin FROM users WHERE id = ?", (user_id,))
        user = cursor.fetchone()
        
        if not user:
            conn.close()
            return False, "User not found"
        
        # Prevent deletion of admin users
        if user['is_admin']:
            conn.close()
            return False, "Cannot delete admin users"
        
        # Delete user's classification history
        cursor.execute("DELETE FROM classification_history WHERE user_id = ?", (user_id,))
        
        # Delete user's medical data
        cursor.execute("DELETE FROM medical_data WHERE user_id = ?", (user_id,))
        
        # Delete user's chat conversations and messages
        cursor.execute("SELECT id FROM chat_conversations WHERE user_id = ?", (user_id,))
        conversations = cursor.fetchall()
        
        for conv in conversations:
            cursor.execute("DELETE FROM chat_messages WHERE conversation_id = ?", (conv['id'],))
        
        cursor.execute("DELETE FROM chat_conversations WHERE user_id = ?", (user_id,))
        
        # Finally, delete the user
        cursor.execute("DELETE FROM users WHERE id = ?", (user_id,))
        
        conn.commit()
        conn.close()
        
        return True, f"User '{user['username']}' deleted successfully"
        
    except Exception as e:
        conn.rollback()
        conn.close()
        return False, f"Error deleting user: {str(e)}"

def get_classification_record(record_id):
    """Get a specific classification record by ID."""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    try:
        cursor.execute("""
            SELECT * FROM classification_history 
            WHERE id = ?
        """, (record_id,))
        
        record = cursor.fetchone()
        conn.close()
        
        if record:
            return dict(record)
        return None
        
    except Exception as e:
        conn.close()
        return None


def get_user_by_id(user_id):
    """Get user data by ID."""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    try:
        cursor.execute("""
            SELECT * FROM users 
            WHERE id = ?
        """, (user_id,))
        
        user = cursor.fetchone()
        conn.close()
        
        if user:
            return dict(user)
        return None
        
    except Exception as e:
        conn.close()
        return None


def get_user_by_email(email: str):
    """Get user by email, or None if not found."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM users WHERE email = ?", (email,))
    row = cursor.fetchone()
    conn.close()
    return dict(row) if row else None


def get_user_by_medical_id(medical_id: str):
    """Get user by medical ID, or None if not found."""
    if medical_id is None:
        return None
    mid = str(medical_id).strip()
    if not mid:
        return None
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM users WHERE medical_id = ?", (mid,))
    row = cursor.fetchone()
    conn.close()
    return dict(row) if row else None


def get_user_by_username(username: str):
    """Get user by username, or None if not found."""
    if username is None:
        return None
    uname = str(username).strip()
    if not uname:
        return None
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM users WHERE username = ?", (uname,))
    row = cursor.fetchone()
    conn.close()
    return dict(row) if row else None

def get_admin_dashboard_charts():
    """Get data for admin dashboard charts: age groups, gender distribution, and severity classification."""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Age groups distribution
    cursor.execute("""
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
        ORDER BY 
            CASE age_group
                WHEN 'Under 18' THEN 1
                WHEN '18-30' THEN 2
                WHEN '31-45' THEN 3
                WHEN '46-60' THEN 4
                WHEN 'Over 60' THEN 5
            END
    """)
    age_groups = {row['age_group']: row['count'] for row in cursor.fetchall()}
    
    # Gender distribution
    cursor.execute("""
        SELECT gender, COUNT(*) as count
        FROM users 
        WHERE gender IS NOT NULL AND is_admin = 0
        GROUP BY gender
    """)
    gender_stats = {row['gender']: row['count'] for row in cursor.fetchall()}
    
    # Severity classification distribution
    cursor.execute("""
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
        WHERE predicted_class IS NOT NULL
        GROUP BY severity
        ORDER BY 
            CASE severity
                WHEN 'Normal' THEN 1
                WHEN 'Mild Anemia' THEN 2
                WHEN 'Moderate Anemia' THEN 3
                WHEN 'Severe Anemia' THEN 4
                WHEN 'Other' THEN 5
            END
    """)
    severity_stats = {row['severity']: row['count'] for row in cursor.fetchall()}
    
    conn.close()
    
    return {
        'age_groups': age_groups,
        'gender_stats': gender_stats,
        'severity_stats': severity_stats
    }


def create_imported_file(filename, original_filename, total_records, imported_by):
    """Create a new imported file record."""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    cursor.execute('''
        INSERT INTO imported_files (filename, original_filename, total_records, imported_by)
        VALUES (?, ?, ?, ?)
    ''', (filename, original_filename, total_records, imported_by))
    
    file_id = cursor.lastrowid
    conn.commit()
    conn.close()
    
    return file_id


def get_imported_files():
    """Get all imported files with their status."""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Check if imported_files table exists
    cursor.execute("""
        SELECT name FROM sqlite_master 
        WHERE type='table' AND name='imported_files'
    """)
    
    if not cursor.fetchone():
        print("imported_files table does not exist")
        conn.close()
        return []
    
    cursor.execute('''
        SELECT 
            id,
            original_filename as filename,
            imported_at,
            total_records,
            is_applied,
            u.username as imported_by
        FROM imported_files f
        LEFT JOIN users u ON f.imported_by = u.id
        ORDER BY imported_at DESC
    ''')
    
    files = []
    for row in cursor.fetchall():
        files.append({
            'id': row['id'],
            'filename': row['filename'],
            'imported_at': row['imported_at'],
            'total_records': row['total_records'],
            'is_applied': bool(row['is_applied']),
            'imported_by': row['imported_by']
        })
    
    print(f"Found {len(files)} imported files")
    conn.close()
    return files


def update_file_status(file_id, is_applied):
    """Update the applied status of an imported file."""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    cursor.execute('''
        UPDATE imported_files 
        SET is_applied = ?
        WHERE id = ?
    ''', (1 if is_applied else 0, file_id))
    
    conn.commit()
    conn.close()


def delete_imported_file(file_id):
    """Delete an imported file and all its data."""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Delete all data associated with this file
    cursor.execute('DELETE FROM classification_import_data WHERE file_id = ?', (file_id,))
    
    # Delete the file record
    cursor.execute('DELETE FROM imported_files WHERE id = ?', (file_id,))
    
    conn.commit()
    conn.close()


def get_applied_imported_data():
    """Get all applied imported data for chart calculations."""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Check if file_id column exists
    cursor.execute("PRAGMA table_info(classification_import_data)")
    columns = [column[1] for column in cursor.fetchall()]
    has_file_id = 'file_id' in columns
    
    if has_file_id:
        # Age groups from applied imported data
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
            FROM classification_import_data cid
            JOIN imported_files f ON cid.file_id = f.id
            WHERE f.is_applied = 1
            GROUP BY age_group
        ''')
        age_groups = {row['age_group']: row['count'] for row in cursor.fetchall()}
        
        # Gender stats from applied imported data
        cursor.execute('''
            SELECT gender, COUNT(*) as count
            FROM classification_import_data cid
            JOIN imported_files f ON cid.file_id = f.id
            WHERE f.is_applied = 1
            GROUP BY gender
        ''')
        gender_stats = {row['gender']: row['count'] for row in cursor.fetchall()}
        
        # Severity stats from applied imported data
        cursor.execute('''
            SELECT category, COUNT(*) as count
            FROM classification_import_data cid
            JOIN imported_files f ON cid.file_id = f.id
            WHERE f.is_applied = 1
            GROUP BY category
        ''')
        severity_stats = {row['category']: row['count'] for row in cursor.fetchall()}
    else:
        # Fallback to old format - get all imported data
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
        age_groups = {row['age_group']: row['count'] for row in cursor.fetchall()}
        
        cursor.execute('''
            SELECT gender, COUNT(*) as count
            FROM classification_import_data
            GROUP BY gender
        ''')
        gender_stats = {row['gender']: row['count'] for row in cursor.fetchall()}
        
        cursor.execute('''
            SELECT category, COUNT(*) as count
            FROM classification_import_data
            GROUP BY category
        ''')
        severity_stats = {row['category']: row['count'] for row in cursor.fetchall()}
    
    conn.close()
    
    return {
        'age_groups': age_groups,
        'gender_stats': gender_stats,
        'severity_stats': severity_stats
    }


def migrate_database():
    """Migrate database to add new tables if they don't exist."""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Check if imported_files table exists
    cursor.execute("""
        SELECT name FROM sqlite_master 
        WHERE type='table' AND name='imported_files'
    """)
    
    if not cursor.fetchone():
        # Create imported_files table
        cursor.execute('''
            CREATE TABLE imported_files (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                filename TEXT NOT NULL,
                original_filename TEXT NOT NULL,
                total_records INTEGER NOT NULL,
                imported_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                is_applied INTEGER DEFAULT 1,
                imported_by INTEGER,
                FOREIGN KEY (imported_by) REFERENCES users(id)
            )
        ''')
        
        # Add file_id column to classification_import_data if it doesn't exist
        try:
            cursor.execute('ALTER TABLE classification_import_data ADD COLUMN file_id INTEGER')
        except:
            # Column might already exist, ignore error
            pass
        
        conn.commit()
        print("Database migrated successfully - added imported_files table")
    
    conn.close()


# Initialize the database when this module is imported
if not os.path.exists(DB_PATH):
    init_db()
else:
    # Run migration for existing databases
    migrate_database()


