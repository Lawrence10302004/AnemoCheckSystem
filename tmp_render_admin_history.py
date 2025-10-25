from app import app, User
import database as db

# Create a fake admin user for rendering context
fake_user_data = {
    'id': 1,
    'username': 'admin',
    'email': 'admin@example.com',
    'first_name': 'Admin',
    'last_name': 'User',
    'gender': 'other',
    'date_of_birth': '1990-01-01',
    'medical_id': '',
    'is_admin': 1,
    'created_at': '2025-01-01 00:00:00',
    'last_login': '2025-10-18 00:00:00'
}
fake_user = User(fake_user_data)

# Get records
records = db.get_all_classification_history()

# Compute stats
total_records = len(records)
anemic_cases = 0
normal_cases = 0
for r in records:
    pc = (r.get('predicted_class') or '').strip()
    if pc in ('Mild', 'Moderate', 'Severe'):
        anemic_cases += 1
    elif pc == 'Normal':
        normal_cases += 1
    else:
        if 'normal' in pc.lower():
            normal_cases += 1

anemia_rate = (anemic_cases / total_records * 100) if total_records else 0.0

with app.test_request_context('/admin/history'):
    try:
        # Inject fake current_user into the global template context
        app.jinja_env.globals['current_user'] = fake_user
        tmpl = app.jinja_env.get_template('admin/history.html')
        rendered = tmpl.render(records=records, total_records=total_records, anemic_cases=anemic_cases, normal_cases=normal_cases, anemia_rate=anemia_rate)
        print('Rendered length:', len(rendered))
        print(rendered[:500])
    except Exception as e:
        print('Template render failed:', e)
