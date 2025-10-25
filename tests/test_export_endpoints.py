import sys
from pathlib import Path
# Ensure project root is on sys.path so imports like `from app import app` work when this
# script is executed from the tests/ subdirectory.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app import app, User
import database as db
import importlib

# Some environments may not have exported routes initialized during import.
# Initialize export endpoints explicitly to ensure routes are registered for the test app.
try:
    export_endpoints = importlib.import_module('export_endpoints')
    if hasattr(export_endpoints, 'init_app'):
        export_endpoints.init_app(app)
except Exception as e:
    print('Warning: failed to import or init export_endpoints:', e)

# Print registered routes for debugging
print('Registered routes:')
for rule in sorted([r.rule for r in app.url_map.iter_rules()]):
    print(rule)

# Create fake admin user object for login
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

# Helper to call endpoint with a logged in user via test_request_context
def call_endpoint(path):
    with app.test_client() as client:
        # Use the test client and set up a login by manipulating the session
        with client.session_transaction() as sess:
            sess['user_id'] = fake_user.id
            sess['_user_id'] = str(fake_user.id)
        resp = client.get(path)
        print('===', path, '===')
        print('Status:', resp.status_code)
        for h in ('Content-Disposition','Content-Type'):
            if h in resp.headers:
                print(h+':', resp.headers[h])
        text = resp.get_data(as_text=True)
        print('Preview:', text[:400])


if __name__ == '__main__':
    # Ensure DB helpers work
    print('Export endpoints to check:')
    paths = ['/admin/export/classification_history.csv','/admin/export/users.csv','/admin/export/medical_data.csv','/export/history.csv']
    for p in paths:
        call_endpoint(p)
