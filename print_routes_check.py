import sys
from pathlib import Path

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    import app
except Exception as e:
    print('ERROR importing app module:', repr(e))
    raise

import sys
print('imported app module from:', getattr(app, '__file__', '<unknown>'))
print('export_endpoints in sys.modules?', 'export_endpoints' in sys.modules)
if 'export_endpoints' in sys.modules:
    print('export_endpoints file:', sys.modules['export_endpoints'].__file__)

print('Module dir:', sorted(name for name in dir(app) if not name.startswith('__')))

# Try to locate the Flask app object on the module
flask_app = None
for candidate in ('app', 'application', 'flask_app'):
    if hasattr(app, candidate):
        flask_app = getattr(app, candidate)
        break

if flask_app is None:
    print('Could not find Flask app object on module. Available attributes above.')
    raise SystemExit(1)

# If export_endpoints is not yet imported, try to import and initialize it now to reproduce app behavior
if 'export_endpoints' not in sys.modules:
    try:
        import export_endpoints
        print('Imported export_endpoints:', getattr(export_endpoints, '__file__', '<no file>'))
        try:
            export_endpoints.init_app(flask_app)
            print('Called export_endpoints.init_app successfully')
        except Exception as e:
            print('Error calling init_app:', repr(e))
    except Exception as e:
        print('Error importing export_endpoints:', repr(e))

routes = sorted([str(r) for r in flask_app.url_map.iter_rules()])
print('--- Registered routes ---')
for r in routes:
    print(r)

expected = [
    '/export/history.csv',
    '/admin/export/users.csv',
    '/admin/export/classification_history.csv',
    '/admin/export/medical_data.csv'
]

print('\n--- Export endpoints present? ---')
for e in expected:
    print(e, e in [rule.rule for rule in flask_app.url_map.iter_rules()])

print('\n--- Module attributes starting with "export" ---')
print([name for name in dir(app) if name.startswith('export') or name.startswith('admin_export')])
