from importlib import reload
import app as appmod
reload(appmod)
app = appmod.app
rules = [r.rule for r in app.url_map.iter_rules()]
print('export_history.csv' in rules, '/export/history.csv' in rules)
print('/admin/export/users.csv' in rules, '/admin/export/classification_history.csv' in rules, '/admin/export/medical_data.csv' in rules)
print('\n'.join(sorted(rules)))
