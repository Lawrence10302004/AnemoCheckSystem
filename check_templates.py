import os
from jinja2 import Environment, FileSystemLoader, TemplateSyntaxError

TEMPLATES_DIR = os.path.join(os.path.dirname(__file__), 'templates')
env = Environment(loader=FileSystemLoader(TEMPLATES_DIR))

errors = []
for root, dirs, files in os.walk(TEMPLATES_DIR):
    for f in files:
        if not f.endswith('.html'):
            continue
        rel_dir = os.path.relpath(root, TEMPLATES_DIR)
        if rel_dir == '.':
            template_name = f
        else:
            template_name = os.path.join(rel_dir, f).replace('\\', '/')
        path = os.path.join(root, f)
        try:
            src = open(path, 'r', encoding='utf-8').read()
            env.parse(src)
        except TemplateSyntaxError as e:
            errors.append((template_name, e.message, e.lineno))
        except Exception as e:
            errors.append((template_name, str(e), None))

if not errors:
    print('All templates parsed successfully')
else:
    print('Template errors found:')
    for t, msg, lineno in errors:
        print(f"- {t}: {msg} (line {lineno})")
    raise SystemExit(1)
