import traceback
from jinja2 import Environment, FileSystemLoader, TemplateSyntaxError

TEMPLATES_DIR = 'templates'
env = Environment(loader=FileSystemLoader(TEMPLATES_DIR))

p = 'templates/history.html'
src = open(p, 'r', encoding='utf-8').read()
try:
    env.parse(src)
    print('parsed ok')
except TemplateSyntaxError as e:
    print('TemplateSyntaxError:', e.message, 'lineno=', e.lineno)
    traceback.print_exc()
except Exception as e:
    print('Other exception:', e)
    traceback.print_exc()
