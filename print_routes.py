from app import app

rules = sorted(app.url_map.iter_rules(), key=lambda r: r.rule)
for r in rules:
    methods = ','.join(sorted(r.methods - {'HEAD', 'OPTIONS'}))
    print(f"{r.rule} -> {methods}")
