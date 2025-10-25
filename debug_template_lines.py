from pathlib import Path
p = Path('templates/history.html')
s = p.read_text(encoding='utf-8')
lines = s.splitlines()
for i in range(170, 193):
    if i < 1 or i > len(lines):
        continue
    print(f"{i:04}: {repr(lines[i-1])}")
