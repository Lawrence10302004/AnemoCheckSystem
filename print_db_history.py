import sqlite3
from pprint import pprint
DB='anemia_classification.db'
conn=sqlite3.connect(DB)
conn.row_factory=sqlite3.Row
c=conn.cursor()
print('DB path:', DB)
c.execute('SELECT COUNT(*) as cnt FROM classification_history')
print('Total classification records:', c.fetchone()['cnt'])

print('\nLast 50 classification_history rows:')
c.execute('SELECT ch.id, ch.user_id, ch.created_at, ch.hgb, ch.predicted_class, ch.confidence, u.username FROM classification_history ch LEFT JOIN users u ON ch.user_id = u.id ORDER BY ch.created_at DESC LIMIT 50')
rows=c.fetchall()
for r in rows:
    print(dict(r))

print('\nCounts by predicted_class:')
c.execute('SELECT predicted_class, COUNT(*) as cnt FROM classification_history GROUP BY predicted_class')
for row in c.fetchall():
    print(dict(row))

conn.close()
