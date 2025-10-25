import requests

try:
    r = requests.get('http://127.0.0.1:5000/admin/export/classification_history.csv')
    print('Status code:', r.status_code)
    print('Headers:')
    for k,v in r.headers.items():
        if k.lower() in ('content-disposition','content-type'):
            print(k+':', v)
    print('\nFirst 400 chars of body:')
    print(r.text[:400])
except Exception as e:
    print('Request failed:', e)
