import urllib.request

url = 'http://127.0.0.1:5000/admin/export/classification_history.csv'
try:
    with urllib.request.urlopen(url, timeout=5) as r:
        print('Status code:', r.getcode())
        headers = r.getheaders()
        for k,v in headers:
            if k.lower() in ('content-disposition','content-type'):
                print(k+':', v)
        body = r.read(400).decode('utf-8', errors='replace')
        print('\nFirst 400 chars of body:')
        print(body)
except Exception as e:
    print('Request failed:', e)
