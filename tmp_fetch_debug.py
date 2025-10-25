import urllib.request, json

def fetch(url):
    try:
        with urllib.request.urlopen(url, timeout=5) as r:
            print('URL:', url)
            body = r.read().decode('utf-8')
            print('Response length:', len(body))
            try:
                data = json.loads(body)
                print(json.dumps(data, indent=2)[:2000])
            except Exception as e:
                print('Non-JSON response, head:\n', body[:800])
    except Exception as e:
        print('URL:', url, 'ERROR:', e)

fetch('http://127.0.0.1:5000/_debug/routes')
print('\n---\n')
fetch('http://192.168.123.38:5000/_debug/routes')
