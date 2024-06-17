import http.client
import json

conn = http.client.HTTPSConnection("google.serper.dev")
payload = json.dumps({
  # "q": "apple inc"
  "q": "甚麼是ai"
})
headers = {
  'X-API-KEY': '38d97764ecf095e003bacdc7b161fa0727f93dda',
  'Content-Type': 'application/json'
}
conn.request("POST", "/search", payload, headers)
res = conn.getresponse()
data = res.read()
print(data.decode("utf-8"))
