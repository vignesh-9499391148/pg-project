import requests
import json
import pandas as pd
import numpy as np

# Create a mock credit card CSV
df = pd.DataFrame({
    'Time': [0, 1, 2],
    'V1': [-1.35, 1.19, -1.35],
    'V2': [-0.07, 0.26, -0.48],
    'V3': [2.53, 0.16, 2.49],
    'Amount': [149.62, 2.69, 378.66],
    'Class': [0, 0, 0]
})
df.to_csv('test_creditcard.csv', index=False)

print("Testing /api/upload (creditcard)...")
files = {'file': ('test_creditcard.csv', open('test_creditcard.csv', 'rb'), 'text/csv')}
data = {'module_type': 'creditcard'}
r = requests.post('http://127.0.0.1:5000/api/upload', files=files, data=data)
print(f"Status: {r.status_code}, Content-Type: {r.headers.get('content-type')}")
print("Response:", r.text[:200])

if r.status_code == 200:
    filename = r.json().get('filename')
    print("\nTesting /api/predict_batch (creditcard)...")
    r2 = requests.post('http://127.0.0.1:5000/api/predict_batch', json={'filename': filename, 'module_type': 'creditcard'})
    print(f"Status: {r2.status_code}, Content-Type: {r2.headers.get('content-type')}")
    print("Response snippet:", r2.text[:500])
