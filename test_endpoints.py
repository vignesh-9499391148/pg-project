import requests
import json

try:
    with open('test_banksim.csv', 'w') as f:
        # Full set of features based on typical BankSim/Synthetic data
        f.write("step,type,amount,nameOrig,oldbalanceOrg,newbalanceOrig,nameDest,oldbalanceDest,newbalanceDest,age,gender,category,isFraud\n")
        f.write("1,PAYMENT,9839.64,C1231,170136.0,160296.36,M197,0.0,0.0,4,M,es_transportation,0\n")
        f.write("1,PAYMENT,186.05,C1666,21249.0,21062.95,M480,0.0,0.0,2,M,es_food,0\n")
except Exception as e:
    print(f"Error creating test file: {e}")

print("Testing /api/upload...")
files = {'file': ('test_banksim.csv', open('test_banksim.csv', 'rb'), 'text/csv')}
data = {'module_type': 'banksim'}
r = requests.post('http://127.0.0.1:5000/api/upload', files=files, data=data)
print(f"Status: {r.status_code}, Content-Type: {r.headers.get('content-type')}")
print("Response text:", r.text[:200])

if r.status_code == 200:
    filename = r.json().get('filename')
    print("\nTesting /api/predict_batch...")
    r2 = requests.post('http://127.0.0.1:5000/api/predict_batch', json={'filename': filename, 'module_type': 'banksim'})
    print(f"Status: {r2.status_code}, Content-Type: {r2.headers.get('content-type')}")
    print("Response text:", r2.text[:200])

print("\nTesting /api/predict_single...")
# Provide all features that were in the CSV to avoid preprocessor errors
r3 = requests.post('http://127.0.0.1:5000/api/predict_single', json={
    'module_type': 'banksim', 
    'features': {
        'step': 1,
        'type': 'PAYMENT',
        'amount': 150.0,
        'nameOrig': 'C123',
        'oldbalanceOrg': 1000.0,
        'newbalanceOrig': 850.0,
        'nameDest': 'M456',
        'oldbalanceDest': 0.0,
        'newbalanceDest': 0.0,
        'age': 3,
        'gender': 'F',
        'category': 'es_food'
    }
})
print(f"Status: {r3.status_code}, Content-Type: {r3.headers.get('content-type')}")
print("Response text:", r3.text[:500])
