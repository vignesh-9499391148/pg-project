import pandas as pd
import numpy as np
import json

class NumpyEncoder(json.JSONEncoder):
    """ Custom encoder for numpy data types """
    def default(self, obj):
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
                            np.int16, np.int32, np.int64, np.uint8,
                            np.uint16, np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

df = pd.DataFrame({
    'Amount': [np.float64(10.5), np.float64(20.1)],
    'Time': [np.int64(1), np.int64(2)],
    'V1': [np.float32(-1.5), np.float32(0.5)]
})
row = df.iloc[0]
results = [{
    'id': 'TXN-1234',
    'amount': row['Amount'],
    'is_fraud': True,
    'fraud_probability': 0.95,
    'xai_explanation': [{'feature': 'Time', 'weight': 0.8}],
    'raw_data': row.to_dict()
}]

try:
    json_str = json.dumps(results, cls=NumpyEncoder)
    print("SUCCESS: JSON serialized properly.")
    print("Output:", json_str)
except Exception as e:
    import traceback
    traceback.print_exc()
    print("ERROR:", type(e).__name__, str(e))
