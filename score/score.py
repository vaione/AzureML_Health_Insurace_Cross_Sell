from xgboost import XGBClassifier
from azureml.core.model import Model
from sklearn.preprocessing import StandardScaler
import joblib
import json
import numpy as np
import time

def init():
    global scaler, xgb_model
    model_path = Model.get_model_path('Health_XGBoost_HD')
    #scaler = joblib.load('./scaler.pkl')
    xgb_model = joblib.load(f"{model_path}/saved_model")
    #Print statement for appinsights custom traces:
    print ("model initialized" + time.strftime("%H:%M:%S"))

def run(raw_data):
    try:
        data = json.loads(raw_data)
        data = np.array(data).reshape(1,-1)
        #data = scaler.transform(data)
        result = xgb_model.predict(data).tolist()
        info = {
                "Time": time.strftime("%H:%M:%S"),
                "input": raw_data,
                "output": result
                }
        print(json.dumps(info))

        return json.dumps(result)
    except Exception as e:
        error = str(e)
        print (error + time.strftime("%H:%M:%S"))
        return error

