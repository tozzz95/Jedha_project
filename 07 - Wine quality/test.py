import requests
response_predict = requests.post("http://127.0.0.1:5000/predict", json={"input": [[7.0, 0.27, 0.36, 20.7, 0.045, 45.0, 170.0, 1.001, 3.0, 0.45, 8.8], [5.0, 0.98, 0.32, 18.9, 0.050, 75.0, 122.0, 0.401, 3.1, 0.21, 1.2]]})
response_predict.json()

print(response_predict, response_predict.json())
