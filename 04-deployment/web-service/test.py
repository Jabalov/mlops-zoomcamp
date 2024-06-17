import requests

ride = {
    "PULocationID": 20,
    "DOLocationID": 56,
    "trip_distance": 90
}

url = 'http://localhost:9696/predict'
response = requests.post(url, json=ride)
print(response.json())
