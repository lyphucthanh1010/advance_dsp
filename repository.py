import requests

url = 'http://localhost:3000/videos/matched-audios/'

headers = {
    'accept': '*/*',
    'Content-Type': 'application/json',
}

def savePredictionsResult(video_id, data):
    response = requests.put(url + video_id, headers=headers, json=data)

    if (response.status_code == 200):
        print("Update predictions result success")