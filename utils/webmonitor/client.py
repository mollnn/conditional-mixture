import json
import requests
from requests.structures import CaseInsensitiveDict

def send_post(taskname, method, msg):
    url = "https://rest-hz.goeasy.io/v2/pubsub/publish"
    headers = CaseInsensitiveDict()
    headers["Accept"] = "application/json"
    headers["Content-Type"] = "application/json"

    data = json.dumps({
        "appkey": "BC-d64d2e4cf2c44dd58347025519502934",
        "channel": "test_channel",
        "content": f"{taskname}&{method}&{msg}"
    })
    resp = requests.post(url, headers=headers, data=data)