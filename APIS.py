import json

import requests


class APIS:

    def FirebaseAPI(self, isSmoke, CamIp):
        url = "https://fcm.googleapis.com/fcm/send"

        if isSmoke:
            payload = json.dumps({
                "to": "/topics/smoking",
                "notification": {
                    "title": "Smoking Detection ",
                    "body": "There is Smoking Detection in Room " + CamIp
                }
            })
        else:
            payload = json.dumps({
                "to": "/topics/smoking",
                "notification": {
                    "title": "Violence Detection ",
                    "body": "There is Violence Detection in Room " + CamIp
                }
            })
        headers = {
            'Authorization': 'key=AAAA-pC--o0:APA91bFClxzhLL0GrwafVqoUIGxlvlUBuDy4451UStHYJljaFnxjLoPxKmPNajDpQ51i4pLEct_g54ZaS8q1vNV9Rp0Qg7vzNVb0KFi-hdfG1QC1gHy-fmcvk087ABAqRJCZ2ZyUVy3s',
            'Content-Type': 'application/json'
        }
        response = requests.request("POST", url, headers=headers, data=payload)
        print(response.text)
