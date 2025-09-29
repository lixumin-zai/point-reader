import requests
import base64

def post_test():
    url = "http://localhost:20070/point-reader"
    with open("test.jpg", "rb") as f:
        image_base64 = base64.b64encode(f.read()).decode("utf-8")
    data = {
        "image_base64": image_base64,
        "prompt": "detect lismin"
    }
    response = requests.post(url, json=data)
    print(response.json())

if __name__ == "__main__":
    post_test()