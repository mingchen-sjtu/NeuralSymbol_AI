# -- coding: utf-8 --**
import requests
import json

# 设置请求头部信息
headers = {"Content-Type": "application/json"}

# 设置机器人的IP地址和端口号
url = "http://192.168.56.2/api/v2.0.0/robots/robot"

# 设置机器人的移动速度，单位是米/秒
speed = 0.5

# 设置机器人要移动到的目的地坐标
destination = {"x": 19.300, "y": 5.450, "theta": 177.797}

# 构造请求体
payload = {
    "positions": [
        {
            "position": {
            "x": destination["x"],
            "y": destination["y"],
            "theta": destination["theta"]
            },
            "speed": speed
        }
    ]
}

# 发送POST请求
response = requests.post(url, headers=headers, data=json.dumps(payload))

# 处理响应结果
result = json.loads(response.text)

if result["success"]:
    print("机器人移动成功！")
else:
    print("机器人移动失败：", result["message"])