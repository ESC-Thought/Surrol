import json
import os
from openai import OpenAI, AuthenticationError
import base64
import requests

image_path = "/home/kejianshi/Desktop/Surgical_Robot/Surrol_Related/IROS_SurRoL/tests/images/pegtransfer_poster.png"

# create image encode function
def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')

base64_image = encode_image(image_path)


client = OpenAI(
    api_key="sk-dpwewyisiiinfmbzkvuqfidhvkxrboerlhivpiuvlrjcmdfi", # 从https://cloud.siliconflow.cn/account/ak获取
    base_url="https://api.siliconflow.cn/v1"
)
response = client.chat.completions.create(
        model="Qwen/Qwen2-VL-72B-Instruct",
        messages=[
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                         "url": f"data:image/jpeg;base64,{base64_image}"
                    }
                },
                {
                    "type": "text",
                    "text": "The given image shows a surgical task named peg transfer, where users are asked to manipulate the robot arm to grasp the red block and put it to the marked target peg. Now, from your observation, cna you please tell me is the task completed, and if not, what should it do next?"
                }
            ],
            "max_tokens": 100,
        }],
        stream=True,
        
)
for chunk in response:
    chunk_message = chunk.choices[0].delta.content
    print(chunk_message, end='', flush=True)