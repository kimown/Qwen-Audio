```test2.py
import requests
import os

os.environ["no_proxy"] = "localhost,127.0.0.1,::1,::"

def main():
    conversation = [
    {'role': 'system', 'content': 'You are a helpful assistant.'},
    {"role": "user", "content": [
        {"type": "audio", "audio_url": "http://127.0.0.1:8080/1736496205751_6061021962085442_a.mp3.wav.mp3"},
        {"type":"text","text":"语音识别音频里的文字，禁止输出其他无关内容"}
    ]},
]

    input = {'conver':conversation}
    res = requests.post("http://127.0.0.1:8602/analyse", json=input)
    # 查看状态码
    print(f"Status Code: {res.status_code}")

    # 查看响应头
    print(f"Headers: {res.headers}")

    # 查看响应体（作为文本）
    print(f"Response Body (Text): {res.text}")
if __name__=='__main__':
    main()
```

```test.py
import requests
import os

os.environ["no_proxy"] = "localhost,127.0.0.1,::1,::"

def main():
    conversation = [
    {'role': 'system', 'content': 'You are a helpful assistant.'},
    {"role": "user", "content": [
        {"type": "audio", "audio": "./30.mp3"},
        {"type":"text","text":"解析音频中的文字"}
    ]},
]

    input = {'conver':conversation}
    res = requests.post("http://[::]:8602/analyse", json=input)
    # 查看状态码
    print(f"Status Code: {res.status_code}")

    # 查看响应头
    print(f"Headers: {res.headers}")

    # 查看响应体（作为文本）
    print(f"Response Body (Text): {res.text}")
if __name__=='__main__':
    main()
```


```audio_api.py
from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import librosa
from transformers import Qwen2AudioForConditionalGeneration, Qwen2AudioProcessor
import torch
from io import BytesIO
import uvicorn



app = FastAPI()
origins = [
    "*"
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 初始化模型和处理器
processor = Qwen2AudioProcessor.from_pretrained("./models")
model = Qwen2AudioForConditionalGeneration.from_pretrained("./models", torch_dtype="auto", device_map="auto").eval()

class Conversations(BaseModel):
    conver: list


def analyse_audio(conversation) -> str:

    text = processor.apply_chat_template(conversation, chat_template=processor.default_chat_template, add_generation_prompt=True, tokenize=False)
    audios = []
    for message in conversation:
        if isinstance(message["content"], list):
            for ele in message["content"]:
                if ele["type"] == "audio":
                    audios.append(
                        librosa.load(ele['audio'], sr=processor.feature_extractor.sampling_rate)[0]
                    )
    inputs = processor(text=text, audios=audios, return_tensors="pt", padding=True)
    inputs["input_ids"] = inputs.input_ids.to("cuda")
    generate_ids = model.generate(**inputs, max_length=4096)
    generate_ids = generate_ids[:, inputs.input_ids.size(1):]
    response = processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

    return response

@app.post("/analyse")
async def analyse(input_data:Conversations):
    print(input_data)
    res = analyse_audio(input_data.conver)
    print(type(res))
    return {'response':res}
    # response = analyse_audio(query=query, audio=audio)
    # return {"response": response}


uvicorn.run(app, host=["::"], port=8602)
```


```a.py
from io import BytesIO
from urllib.request import urlopen
import librosa
from transformers import Qwen2AudioForConditionalGeneration, AutoProcessor

print("0---------------------")
processor = AutoProcessor.from_pretrained("./models")
model = Qwen2AudioForConditionalGeneration.from_pretrained("./models", device_map="auto")

conversation1 = [
    {'role': 'system', 'content': '你是一个专业的音乐助手，不论用户用什么语言与你沟通，你都要用中文回答用户。'},
    {"role": "user", "content": [
        {"type": "audio", "audio_url": "./30.mp3"},
        {"type": "text", "text": "有几个人在说话?"}
    ]},
]

conversation2 = [
    {'role': 'system', 'content': '你是一个专业的音乐助手，不论用户用什么语言与你沟通，你都要用中文回答用户。'},
    {"role": "user", "content": [
        {"type": "audio", "audio_url": "./30.mp3"},
        {"type": "text", "text": "是男声还是女声?"}
    ]},
]

conversation3 = [
    {'role': 'system', 'content': '你是一个专业的音乐助手，不论用户用什么语言与你沟通，你都要用中文回答用户。'},
    {"role": "user", "content": [
        {"type": "audio", "audio_url": "./30.mp3"},
        {"type": "text", "text": "这段音频人物的情绪是怎么样的?"}
    ]},
]

conversation4 = [
    {'role': 'system', 'content': '你是一个专业的语音助手，不论用户用什么语言与你沟通，你都要用中文回答用户。'},
    {"role": "user", "content": [
        {"type": "audio", "audio_url": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen2-Audio/demo/audio_1719289379143.wav"},
        {"type": "text", "text": "语音识别音频里的文字，禁止输出其他无关内容"}
    ]},
]


conversations = [conversation1, conversation2, conversation3]
conversations = [conversation4]

text = [processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False) for conversation in conversations]

audios = []
for conversation in conversations:
    for message in conversation:
        if isinstance(message["content"], list):
            for ele in message["content"]:
                if ele["type"] == "audio":
                    if ele['audio_url'].startswith("http"):
                        audios.append(
                            librosa.load(
                                BytesIO(urlopen(ele['audio_url']).read()),
                                sr=processor.feature_extractor.sampling_rate)[0])
                    else:
                        audios.append(
                            librosa.load(
                                ele['audio_url'],
                                sr=processor.feature_extractor.sampling_rate)[0])

print("1---------------------")
inputs = processor(text=text, audios=audios, return_tensors="pt", padding=True)
# inputs.input_ids = inputs.input_ids.to("cuda")
inputs = dict(**inputs)
inputs["input_ids"] = inputs["input_ids"].to("cuda")
print("12---------------------")
generate_ids = model.generate(**inputs, max_length=256)
# generate_ids = generate_ids[:, inputs.input_ids.size(1):]
generate_ids = generate_ids[:, inputs["input_ids"].size(1):]

response = processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
print("12311---------------------")
print(response)
print("123---------------------")
```
