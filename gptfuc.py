import os

import requests
from gpt_index import GPTSimpleVectorIndex, SimpleDirectoryReader

# os.environ["OPENAI_API_KEY"] = os.environ.get("OPENAI_API_KEY")

uploadfolder = "uploads"
backendurl = "http://localhost:8000"

# openai_api_key=os.environ.get("OPENAI_API_KEY")
# if openai_api_key is None:
#     print("请设置OPENAI_API_KEY")


def build_index():
    documents = SimpleDirectoryReader(uploadfolder, recursive=True).load_data()
    index = GPTSimpleVectorIndex(documents)
    index.save_to_disk(os.path.join(uploadfolder, "filedata.json"))


def gpt_answer(question):
    try:
        url = backendurl + "/answer"
        payload = {
            "question": question,
        }
        headers = {}
        res = requests.post(url, headers=headers, params=payload)
        result = res.json()
        print("成功")
    except Exception as e:
        print("错误: " + str(e))
        result = "错误: " + str(e)
    return result
