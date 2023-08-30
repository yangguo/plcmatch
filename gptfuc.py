import os

import requests

from supabase import Client, create_client
from dotenv import load_dotenv
# import pinecone


load_dotenv()

supabase_url = os.environ.get("SUPABASE_URL")
supabase_key = os.environ.get("SUPABASE_KEY")

uploadfolder = "uploads"
backendurl = "http://localhost:8000"

# @st.cache_resource
def init_supabase():
    supabase: Client = create_client(supabase_url, supabase_key)
    return supabase


def build_index():
    # documents = SimpleDirectoryReader(uploadfolder, recursive=True).load_data()
    # index = GPTSimpleVectorIndex(documents)
    # index.save_to_disk(os.path.join(uploadfolder, "filedata.json"))
    print('build_index')

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


# convert industry chinese name to english name
def industry_name_to_code(industry_name):
    """
    Converts an industry name to an industry code.
    """
    industry_name = industry_name.lower()
    if industry_name == "银行":
        return "bank"
    elif industry_name == "保险":
        return "insurance"
    elif industry_name == "证券":
        return "securities"
    elif industry_name == "基金":
        return "fund"
    elif industry_name == "期货":
        return "futures"
    elif industry_name == "投行":
        return "invbank"
    else:
        return "other"