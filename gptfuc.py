import os
from gpt_index import GPTSimpleVectorIndex, SimpleDirectoryReader

def build_index(folder_path):
    documents = SimpleDirectoryReader(folder_path, recursive=True).load_data()
    index = GPTSimpleVectorIndex(documents)
    index.save_to_disk('data.json')
