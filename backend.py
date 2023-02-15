import os

from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse
from gpt_index import GPTSimpleVectorIndex, LLMPredictor
from langchain import OpenAI

uploadfolder = "uploads"

app = FastAPI()
llm_predictor = LLMPredictor(
    llm=OpenAI(temperature=0, model_name="text-davinci-003", max_tokens=1024)
)
filepath = os.path.join(uploadfolder, "filedata.json")
index = GPTSimpleVectorIndex.load_from_disk(filepath, llm_predictor=llm_predictor)

# If you don't care about long answers, you can initialize the index with default 256 token limit simply by:
# index = GPTSimpleVectorIndex.load_from_disk('data.json')


@app.post("/answer")
async def answer(question: str):
    prompt = f'You are a helpful support agent. You are asked: "{question}". Try to use only the information provided. Format your answer nicely as a Markdown page.'
    response = index.query(prompt).response.strip()
    return JSONResponse(content=response)


@app.get("/")
async def main():
    content = "hello world"
    return Response(content=content, media_type="text/markdown")
