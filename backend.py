from fastapi import FastAPI, Request,Response
from gpt_index import GPTSimpleVectorIndex, LLMPredictor
from langchain import OpenAI

app = FastAPI()
llm_predictor = LLMPredictor(llm=OpenAI(temperature=0, model_name="text-davinci-003", max_tokens=1024))
index = GPTSimpleVectorIndex.load_from_disk('data.json', llm_predictor=llm_predictor)

# If you don't care about long answers, you can initialize the index with default 256 token limit simply by:
# index = GPTSimpleVectorIndex.load_from_disk('data.json')

@app.get('/answer')
async def answer(request: Request, question: str):
    prompt = f'You are a helpful support agent. You are asked: "{question}". Try to use only the information provided. Format your answer nicely as a Markdown page.'
    response = index.query(prompt).response.strip()
    return Response(content=response, media_type='text/markdown')

@app.get('/')
async def main():
    content = open('index.html', 'r').read()
    return Response(content=content, media_type='text/html')
