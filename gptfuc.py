import os
from operator import itemgetter

import numpy as np
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from langchain import hub
from langchain.document_loaders import DirectoryLoader
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.schema import StrOutputParser
from langchain.text_splitter import (
    CharacterTextSplitter,
    RecursiveCharacterTextSplitter,
)
from langchain_community.chat_models import QianfanChatEndpoint
from langchain_community.chat_models.tongyi import ChatTongyi
from langchain_community.embeddings import HuggingFaceHubEmbeddings
from langchain_community.vectorstores import FAISS, SupabaseVectorStore
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from supabase import Client, create_client

# import pinecone
from upload import add_upload_folder

load_dotenv()

supabase_url = os.environ.get("SUPABASE_URL")
supabase_key = os.environ.get("SUPABASE_KEY")
HF_API_TOKEN = os.environ.get("HF_API_TOKEN")
AZURE_BASE_URL = os.environ.get("AZURE_BASE_URL")
AZURE_API_KEY = os.environ.get("AZURE_API_KEY")
AZURE_DEPLOYMENT_NAME = os.environ.get("AZURE_DEPLOYMENT_NAME")
AZURE_DEPLOYMENT_NAME_16K = os.environ.get("AZURE_DEPLOYMENT_NAME_16K")
AZURE_DEPLOYMENT_NAME_GPT4 = os.environ.get("AZURE_DEPLOYMENT_NAME_GPT4")
AZURE_DEPLOYMENT_NAME_GPT4_32K = os.environ.get("AZURE_DEPLOYMENT_NAME_GPT4_32K")
AZURE_DEPLOYMENT_NAME_GPT4_TURBO = os.environ.get("AZURE_DEPLOYMENT_NAME_GPT4_TURBO")
AZURE_DEPLOYMENT_NAME_EMBEDDING = os.environ.get("AZURE_DEPLOYMENT_NAME_EMBEDDING")

# from qdrant_client import QdrantClient
model_name = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
# embeddings =HuggingFaceEmbeddings(model_name=model_name)
# embeddings = OpenAIEmbeddings()

embeddings = HuggingFaceHubEmbeddings(
    repo_id=model_name,
    task="feature-extraction",
    huggingfacehub_api_token=HF_API_TOKEN,
)

embeddings_openai = AzureOpenAIEmbeddings(
    azure_endpoint=AZURE_BASE_URL,
    azure_deployment=AZURE_DEPLOYMENT_NAME_EMBEDDING,
    openai_api_version="2023-08-01-preview",
    openai_api_key=AZURE_API_KEY,
)

filerawfolder = "fileraw"
fileidxfolder = "fileidx"

uploadfolder = "uploads"
# backendurl = "http://localhost:8000"

# convert gpt model name to azure deployment name
gpt_to_deployment = {
    "gpt-35-turbo": AZURE_DEPLOYMENT_NAME,
    "gpt-35-turbo-16k": AZURE_DEPLOYMENT_NAME_16K,
    "gpt-4": AZURE_DEPLOYMENT_NAME_GPT4,
    "gpt-4-32k": AZURE_DEPLOYMENT_NAME_GPT4_32K,
    "gpt-4-turbo": AZURE_DEPLOYMENT_NAME_GPT4_TURBO,
}


# choose chatllm base on model name
def get_chatllm(model_name):
    if model_name == "tongyi":
        llm = ChatTongyi(
            streaming=True,
        )
    elif (
        model_name == "ERNIE-Bot-4"
        or model_name == "ERNIE-Bot-turbo"
        or model_name == "ChatGLM2-6B-32K"
        or model_name == "Yi-34B-Chat"
    ):
        llm = QianfanChatEndpoint(
            model=model_name,
        )
    elif model_name == "gemini-pro":
        llm = ChatGoogleGenerativeAI(
            model=model_name, convert_system_message_to_human=True
        )
    else:
        llm = get_azurellm(model_name)
    return llm


# use azure llm based on model name
def get_azurellm(model_name):
    deployment_name = gpt_to_deployment[model_name]
    llm = AzureChatOpenAI(
        azure_endpoint=AZURE_BASE_URL,
        openai_api_version="2023-12-01-preview",
        azure_deployment=deployment_name,
        openai_api_key=AZURE_API_KEY,
    )
    return llm


@st.cache_resource
def init_supabase():
    supabase: Client = create_client(supabase_url, supabase_key)
    return supabase


supabase = init_supabase()


def build_index():
    """
    Ingests data into LangChain by creating an FAISS index of OpenAI embeddings for text files in a folder "fileraw".
    The created index is saved to a file in the folder "fileidx".
    """

    loader = DirectoryLoader(filerawfolder, glob="**/*.*")
    documents = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    # use tiktoken
    # text_splitter = CharacterTextSplitter.from_tiktoken_encoder(chunk_size=1000, chunk_overlap=0)
    docs = text_splitter.split_documents(documents)
    # generate metadata from file path

    # Create vector store from documents and save to disk
    # store = FAISS.from_texts(docs, OpenAIEmbeddings())
    store = FAISS.from_documents(docs, embeddings)
    store.save_local(fileidxfolder)
    # db2 = Chroma.from_documents(docs, embeddings)


# create function to add new documents to the index
def add_to_index():
    """
    Adds new documents to the LangChain index by creating an FAISS index of OpenAI embeddings for text files in a folder "fileraw".
    The created index is saved to a file in the folder "fileidx".
    """

    loader = DirectoryLoader(filerawfolder, glob="**/*.*")
    documents = loader.load()
    # text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)

    # use tiktoken
    # text_splitter = CharacterTextSplitter.from_tiktoken_encoder(chunk_size=500, chunk_overlap=50)
    docs = text_splitter.split_documents(documents)
    # print("docs",docs)
    # get faiss client
    store = FAISS.load_local(fileidxfolder, embeddings)

    # Create vector store from documents and save to disk
    store.add_documents(docs)
    store.save_local(fileidxfolder)


def similarity_search(question, topk=4, industry="", items=[]):
    collection_name = industry_name_to_code(industry)

    # get supabase
    store = SupabaseVectorStore(
        client=supabase,
        table_name=collection_name,
        query_name="match_" + collection_name,
        embedding=embeddings_openai,
    )

    filter = convert_list_to_filter(items)
    print(filter)

    docs = store.similarity_search(query=question, k=topk, filter=filter)
    # retriever = store.as_retriever(search_type="similarity",search_kwargs={ "k":topk ,"filter":filter})
    # docs = retriever.get_relevant_documents(question)
    df = docs_to_df(docs)
    return df


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


# convert document list to pandas dataframe
def docs_to_df(docs):
    """
    Converts a list of documents to a pandas dataframe.
    """
    data = []
    for document in docs:
        page_content = document.page_content
        metadata = document.metadata
        plc = metadata["监管要求"]
        sec = metadata["结构"]
        row = {"条款": page_content, "监管要求": plc, "结构": sec}
        data.append(row)
    df = pd.DataFrame(data)
    return df


def convert_list_to_filter(lst):
    if len(lst) >= 1:
        return {"监管要求": lst[0]}
    else:
        return {}
        # return {"监管要求": {"$in": [item for item in lst]}}


def convert_index_to_df(rule_choice):
    """
    Converts the index of LangChain to a pandas dataframe.
    """
    store = FAISS.load_local(fileidxfolder, embeddings)
    index = store.index

    # restruct index
    n = index.ntotal  # Total number of vectors in the index
    d = index.d  # Dimension of each vector

    # Initialize an empty array to hold the vectors
    vectors = np.zeros((n, d), dtype=np.float32)

    # Retrieve vectors
    for i in range(n):
        vectors[i] = index.reconstruct(i)

    # Retrieve to vectors list
    vecls = vectors.tolist()

    print(len(vecls))

    # convert to numpy
    # emb = np.array(vectors)

    # Convert to DataFrame
    # indexdf = pd.DataFrame(vectors, columns=[f'dim_{i}' for i in range(d)])

    # get docstore
    db = store.docstore._dict

    # get texts
    all_texts = [doc.page_content for doc in db.values()]

    # get metadata
    all_metadata = [doc.metadata["source"] for doc in db.values()]

    # get index_to_docstore_id
    all_keys = list(db.keys())

    # get enumberate index
    all_index = list(range(len(all_keys)))

    # convert texts, metadata, keys to dataframe
    plcdf = pd.DataFrame(
        {
            "条款": all_texts,
            "监管要求": all_metadata,
            "结构": all_index,
            "id": all_keys,
            "embeddings": vecls,
        }
    )

    # convert rule_choice to filter
    filter = add_upload_folder(rule_choice)
    print(filter)
    # filter by rule_choice
    plcdf = plcdf[plcdf["监管要求"].isin(filter)]
    embls = plcdf[plcdf["监管要求"].isin(filter)]["embeddings"].tolist()

    emb = np.array(embls)
    print(emb.shape)

    return plcdf, emb


def searchupload(question, upload_choice, topk):

    store = FAISS.load_local(fileidxfolder, embeddings)
    filter = upload_to_dict(upload_choice)
    print(filter)

    docs = store.similarity_search(query=question, k=topk, filter=filter)
    # retriever = store.as_retriever(search_type="similarity",search_kwargs={ "k":topk ,"filter":filter})
    # docs = retriever.get_relevant_documents(question)
    df = upload_to_df(docs)
    return df


def upload_to_df(docs):
    """
    Converts a list of documents to a pandas dataframe.
    """
    data = []
    for document in docs:
        page_content = document.page_content
        metadata = document.metadata
        plc = metadata["source"]
        row = {"条款": page_content, "来源": plc}
        data.append(row)
    df = pd.DataFrame(data)
    return df


def upload_to_dict(lst):
    if len(lst) == 1:
        return {"source": filerawfolder + "/" + lst[0]}
    else:
        return {}
    # else:
    #     rulels=[]
    #     for rule in lst:
    #         rulels.append(filerawfolder+'/'+ rule)
    #     return {"source": {"$in": rulels}}


def gpt_vectoranswer(
    question, upload_choice, chaintype="stuff", top_k=4, model_name="gpt-35-turbo"
):
    # get faiss client
    store = FAISS.load_local(fileidxfolder, embeddings)
    filter = upload_to_dict(upload_choice)

    # system_template = """根据提供的背景信息，请准确和全面地回答用户的问题。
    # 如果您不确定或不知道答案，请直接说明您不知道，避免编造任何信息。
    # {context}"""

    # messages = [
    #     SystemMessagePromptTemplate.from_template(system_template),
    #     HumanMessagePromptTemplate.from_template("{question}"),
    # ]
    # prompt = ChatPromptTemplate.from_messages(messages)

    # chain_type_kwargs = {"prompt": prompt}
    prompt = hub.pull("vyang/gpt_answer")

    llm = get_chatllm(model_name)

    retriever = store.as_retriever(search_kwargs={"k": top_k, "filter": filter})

    # chain = RetrievalQA.from_chain_type(
    #     get_azurellm(model_name),
    #     chain_type=chaintype,
    #     # vectorstore=store,
    #     retriever=retriever,
    #     # k=top_k,
    #     return_source_documents=True,
    #     chain_type_kwargs=chain_type_kwargs,
    # )
    # result = chain({"query": question})

    retriever_from_llm = MultiQueryRetriever.from_llm(retriever=retriever, llm=llm)

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    output_parser = StrOutputParser()

    rag_chain_from_docs = (
        {
            "context": lambda input: format_docs(input["documents"]),
            "question": itemgetter("question"),
        }
        | prompt
        | llm
        | output_parser
    )

    rag_chain_with_source = RunnableParallel(
        {"documents": retriever_from_llm, "question": RunnablePassthrough()}
    ) | {
        # "contents": lambda input: [doc.page_content for doc in input["documents"]],
        "documents": lambda input: [doc for doc in input["documents"]],
        "answer": rag_chain_from_docs,
    }

    result = rag_chain_with_source.invoke(question)

    answer = result["answer"]
    # sourcedf=None
    source = result["documents"]
    sourcedf = upload_to_df(source)

    return answer, sourcedf


def get_matchplc(question, filetype, industry, choicels, model_name="gpt-35-turbo"):

    if filetype == "上传文件":
        # get faiss client
        store = FAISS.load_local(fileidxfolder, embeddings)
        filter = upload_to_dict(choicels)

    else:
        collection_name = industry_name_to_code(industry)

        # get supabase
        store = SupabaseVectorStore(
            client=supabase,
            table_name=collection_name,
            query_name="match_" + collection_name,
            embedding=embeddings_openai,
        )

        filter = convert_list_to_filter(choicels)

    retriever = store.as_retriever(
        search_type="similarity", search_kwargs={"k": 4, "filter": filter}
    )

    chat_prompt = hub.pull("vyang/get_matchplc")

    output_parser = StrOutputParser()

    llm = get_chatllm(model_name)

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    output_parser = StrOutputParser()

    rag_chain_from_docs = (
        {
            "context": lambda input: format_docs(input["documents"]),
            "question": itemgetter("question"),
        }
        | chat_prompt
        | llm
        | output_parser
    )

    rag_chain_with_source = RunnableParallel(
        {"documents": retriever, "question": RunnablePassthrough()}
    ) | {
        # "contents": lambda input: [doc.page_content for doc in input["documents"]],
        "documents": lambda input: [doc for doc in input["documents"]],
        "answer": rag_chain_from_docs,
    }

    result = rag_chain_with_source.invoke(question)

    answer = result["answer"]
    # sourcedf=None
    source = result["documents"]

    if filetype == "上传文件":
        sourcedf = upload_to_df(source)
    else:
        sourcedf = docs_to_df(source)

    return answer, sourcedf
