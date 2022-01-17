import os, glob
import numpy as np
import pandas as pd
import streamlit as st
import asyncio
import torch
import jieba.analyse
import spacy

from textrank4zh import TextRank4Sentence
from transformers import RoFormerModel, RoFormerTokenizer

modelfolder = 'junnyu/roformer_chinese_sim_char_ft_base'
rulefolder = 'rules'

tokenizer = RoFormerTokenizer.from_pretrained(modelfolder)
model = RoFormerModel.from_pretrained(modelfolder)

nlp = spacy.load('zh_core_web_trf')


# def async sent2emb(sentences):
def sent2emb_async(sentences):
    """
    run sent2emb in async mode
    """
    # create new loop
    loop = asyncio.new_event_loop()
    # run async code
    asyncio.set_event_loop(loop)
    # run code
    task = loop.run_until_complete(sent2emb(sentences))
    # close loop
    loop.close()
    return task


async def sent2emb(sents):
    embls = []
    for sent in sents:
        # get summary of sent
        summarize = get_summary(sent)
        sentence_embedding = roformer_encoder(summarize)
        embls.append(sentence_embedding)
    all_embeddings = np.concatenate(embls)
    return all_embeddings


# get summary of text
def get_summary(text):
    tr4s = TextRank4Sentence()
    tr4s.analyze(text=text, lower=True, source='all_filters')
    sumls = []
    for item in tr4s.get_key_sentences(num=3):
        sumls.append(item.sentence)
    summary = ''.join(sumls)
    return summary


# Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(model_output, attention_mask):
    # First element of model_output contains all token embeddings
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(
        token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
        input_mask_expanded.sum(1), min=1e-9)


def roformer_encoder(sentences):
    # Tokenize sentences
    encoded_input = tokenizer(sentences,
                              max_length=512,
                              padding=True,
                              truncation=True,
                              return_tensors='pt')

    # Compute token embeddings
    with torch.no_grad():
        model_output = model(**encoded_input)

    # Perform pooling. In this case, max pooling.
    sentence_embeddings = mean_pooling(
        model_output, encoded_input['attention_mask']).numpy()
    return sentence_embeddings


@st.cache
def get_csvdf(rulefolder):
    files2 = glob.glob(rulefolder + '**/*.csv', recursive=True)
    dflist = []
    for filepath in files2:
        basename = os.path.basename(filepath)
        filename = os.path.splitext(basename)[0]
        newdf = rule2df(filename, filepath)[['监管要求', '结构', '条款']]
        dflist.append(newdf)
    alldf = pd.concat(dflist, axis=0)
    return alldf


def rule2df(filename, filepath):
    docdf = pd.read_csv(filepath)
    docdf['监管要求'] = filename
    return docdf


def get_embedding(folder, emblist):
    dflist = []
    for file in emblist:
        filepath = os.path.join(folder, file + '.npy')
        embeddings = np.load(filepath)
        dflist.append(embeddings)
    alldf = np.concatenate(dflist)
    return alldf


# split string by space into words, add brackets before and after words, combine into text
def split_words(text):
    words = text.split()
    words = ['(?=.*' + word + ')' for word in words]
    new = ''.join(words)
    return new


# get section list from df
def get_section_list(searchresult, make_choice):
    '''
    get section list from df
    
    args: searchresult, make_choice
    return: section_list
    '''
    df = searchresult[(searchresult['监管要求'].isin(make_choice))]
    conls = df['结构'].drop_duplicates().tolist()
    unils = []
    for con in conls:
        itemls = con.split('/')
        for item in itemls[:2]:
            unils.append(item)
    # drop duplicates and keep list order
    section_list = list(dict.fromkeys(unils))
    return section_list


# get folder name list from path
def get_folder_list(path):
    folder_list = [
        folder for folder in os.listdir(path)
        if os.path.isdir(os.path.join(path, folder))
    ]
    return folder_list


def get_rulefolder(industry_choice):
    # join folder with industry_choice
    folder = os.path.join(rulefolder, industry_choice)
    return folder


def tfidfkeyword(text, top_n=5):
    text = ' '.join(cut_sentences(text))
    tags = jieba.analyse.extract_tags(text,
                                      topK=top_n,
                                      allowPOS=('ns', 'n', 'nr', 'm', 'ns',
                                                'nt', 'nz', 't', 'q'))
    return tags


# cut text into words using spacy
def cut_sentences(text):
    # cut text into words
    doc = nlp(text)
    sents = [t.text for t in doc]
    return sents


# find similar words in doc embedding
def find_similar_words(words, doc, threshold_key=0.5, top_n=3):
    # compute similarity
    similarities = {}
    for word in words:
        tok = nlp(word)
        similarities[tok.text] = {}
        for tok_ in doc:
            similarities[tok.text].update({tok_.text: tok.similarity(tok_)})
    # sort
    topk = lambda x: {
        k: v
        for k, v in sorted(similarities[x].items(),
                           key=lambda item: item[1],
                           reverse=True)[:top_n]
    }
    result = {word: topk(word) for word in words}
    # filter by threshold
    result_filter = {
        word: {k: v
               for k, v in result[word].items() if v >= threshold_key}
        for word in result
    }
    return result_filter


# convert text spacy to word embedding
def text2emb(text):
    # cut text into words
    doc = nlp(text)
    return doc


# get similarity using keywords between two docs
def get_similar_keywords(keyls,
                               audit_list,
                               key_top_n=3,
                               threshold_key=0.5):

    audit_keywords = dict()
    for idx,audit in enumerate(audit_list):

        doc = text2emb(audit)
        result = find_similar_words(keyls, doc, threshold_key, top_n=key_top_n)
        subls = []
        for key in keyls:
            subls.append(list(result[key].keys()))
        # flatten subls
        subls = [item for sub in subls for item in sub]
        # remove duplicates
        subls = list(set(subls))
        audit_keywords[idx] = subls

        # get audit_keywords keys sorted by value length
        audit_keywords_sorted = sorted(audit_keywords.items(), key=lambda x: len(x[1]), reverse=True)
        # get keys of audit_keywords_sorted
        audit_keywords_keys = [key for key, value in audit_keywords_sorted]
    return audit_keywords_keys

# get most similar from list of sentences
def get_most_similar(keyls,audit_list, top_n=3):
    audit_list_sorted = get_similar_keywords(keyls, audit_list, key_top_n=3,threshold_key=0.5)
    return audit_list_sorted[:top_n]

# get tfidf keywords list
def get_keywords(proc_list, key_num=5):
    key_list = []
    for proc in proc_list:
        key_list.append(tfidfkeyword(proc, key_num))
    return key_list

# get exact match
def get_exect_similar(searchresult, item_text,top_num):
    # join item_text
    item_text = ' '.join(item_text)
    # split words item_text
    item_text_list = split_words(item_text)
    # print(item_text_list)
    plcsam = searchresult[ (searchresult['条款'].str.contains(item_text_list))]
    return plcsam[:top_num]