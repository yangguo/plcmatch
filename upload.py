import os
import pandas as pd
import streamlit as st
import numpy as np

from utils import get_csvdf, get_embedding, sent2emb_async

import docx
import glob
import pdfplumber

uploadfolder = 'uploads'


def get_uploadfiles():
    fileslist = glob.glob(uploadfolder + '/*.csv', recursive=True)
    filenamels = []
    for filepath in fileslist:
        filename = os.path.basename(filepath)
        name = filename.split('.')[0]
        filenamels.append(name)
    return filenamels


def remove_uploadfiles():
    files = glob.glob(uploadfolder + '/*.*', recursive=True)

    for f in files:
        try:
            os.remove(f)
        except OSError as e:
            st.error("Error: %s : %s" % (f, e.strerror))


def savedf(txtlist, filename):
    df = pd.DataFrame(txtlist)
    df.columns = ['条款']
    df['制度'] = filename
    df['结构'] = df.index
    basename = filename.split('.')[0]
    savename = basename + '.csv'
    savepath = os.path.join(uploadfolder, savename)
    df.to_csv(savepath)


def txt2df(filename, filepath):
    with open(filepath) as f:
        contents = f.readlines()
    f.close()
    text = ''.join(contents)
    itemlist = text.replace(' ', '').split('\n')
    dflist = [item for item in itemlist if len(item) > 0]
    savedf(dflist, filename)


def get_txtdf():
    fileslist = glob.glob(uploadfolder + '**/*.txt', recursive=True)
    # get csv file name list
    csvfiles = get_csvfilelist()
    for filepath in fileslist:
        # get file name
        name = getfilename(filepath)
        if name not in csvfiles:
            txt2df(name, filepath)


# convert pdf to dataframe usng pdfplumber
def pdf2df_plumber(filename, filepath):
    result = ''
    with pdfplumber.open(filepath) as pdf:
        for page in pdf.pages:
            txt = page.extract_text()
            if txt != '':
                result += txt
    dflist = result.replace('\x0c', '').replace('\n', '').split('。')
    savedf(dflist, filename)
 

def get_pdfdf():
    fileslist = glob.glob(uploadfolder + '**/*.pdf', recursive=True)
    # get csv file name list
    csvfiles = get_csvfilelist()

    for filepath in fileslist:
        # get file name
        name = getfilename(filepath)
        # if name not in csvfiles
        if name not in csvfiles:
           pdf2df_plumber(name, filepath)


def doc2df(filename, filepath):
    # open connection to Word Document
    doc = docx.Document(filepath)
    # get all paragraphs in the document
    dflist = []
    for para in doc.paragraphs:
        txt = para.text
        if txt != '':
            dflist.append(txt)
    savedf(dflist, filename)
 

def get_docdf():
    fileslist = glob.glob(uploadfolder + '**/*.docx', recursive=True)
    # get csv file name list
    csvfiles = get_csvfilelist()
    for filepath in fileslist:
        # get file name
        name = getfilename(filepath)
        # if name not in csvfiles
        if name not in csvfiles:
            doc2df(name, filepath)


# return corpus_embeddings
def getfilename(file):
    filename = os.path.basename(file)
    name = filename.split('.')[0]
    return name


def file2embedding(file):
    df = pd.read_csv(file)
    sentences = df['条款'].tolist()
    # all_embeddings = sent2emb(sentences)
    # use async to get embeddings
    all_embeddings = sent2emb_async(sentences)
    name = getfilename(file)
    savename = name + '.npy'
    savepath = os.path.join(uploadfolder, savename)
    np.save(savepath, all_embeddings)


def encode_plclist():
    files = glob.glob(uploadfolder + '**/*.csv', recursive=True)
    # get npy file name list
    npyfiles = get_npyfilelist()
    for file in files:
        # get file name
        name = getfilename(file)
        # check if file is not in npy file list
        if name not in npyfiles:
            try:
                file2embedding(file)
            except Exception as e:
                st.error(str(e))


# get npy file name list
def get_npyfilelist():
    files2 = glob.glob(uploadfolder + '**/*.npy', recursive=True)
    filelist = []
    for file in files2:
        name = getfilename(file)
        filelist.append(name)
    return filelist


# get csv file name list
def get_csvfilelist():
    files2 = glob.glob(uploadfolder + '**/*.csv', recursive=True)
    filelist = []
    for file in files2:
        name = getfilename(file)
        filelist.append(name)
    return filelist


def get_upload_data(plc_list):
    plcdf = get_csvdf(uploadfolder)
    selectdf = plcdf[plcdf['监管要求'].isin(plc_list)]
    emblist = selectdf['监管要求'].unique().tolist()
    plc_encode = get_embedding(uploadfolder, emblist)
    return selectdf, plc_encode


def upload_data():
    try:
        get_docdf()
        get_txtdf()
        get_pdfdf()
        encode_plclist()
    except Exception as e:
        st.error(str(e))


def save_uploadedfile(uploadedfile):
    with open(os.path.join(uploadfolder, uploadedfile.name), "wb") as f:
        f.write(uploadedfile.getbuffer())
    return st.success("上传文件:{} 成功。".format(uploadedfile.name))
