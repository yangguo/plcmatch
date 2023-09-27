import codecs
import json
from ast import literal_eval

import numpy as np
import pandas as pd
import streamlit as st

from gptfuc import convert_list_to_filter, industry_name_to_code, init_supabase
from utils import get_csvdf, get_rulefolder, split_words

supabase = init_supabase()


def get_samplerule(key_list, industry_choice):
    rulefolder = get_rulefolder(industry_choice)
    plcdf = get_csvdf(rulefolder)
    selectdf = plcdf[plcdf["监管要求"].isin(key_list)]
    tb_sample = selectdf[["监管要求", "结构", "条款"]]
    return tb_sample.reset_index(drop=True)


def searchByName(search_text, industry_choice):
    rulefolder = get_rulefolder(industry_choice)
    plcdf = get_csvdf(rulefolder)
    plc_list = plcdf["监管要求"].drop_duplicates().tolist()
    choicels = []
    for plc in plc_list:
        if search_text in plc:
            choicels.append(plc)
    plcsam = get_samplerule(choicels, industry_choice)
    return plcsam, choicels


def searchByItem(searchresult, make_choice, column_text, item_text):
    # split words item_text
    item_text_list = split_words(item_text)
    column_text = fix_section_text(column_text)
    plcsam = searchresult[
        (searchresult["监管要求"].isin(make_choice))
        & (searchresult["结构"].astype(str).str.contains(column_text))
        & (searchresult["条款"].str.contains(item_text_list))
    ]
    total = len(plcsam)
    return plcsam, total


# fix section text with +
def fix_section_text(section_text):
    if "+" in section_text:
        section_text = section_text.replace("+", "\\+")
    return section_text


# def get_rule_data(key_list, industry_choice):
#     rulefolder = get_rulefolder(industry_choice)
#     plcdf = get_csvdf(rulefolder)
#     selectdf = plcdf[plcdf['监管要求'].isin(key_list)]
#     emblist = selectdf['监管要求'].unique().tolist()
#     rule_encode = get_embedding(rulefolder, emblist)
#     return selectdf, rule_encode


@st.cache_data
def searchByNamesupa(rule_choice, industry_choice):
    table_name = industry_name_to_code(industry_choice)

    filter = convert_list_to_filter(rule_choice)
    # print(filter)
    filter_value = json.dumps(filter, ensure_ascii=False)
    # print(filter_value)
    # Get all records from table and cast 'metadata' to text type
    result = (
        supabase.table(table_name)
        .select("content, metadata, embedding")
        .filter("metadata", "cs", filter_value)
        .execute()
    )

    # print(result.data)
    # Convert the results to a DataFrame
    df = pd.json_normalize(result.data)
    df.columns = ["条款", "embedding", "结构", "监管要求"]
    # get plcdf and embedding
    plcdf = df[["监管要求", "结构", "条款"]]
    # Convert string representation of lists to actual lists
    df["embedding"] = df["embedding"].apply(literal_eval)
    embls = df["embedding"].tolist()
    # convert to numpy array
    rule_encode_np = np.array(embls)
    # check shape
    # print(rule_encode_np.shape)
    # reshape to 2-dimensional
    # rule_encode_np = rule_encode_np.reshape((rule_encode_np.shape[0], -1))
    # print(rule_encode_np.shape)
    return plcdf, rule_encode_np


@st.cache_data
def searchByIndustrysupa(industry_choice):
    table_name = industry_name_to_code(industry_choice)

    metadata_name = table_name + "_metadata"
    # Get all records from table and cast 'metadata' to text type
    result = supabase.table(metadata_name).select("plc_value").execute()
    # Convert JSON data to list
    converted_list = [item["plc_value"] for item in result.data]
    return converted_list


def encode_utf8(string):
    return codecs.encode(string, "utf-8")
