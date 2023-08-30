from utils import split_words, get_csvdf, get_rulefolder, get_embedding
from gptfuc import init_supabase, industry_name_to_code
import pandas as pd

supabase = init_supabase()


def get_samplerule(key_list, industry_choice):
    rulefolder = get_rulefolder(industry_choice)
    plcdf = get_csvdf(rulefolder)
    selectdf = plcdf[plcdf['监管要求'].isin(key_list)]
    tb_sample = selectdf[['监管要求', '结构', '条款']]
    return tb_sample.reset_index(drop=True)


def searchByName(search_text, industry_choice):
    rulefolder = get_rulefolder(industry_choice)
    plcdf = get_csvdf(rulefolder)
    plc_list = plcdf['监管要求'].drop_duplicates().tolist()
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
    plcsam = searchresult[(searchresult['监管要求'].isin(make_choice))
                          & (searchresult['结构'].astype(str).str.contains(column_text)) &
                          (searchresult['条款'].str.contains(item_text_list))]
    total = len(plcsam)
    return plcsam, total


# fix section text with +
def fix_section_text(section_text):
    if '+' in section_text:
        section_text = section_text.replace('+', '\\+')
    return section_text


def get_rule_data(key_list, industry_choice):
    rulefolder = get_rulefolder(industry_choice)
    plcdf = get_csvdf(rulefolder)
    selectdf = plcdf[plcdf['监管要求'].isin(key_list)]
    emblist = selectdf['监管要求'].unique().tolist()
    rule_encode = get_embedding(rulefolder, emblist)
    return selectdf, rule_encode


def searchByNamesupa(search_text, industry_choice):
    table_name = industry_name_to_code(industry_choice)

    # print(table_name)
    # Get all records from table and cast 'metadata' to text type
    result = supabase.table(table_name).select("content, metadata").execute()

    # print(result.data)
    # Convert the results to a DataFrame
    df = pd.json_normalize(result.data)
    df.columns = ["条款", "结构", "监管要求"]
    # print(df)
    # Filter DataFrame based on conditions
    filtered_results = df[df["监管要求"].str.contains(f".*{search_text}.*")]

    choicels = filtered_results["监管要求"].unique().tolist()

    return filtered_results, choicels