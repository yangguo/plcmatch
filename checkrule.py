from utils import split_words, get_csvdf, get_rulefolder, get_embedding

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
                          & (searchresult['结构'].str.contains(column_text)) &
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