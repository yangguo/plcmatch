import pandas as pd
import scipy.spatial
import plotly.express as px
import streamlit as st


def get_matchplc(querydf, query_embeddings, sentencedf, sentence_embeddings,
                 top):
    queries = querydf['条款'].tolist()
    sentences = sentencedf['条款'].tolist()
    resdata = {}
    querylist = []
    answerlist = []
    policylist = []
    filelist = []
    avglist = []
    number_top_matches = top
    for query, query_embedding in zip(queries, query_embeddings):
        distances = scipy.spatial.distance.cdist([query_embedding],
                                                 sentence_embeddings,
                                                 "cosine")[0]

        results = zip(range(len(distances)), distances)
        results = sorted(results, key=lambda x: x[1])

        querylist.append(query)

        itemlist = []
        plclist = []
        filist = []
        avg = 0
        for idx, distance in results[0:number_top_matches]:
            ansstr = sentences[idx].strip()
            plcstr = sentencedf.iloc[idx]['结构']
            filestr = sentencedf.iloc[idx]['监管要求']
            itemlist.append(ansstr)
            plclist.append(plcstr)
            filist.append(filestr)
            avg += (1 - distance)
        answerlist.append(itemlist)
        policylist.append(plclist)
        filelist.append(filist)
        avglist.append(avg / number_top_matches)
    resdata['query'] = querylist
    resdata['匹配条款'] = answerlist
    resdata['匹配章节'] = policylist
    resdata['匹配制度'] = filelist
    resdata['匹配度'] = avglist
    resdf = pd.DataFrame(resdata)
    return resdf


def do_plot_match(combdf, title):

    combdf['章节'] = combdf['结构'].astype(str).str.extract('(.*?)/')
    chartdb = combdf.groupby('章节',
                             sort=False)['是否匹配'].mean().reset_index(name='均值')

    # change the color of plotly diagram
    fig = px.bar(
        chartdb,
        x='章节',
        y='均值',
        color='均值',
        hover_data=['章节', '均值'],
        range_y=[0, 1],
        )

    fig.update_layout(
        title=title,
        xaxis_title='章节',
        yaxis_title='均值',
        )
    st.plotly_chart(fig)


def df2list(df):
    # dis1
    dis1 = []
    dis2 = []
    dfdis1 = df[['监管要求', '结构', '条款', '匹配度']]
    # conver each df row to df list
    for index, row in dfdis1.iterrows():
        # conver row to str
        rowstr1 = str(row[0]) + ' /' + str(row[1])
        rowstr2 = ' 条款:' + str(row[2]) + ' [匹配度:' + str(row[3]) + ']'
        dis1.append(rowstr1)
        dis2.append(rowstr2)
    # dis3
    plcls = df['匹配条款'].tolist()
    sectionls = df['匹配章节'].tolist()
    filels = df['匹配制度'].tolist()

    dis3 = []
    for plc, section, file in zip(plcls, sectionls, filels):
        content = {}
        content['匹配制度'] = file
        content['匹配章节'] = section
        content['匹配条款'] = plc
        df = pd.DataFrame(content)
        dis3.append(df)

    return dis1, dis2, dis3
