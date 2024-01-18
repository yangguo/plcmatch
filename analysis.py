# import plotly.express as px
import streamlit as st

# from checkrule import searchByIndustrysupa, searchByItem, searchByNamesupa
from gptfuc import get_matchplc

# import scipy.spatial


# def get_matchplc(
#     querydf, query_embeddings, sentencedf, sentence_embeddings, top, threshold
# ):
#     queries = querydf["条款"].tolist()
#     sentences = sentencedf["条款"].tolist()
#     resdata = {}
#     querylist = []
#     answerlist = []
#     policylist = []
#     filelist = []
#     distlist = []
#     statuslist = []
#     number_top_matches = top
#     for query, query_embedding in zip(queries, query_embeddings):
#         distances = scipy.spatial.distance.cdist(
#             [query_embedding], sentence_embeddings, "cosine"
#         )[0]

#         results = zip(range(len(distances)), distances)
#         results = sorted(results, key=lambda x: x[1])

#         querylist.append(query)

#         itemlist = []
#         plclist = []
#         filist = []
#         disls = []
#         stals = []
#         # avg = 0
#         for idx, distance in results[0:number_top_matches]:
#             ansstr = sentences[idx].strip()
#             plcstr = sentencedf.iloc[idx]["结构"]
#             filestr = sentencedf.iloc[idx]["监管要求"]
#             itemlist.append(ansstr)
#             plclist.append(plcstr)
#             filist.append(filestr)
#             disls.append(1 - distance)
#             # avg += (1 - distance)
#             if (1 - distance) >= threshold:
#                 stals.append(1)
#             else:
#                 stals.append(0)
#         answerlist.append(itemlist)
#         policylist.append(plclist)
#         filelist.append(filist)
#         distlist.append(disls)
#         # avglist.append(avg / number_top_matches)
#         statuslist.append(stals)

#     resdata["query"] = querylist
#     resdata["匹配条款"] = answerlist
#     resdata["匹配章节"] = policylist
#     resdata["匹配制度"] = filelist
#     resdata["匹配度"] = distlist
#     # resdata['平均匹配度'] = avglist
#     resdata["匹配状态"] = statuslist
#     resdf = pd.DataFrame(resdata)
#     return resdf


# perform matchplc batch
def get_matchplc_batch(
    query_list, file2_filetype, file2_industry, file2_rulechoice, top, model_name
):

    answerls = []
    sourcels = []
    for query in query_list:
        answer, source = get_matchplc(
            query, file2_filetype, file2_industry, file2_rulechoice, model_name
        )
        answerls.append(answer)
        sourcels.append(source)

    return answerls, sourcels


# def do_plot_match(combdf, title):

#     combdf["章节"] = combdf["结构"].astype(str).str.extract("([^\/]*)\/?")
#     chartdb = combdf.groupby("章节", sort=False)["是否匹配"].mean().reset_index(name="均值")

#     # change the color of plotly diagram
#     fig = px.bar(
#         chartdb,
#         x="章节",
#         y="均值",
#         color="均值",
#         hover_data=["章节", "均值"],
#         range_y=[0, 1],
#     )

#     fig.update_layout(
#         title=title,
#         xaxis_title="章节",
#         yaxis_title="均值",
#     )
#     st.plotly_chart(fig)


def do_plot_match(combdf, title):

    combdf["章节"] = combdf["结构"].astype(str).str.extract(r"([^\/]*)\/?")
    chartdb = combdf.groupby("章节", sort=False)["是否匹配"].mean().reset_index(name="均值")
    # Use Streamlit's native charting function
    st.bar_chart(chartdb, x="章节", y="均值", color="均值")


# def df2list(df, flag):
#     # dis1
#     dis1 = []
#     dis2 = []
#     dfdis1 = df[["监管要求", "结构", "条款"]]
#     # conver each df row to df list
#     for index, row in dfdis1.iterrows():
#         # conver row to str
#         rowstr1 = str(row[0]) + " /" + str(row[1])
#         rowstr2 = " 条款:" + str(row[2])  # + ' [平均匹配度:' + str(row[3]) + ']'
#         dis1.append(rowstr1)
#         dis2.append(rowstr2)
#     # dis3
#     plcls = df["匹配条款"].tolist()
#     sectionls = df["匹配章节"].tolist()
#     filels = df["匹配制度"].tolist()
#     disls = df["匹配度"].tolist()
#     stals = df["匹配状态"].tolist()

#     dis3 = []
#     for plc, section, file, dist, status in zip(plcls, sectionls, filels, disls, stals):
#         content = {}
#         subfilels = []
#         subsectionls = []
#         subplcls = []
#         subdistls = []
#         if flag == 0:
#             # reverse status
#             status = [not i for i in status]
#         for subplc, subsection, subfile, subdist, substatus in zip(
#             plc, section, file, dist, status
#         ):
#             if substatus:
#                 subfilels.append(subfile)
#                 subsectionls.append(subsection)
#                 subplcls.append(subplc)
#                 subdistls.append(subdist)
#         content["匹配制度"] = subfilels
#         content["匹配章节"] = subsectionls
#         content["匹配条款"] = subplcls
#         content["匹配度"] = subdistls
#         df = pd.DataFrame(content)
#         dis3.append(df)

#     return dis1, dis2, dis3
