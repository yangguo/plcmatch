import streamlit as st
import pandas as pd
import ast

from analysis import get_matchplc, do_plot_match, df2list
from checkrule import searchByName, searchByItem, get_rule_data
from utils import get_folder_list, get_section_list, get_most_similar, get_keywords, get_exect_similar
from upload import save_uploadedfile, upload_data, get_uploadfiles, remove_uploadfiles, get_upload_data

rulefolder = 'rules'


def main():

    # st.subheader("制度匹配分析")
    menu = ['文件上传', '制度匹配分析']
    choice = st.sidebar.selectbox("选择", menu)
    if choice == '文件上传':
        uploaded_file_ls = st.file_uploader("选择新文件上传",
                                            type=['docx', 'pdf', 'txt'],
                                            accept_multiple_files=True,
                                            help='选择文件上传')

        for uploaded_file in uploaded_file_ls:
            if uploaded_file is not None:

                # Check File Type
                if (uploaded_file.type ==
                        "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
                    ) | (uploaded_file.type == "application/pdf") | (
                        uploaded_file.type == "text/plain"):
                    save_uploadedfile(uploaded_file)
                else:
                    st.error('不支持文件类型')
        submit = st.button('文件编码')
        if submit:
            with st.spinner('正在处理中...'):
                upload_data()
                st.success('文件编码完成')
        # display all policy
        st.write('已编码的文件：')
        uploadfilels = get_uploadfiles()
        st.write(uploadfilels)
        remove = st.button('删除已上传文件')
        if remove:
            remove_uploadfiles()
            st.success('删除成功')

    elif choice == '制度匹配分析':

        industry_list = get_folder_list(rulefolder)
        industry_choice = st.sidebar.selectbox('选择行业:', industry_list)

        if industry_choice != '':
            name_text = ''
            rule_val, rule_list = searchByName(name_text, industry_choice)

            rule_choice = st.sidebar.multiselect('选择匹配监管要求:', rule_list)

            rule_section_list = get_section_list(rule_val, rule_choice)
            rule_column_ls = st.sidebar.multiselect('选择章节:', rule_section_list)
            if rule_column_ls == []:
                column_rule = ''
            else:
                column_rule = '|'.join(rule_column_ls)

        upload_list = get_uploadfiles()
        upload_choice = st.sidebar.multiselect('选择已上传文件:', upload_list)

        if (upload_choice != []) & (rule_choice != []):
            uploaddf, upload_embeddings = get_upload_data(upload_choice)

            ruledf, rule_embeddings = get_rule_data(rule_choice,
                                                    industry_choice)

            subruledf, _ = searchByItem(ruledf, rule_choice, column_rule, '')

            # get index of rule
            rule_index = subruledf.index.tolist()
            subrule_embeddings = rule_embeddings[rule_index]

            # choose match method
            match_method = st.sidebar.radio('匹配方法选择',
                                            ('关键词匹配', '语义匹配'))

            if match_method == '关键词匹配':
                # initialize session value proc_list
                if 'proc_list' not in st.session_state:
                    st.session_state['proc_list'] = []
                # initialize session value keyword_list
                if 'keyword_list' not in st.session_state:
                    st.session_state['keyword_list'] = []

                # get proc_list
                proc_list = subruledf['条款'].tolist()
                # get length of proc_list
                proc_len = len(proc_list)

                # use expander
                with st.sidebar.expander('参数设置'):
                    # silidebar to choose key_num
                    key_num = st.slider('选择关键词数量', 1, 10, 3)
                    # get top number
                    top_num = st.slider('选择匹配结果数量', 1, 10, 3)
                    # get start index
                    start_index = st.number_input('选择开始索引', 0, proc_len-1, 0)
                    # convert to int
                    start_index = int(start_index)
                    # get end index
                    end_index = st.number_input('选择结束索引', start_index,
                                                proc_len-1, proc_len-1)
                    # convert to int
                    end_index = int(end_index)
                    # match mode
                    match_mode = st.radio('精确模式', ('精确', '模糊'))

                # get keywords button
                get_keywords_button = st.sidebar.button('获取关键词')
                if get_keywords_button:
                    # column_rule not empty
                    if column_rule != '':
                        proc_list = proc_list[start_index:end_index+1]
                        # display proc_list
                        st.write('匹配监管要求：')
                        keywords_list = get_keywords(proc_list, key_num)
                        # convert proc_list, keywords_list into dataframe and display
                        proc_df = pd.DataFrame({'条款': proc_list, '关键词': keywords_list})
                        st.table(proc_df)                       
                        # display keywords_list
                        # new_keywords_str = st.text_area(
                        #     '关键词列表更新：', keywords_list)
                        # # read literal new_keywords_str as raw string
                        # new_keywords_list = ast.literal_eval(new_keywords_str)
                        # update session value keyword_list
                        st.session_state['keyword_list'] = keywords_list
                        # update session value proc_list
                        st.session_state['proc_list'] = proc_list
                    else:
                        st.error('请选择章节')
                        keywords_list = []
                        proc_list = []
                else:
                    keywords_list = st.session_state['keyword_list']
                    proc_list = st.session_state['proc_list']
                    proc_df = pd.DataFrame({'条款': proc_list, '关键词': keywords_list})
                    if proc_df.empty:
                        st.error('请先点击获取关键词')
                    else:
                        st.table(proc_df)                       

                if keywords_list != []:
                    # display keywords_list
                    new_keywords_str = st.text_area(
                        '关键词列表更新：', keywords_list)
                    # read literal new_keywords_str as raw string
                    new_keywords_list = ast.literal_eval(new_keywords_str)
                    # update session value keyword_list
                    st.session_state['keyword_list'] = new_keywords_list
                else:
                    new_keywords_list = []
                 
                # display button
                submit = st.sidebar.button('开始匹配分析')
                if submit:
                    # # get keyword_list
                    # new_keywords_list = st.session_state['keyword_list']
                    # # get proc_list
                    # proc_list = st.session_state['proc_list']
                    st.subheader('匹配结果：')
                    if match_mode == '精确':
                        for i, (proc, keywords) in enumerate(
                                zip(proc_list, new_keywords_list)):
                            with st.spinner('正在处理中...'):

                                st.warning('序号' + str(i + 1) + ': ' + proc)
                                st.info('关键词: ' + '/'.join(keywords))

                                subuploaddf = get_exect_similar(
                                    uploaddf, keywords, top_num)
                                # display result
                                if subuploaddf.empty:
                                    st.write('没有匹配结果')
                                else:
                                    st.table(subuploaddf)
                                    st.write('-' * 20)

                    elif match_mode == '模糊':
                        audit_list = uploaddf['条款'].tolist()
                        # get keywords list

                        # display result
                        for i, (proc, keywords) in enumerate(
                                zip(proc_list, new_keywords_list)):
                            with st.spinner('正在处理中...'):

                                st.warning('序号' + str(i + 1) + ': ' + proc)
                                st.info('关键词: ' + '/'.join(keywords))

                                result = get_most_similar(
                                    keywords, audit_list, top_num)

                                # get subuploaddf based on index list
                                subuploaddf = uploaddf.loc[result]
                                # display result
                                st.table(subuploaddf)
                                st.write('-' * 20)

            elif match_method == '语义匹配':
                # use expander
                with st.sidebar.expander('参数设置'):
                    top = st.slider('匹配数量选择',
                                    min_value=1,
                                    max_value=10,
                                    value=2)

                    x = st.slider('匹配阈值选择%',
                                  min_value=0,
                                  max_value=100,
                                  value=80)
                    st.write('匹配阈值:', x / 100)
                    # review mode
                    review_mode=st.radio('审阅模式', ('否', '是'))

                if review_mode == '否':
                    querydf, query_embeddings = subruledf, subrule_embeddings
                    sentencedf, sentence_embeddings = uploaddf, upload_embeddings

                elif review_mode == '是':
                    querydf, query_embeddings = uploaddf, upload_embeddings
                    sentencedf, sentence_embeddings = subruledf, subrule_embeddings

                validdf = get_matchplc(querydf, query_embeddings, sentencedf,
                                       sentence_embeddings, top)
                combdf = pd.concat([querydf.reset_index(drop=True), validdf],
                                   axis=1)
                match = st.sidebar.radio('条款匹配分析条件', ('查看匹配条款', '查看不匹配条款'))

                if match == '查看匹配条款':
                    combdf['是否匹配'] = (combdf['匹配度'] >= x / 100).astype(int)
                else:
                    combdf['是否匹配'] = (combdf['匹配度'] < x / 100).astype(int)

                if review_mode == '否':
                    do_plot_match(combdf, match)

                sampledf = combdf.loc[
                    combdf['是否匹配'] == 1,
                    ['监管要求', '结构', '条款', '匹配条款', '匹配章节', '匹配制度', '匹配度']]

                # st.sidebar.write('监管要求: ', '/'.join(rule_choice))
                # st.sidebar.write('章节: ', column_rule)
                # st.sidebar.write('内部制度: ', '/'.join(upload_choice))

                # calculate the percentage of matched items
                matchrate = sampledf.shape[0] / combdf.shape[0]
                st.sidebar.write('匹配率:', matchrate)
                st.sidebar.write('总数:', sampledf.shape[0], '/',
                                 combdf.shape[0])

                dis1ls, dis2ls, dis3ls = df2list(sampledf)
                # enumerate each list with index
                for i, (dis1, dis2,
                        dis3) in enumerate(zip(dis1ls, dis2ls, dis3ls)):
                    st.info('序号' + str(i + 1) + ': ' + dis1)
                    st.warning(dis2)
                    st.table(dis3)
                # analysis is done
                st.sidebar.success('分析完成')
                st.sidebar.download_button(label='下载结果',
                                           file_name='内外部合规分析结果.csv',
                                           data=sampledf.to_csv(),
                                           mime='text/csv')


if __name__ == '__main__':
    main()