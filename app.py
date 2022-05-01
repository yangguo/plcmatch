import streamlit as st
import pandas as pd
import ast

from analysis import get_matchplc, do_plot_match, df2list
from checkrule import searchByName, searchByItem, get_rule_data
from utils import get_folder_list, get_section_list, get_most_similar, get_keywords, get_exect_similar,df2aggrid
from upload import save_uploadedfile, upload_data, get_uploadfiles, remove_uploadfiles, get_upload_data,savedf
# Import for dyanmic tagging
from streamlit_tags import st_tags, st_tags_sidebar

rulefolder = 'rules'


def main():

    # st.subheader("制度匹配分析")
    menu = ['文件上传','文件选择', '匹配分析']
    choice = st.sidebar.selectbox("选择", menu)

    # initialize session value file1df, file1_embeddings
    if 'file1df' not in st.session_state:
        st.session_state['file1df'] = None
    if 'file1_embeddings' not in st.session_state:
        st.session_state['file1_embeddings'] = None
    if 'file1_industry' not in st.session_state:
        st.session_state['file1_industry'] = ''
    if 'file1_rulechoice' not in st.session_state:
        st.session_state['file1_rulechoice'] = []
    if 'file1_filetype' not in st.session_state:
        st.session_state['file1_filetype'] = ''
    if 'file1_section_list' not in st.session_state:
        st.session_state['file1_section_list'] = []
    # initialize session value file2df, file2_embeddings
    if 'file2df' not in st.session_state:
        st.session_state['file2df'] = None
    if 'file2_embeddings' not in st.session_state:
        st.session_state['file2_embeddings'] = None
    if 'file2_industry' not in st.session_state:
        st.session_state['file2_industry'] = ''
    if 'file2_rulechoice' not in st.session_state:
        st.session_state['file2_rulechoice'] = []
    if 'file2_filetype' not in st.session_state:
        st.session_state['file2_filetype'] = ''
    if 'file2_section_list' not in st.session_state:
        st.session_state['file2_section_list'] = []

    if choice == '文件上传':
        uploaded_file_ls = st.file_uploader("选择新文件上传",
                                            type=['docx', 'pdf', 'txt','xlsx'],
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

                            # if upload file is xlsx
                elif uploaded_file.type == 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet':
                    # get sheet names list from excel file
                    xls = pd.ExcelFile(uploaded_file)
                    sheets = xls.sheet_names
                    # choose sheet name and click button
                    sheet_name = st.selectbox('选择表单', sheets)

                    # choose header row
                    header_row = st.number_input('选择表头行',
                                                min_value=0,
                                                max_value=10,
                                                value=0,key='header_row')
                    df = pd.read_excel(uploaded_file,
                                    header=header_row,
                                    sheet_name=sheet_name)
                    # filllna
                    df = df.fillna('')
                    # display the first five rows
                    st.write(df.astype(str))

                    # get df columns
                    cols = df.columns
                    # choose proc_text and audit_text column
                    proc_col = st.sidebar.selectbox('选择文本列', cols)
                                                   
                    # get proc_text and audit_text list
                    proc_list = df[proc_col].tolist()

                    # get proc_list and audit_list length
                    proc_len = len(proc_list)

                    # if proc_list or audit_list is empty or not equal
                    if proc_len == 0:
                        st.error('文本列为空，请重新选择')
                        return
                    else:
                        # choose start and end index
                        start_idx = st.sidebar.number_input('选择开始索引',
                                                            min_value=0,
                                                            max_value=proc_len - 1,
                                                            value=0)
                        end_idx = st.sidebar.number_input('选择结束索引',
                                                        min_value=start_idx,
                                                        max_value=proc_len - 1,
                                                        value=proc_len - 1)
                        # get proc_list and audit_list
                        subproc_list = proc_list[start_idx:end_idx + 1]
                        # get basename of uploaded file
                        basename = uploaded_file.name.split('.')[0]
                        # save subproc_list to file using upload
                        savedf(subproc_list, basename)

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

    elif choice == '文件选择':
        # choose radio for file1 or file2
        file_choice = st.sidebar.radio('选择文件', ['选择文件1', '选择文件2'])

        if file_choice == '选择文件1':       
            # get current file1 value from session
            industry_choice = st.session_state['file1_industry']
            rule_choice = st.session_state['file1_rulechoice']
            filetype_choice = st.session_state['file1_filetype']
            section_choice = st.session_state['file1_section_list']
        elif file_choice == '选择文件2':
            # get current file2 value from session
            industry_choice = st.session_state['file2_industry']
            rule_choice = st.session_state['file2_rulechoice']
            filetype_choice = st.session_state['file2_filetype']
            section_choice = st.session_state['file2_section_list']
        # get preselected filetype index
        file_typels=['外部制度', '已上传文件']
        if filetype_choice == '':
            filetype_index = 0
        else:
            filetype_index = file_typels.index(filetype_choice)
        # choose file type
        file_type = st.sidebar.selectbox('选择文件类型', ['外部制度', '已上传文件'], index=filetype_index,)
        if file_type == '外部制度':
            industry_list = get_folder_list(rulefolder)
            # get preselected industry index
            if industry_choice == '':
                industry_index = 0
                rule_choice = []
            else:
                industry_index = industry_list.index(industry_choice)
            industry_choice = st.sidebar.selectbox('选择行业:', industry_list, index=industry_index)
            
            name_text = ''
            rule_val, rule_list = searchByName(name_text, industry_choice)
            rule_choice = st.sidebar.multiselect('选择匹配监管要求:', rule_list, rule_choice)
            rule_section_list = get_section_list(rule_val, rule_choice)
            rule_column_ls = st.sidebar.multiselect('选择章节:', rule_section_list,section_choice)
            if rule_column_ls == []:
                column_rule = ''
            else:
                column_rule = '|'.join(rule_column_ls)

            if rule_choice != []:
                ruledf, rule_embeddings = get_rule_data(rule_choice,
                                                industry_choice)
                choosedf, _ = searchByItem(ruledf, rule_choice, column_rule, '')
                # get index of rule
                rule_index = choosedf.index.tolist()
                choose_embeddings = rule_embeddings[rule_index]
            else:
                choosedf, choose_embeddings = None, None

        elif file_type == '已上传文件':
            if industry_choice != '':
                rule_choice=[]
            upload_list = get_uploadfiles()
            upload_choice = st.sidebar.multiselect('选择已上传文件:', upload_list,rule_choice)
            if upload_choice != []:
                choosedf, choose_embeddings = get_upload_data(upload_choice)
            else:
                choosedf, choose_embeddings = None, None
            industry_choice = ''
            rule_choice = upload_choice
            rule_column_ls = []

        # file choose button
        file_button = st.sidebar.button('选择文件')
        if file_button:
            if file_choice == '选择文件1':
                file1df, file1_embeddings = choosedf, choose_embeddings
                file1_industry = industry_choice
                file1_rulechoice = rule_choice
                file1_filetype = file_type
                file1_section_list = rule_column_ls
                st.session_state['file1df'] = file1df
                st.session_state['file1_embeddings'] = file1_embeddings
                st.session_state['file1_industry'] = file1_industry
                st.session_state['file1_rulechoice'] = file1_rulechoice
                st.session_state['file1_filetype'] = file1_filetype
                st.session_state['file1_section_list'] = file1_section_list
                file2df= st.session_state['file2df']
                file2_embeddings = st.session_state['file2_embeddings']
                file2_industry = st.session_state['file2_industry']
                file2_rulechoice = st.session_state['file2_rulechoice']
                file2_filetype = st.session_state['file2_filetype']
                file2_section_list = st.session_state['file2_section_list']
            elif file_choice == '选择文件2':
                file2df, file2_embeddings = choosedf, choose_embeddings
                file2_industry = industry_choice
                file2_rulechoice = rule_choice
                file2_filetype = file_type
                file2_section_list = rule_column_ls
                st.session_state['file2df'] = file2df
                st.session_state['file2_embeddings'] = file2_embeddings
                st.session_state['file2_industry'] = file2_industry
                st.session_state['file2_rulechoice'] = file2_rulechoice
                st.session_state['file2_filetype'] = file2_filetype
                st.session_state['file2_section_list'] = file2_section_list
                file1df = st.session_state['file1df']
                file1_embeddings = st.session_state['file1_embeddings']
                file1_industry = st.session_state['file1_industry']
                file1_rulechoice = st.session_state['file1_rulechoice']
                file1_filetype = st.session_state['file1_filetype']
                file1_section_list = st.session_state['file1_section_list']
        else:
            file1df= st.session_state['file1df']
            file2df= st.session_state['file2df']
            file1_embeddings = st.session_state['file1_embeddings']
            file2_embeddings = st.session_state['file2_embeddings']
            file1_industry = st.session_state['file1_industry']
            file2_industry = st.session_state['file2_industry']
            file1_rulechoice = st.session_state['file1_rulechoice']
            file2_rulechoice = st.session_state['file2_rulechoice']
            file1_filetype = st.session_state['file1_filetype']
            file2_filetype = st.session_state['file2_filetype']
            file1_section_list = st.session_state['file1_section_list']
            file2_section_list = st.session_state['file2_section_list']

        st.subheader('已选择的文件1：')
        # display file1 rulechoice
        if file1_rulechoice != []:
            # convert to string
            file1_rulechoice_str = '| '.join(file1_rulechoice)
            # display string
            st.warning('文件1：'+file1_rulechoice_str)
        else:
            st.error('文件1：无')
        # display file1 section
        if file1_section_list != []:
            # convert to string
            file1_section_str = '| '.join(file1_section_list)
            # display string
            st.info('章节1：'+file1_section_str)
        else:
            st.info('章节1：全部')

        st.subheader('已选择的文件2：')
        # display file2 rulechoice
        if file2_rulechoice != []:
            # convert to string
            file2_rulechoice_str = '| '.join(file2_rulechoice)
            # display string
            st.warning('文件2：'+file2_rulechoice_str)
        else:   
            st.error('文件2：无')

        # display file2 section
        if file2_section_list != []:
            # convert to string
            file2_section_str = '| '.join(file2_section_list)
            # display string
            st.info('章节2：'+file2_section_str)
        else:
            st.info('章节2：全部')

    elif choice == '匹配分析':
        file1df= st.session_state['file1df']
        file2df= st.session_state['file2df']
        file1_embeddings = st.session_state['file1_embeddings']
        file2_embeddings = st.session_state['file2_embeddings']
        file1_rulechoice = st.session_state['file1_rulechoice']
        file2_rulechoice = st.session_state['file2_rulechoice']
        file1_section_list = st.session_state['file1_section_list']
        file2_section_list = st.session_state['file2_section_list']

        st.subheader('已选择的文件：')
        # display file1 rulechoice
        if file1_rulechoice != []:
            # convert to string
            file1_rulechoice_str = '| '.join(file1_rulechoice)
            # display string
            st.warning('文件1：'+file1_rulechoice_str)
        else:
            st.error('文件1：无')
        # display file1 section
        if file1_section_list != []:
            # convert to string
            file1_section_str = '| '.join(file1_section_list)
            # display string
            st.info('章节1：'+file1_section_str)
        else:
            st.info('章节1：全部')

        # display file2 rulechoice
        if file2_rulechoice != []:
            # convert to string
            file2_rulechoice_str = '| '.join(file2_rulechoice)
            # display string
            st.warning('文件2：'+file2_rulechoice_str)
        else:   
            st.error('文件2：无')

        # display file2 section
        if file2_section_list != []:
            # convert to string
            file2_section_str = '| '.join(file2_section_list)
            # display string
            st.info('章节2：'+file2_section_str)
        else:
            st.info('章节2：全部')

        # if file1df is None
        if file1df is None:
            st.error('请选择文件1')
            return
        if file2df is None:
            st.error('请选择文件2')
            return
  
        match_method = st.sidebar.radio('匹配方法选择',
                                        ( '语义匹配','关键词匹配'))
        subruledf = file1df
        subrule_embeddings = file1_embeddings
        uploaddf= file2df
        upload_embeddings = file2_embeddings

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
                start_index = st.number_input('选择开始索引', 0, proc_len-1, 0,key='start_index')
                # convert to int
                start_index = int(start_index)
                # get end index
                end_index = st.number_input('选择结束索引', start_index,
                                            proc_len-1, proc_len-1,key='end_index')
                # convert to int
                end_index = int(end_index)
                # match mode
                match_mode = st.radio('精确模式', ('精确', '模糊'))
            st.subheader('关键词分析')
            # get keywords button
            get_keywords_button = st.sidebar.button('获取关键词')
            if get_keywords_button:
                proc_list = proc_list[start_index:end_index+1]
                keywords_list = get_keywords(proc_list, key_num) 
                # update session value keyword_list
                st.session_state['keyword_list'] = keywords_list
                # update session value proc_list
                st.session_state['proc_list'] = proc_list
            else:
                keywords_list = st.session_state['keyword_list']
                proc_list = st.session_state['proc_list']
            
            proc_df = pd.DataFrame({'条款': proc_list, '关键词': keywords_list})
            # add index new column
            proc_df['序号'] = proc_df.index
            
            if proc_df.empty:
                st.error('请先点击获取关键词')
                st.stop()

            # st.table(proc_df) 
            select_df=df2aggrid(proc_df)  
            selected_rows = select_df["selected_rows"]
            if selected_rows==[]:
                st.error('请先选择查看的条款')
                st.stop()
            # get proc
            select_proc = selected_rows[0]['条款']
            # get keyword
            select_keyword = selected_rows[0]['关键词']
            # convert to list
            select_keyword =  ast.literal_eval(select_keyword)
            # get index
            select_index = selected_rows[0]['序号']
            # update keyword_list
            keyword_update = st_tags(
                label='### 关键词更新',
                text='按回车键添加关键词',
                value=select_keyword,
                suggestions=select_keyword,
                maxtags = key_num
                )
            # display select_proc
            st.write('选择的条款：'+select_proc)
            # convert list to string
            select_keyword_str = '| '.join(select_keyword)
            # display select_keyword
            st.write('原关键词列表：'+select_keyword_str)
            
            # add update keyword button
            update_keyword_button = st.button('更新关键词')
            if update_keyword_button:
                # update keywords_list by index
                keywords_list[select_index] = keyword_update
                # update session value keyword_list
                st.session_state['keyword_list'] = keywords_list
                # rerun page
                st.experimental_rerun()

            # display button
            submit = st.sidebar.button('开始匹配分析')
            if submit:
                # # get keyword_list
                new_keywords_list = st.session_state['keyword_list']
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
                combdf['是否匹配'] = (combdf['平均匹配度'] >= x / 100).astype(int)
            else:
                combdf['是否匹配'] = (combdf['平均匹配度'] < x / 100).astype(int)

            if review_mode == '否':
                do_plot_match(combdf, match)

            sampledf = combdf.loc[
                combdf['是否匹配'] == 1,
                ['监管要求', '结构', '条款', '匹配条款', '匹配章节', '匹配制度', '匹配度','平均匹配度']]


            # calculate the percentage of matched items
            matchrate = sampledf.shape[0] / combdf.shape[0]
            # format the percentage
            matchrate = '{:.2%}'.format(matchrate)
            st.sidebar.metric('匹配率:', matchrate)
            # total number of matched items
            totalstr=str(sampledf.shape[0])+'/'+str(combdf.shape[0])
            st.sidebar.metric('匹配条款总数:', totalstr)

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