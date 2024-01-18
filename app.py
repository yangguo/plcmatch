# import ast
import io

import pandas as pd
import streamlit as st

from analysis import get_matchplc_batch

# from checkaudit import get_sampleaudit, searchauditByItem, searchauditByName
from checkrule import searchByIndustrysupa, searchByItem, searchByNamesupa
from gptfuc import (  # add_to_index,
    build_index,
    convert_index_to_df,
    gpt_vectoranswer,
    searchupload,
)
from upload import (  # searchupload,; upload_data,; get_upload_data,
    add_upload_folder,
    copy_files,
    get_uploadfiles,
    remove_uploadfiles,
    save_uploadedfile,
    savedf,
)
from utils import get_folder_list

# import os


# Import for dyanmic tagging
# from streamlit_tags import st_tags, st_tags_sidebar


rulefolder = "data/rules"
auditfolder = "data/audit"
uploadfolder = "uploads"
filerawfolder = "fileraw"


def main():

    # st.subheader("制度匹配分析")
    menu = ["文件上传", "文件选择", "匹配分析", "文件浏览", "文件问答"]
    choice = st.sidebar.selectbox("选择", menu)

    # initialize session value file1df, file1_embeddings
    # if "file1df" not in st.session_state:
    #     st.session_state["file1df"] = None
    # if "file1_embeddings" not in st.session_state:
    #     st.session_state["file1_embeddings"] = None
    if "file1_industry" not in st.session_state:
        st.session_state["file1_industry"] = ""
    if "file1_rulechoice" not in st.session_state:
        st.session_state["file1_rulechoice"] = []
    if "file1_filetype" not in st.session_state:
        st.session_state["file1_filetype"] = ""
    if "file1_section_list" not in st.session_state:
        st.session_state["file1_section_list"] = []
    # initialize session value file2df, file2_embeddings
    # if "file2df" not in st.session_state:
    #     st.session_state["file2df"] = None
    # if "file2_embeddings" not in st.session_state:
    #     st.session_state["file2_embeddings"] = None
    if "file2_industry" not in st.session_state:
        st.session_state["file2_industry"] = ""
    if "file2_rulechoice" not in st.session_state:
        st.session_state["file2_rulechoice"] = []
    if "file2_filetype" not in st.session_state:
        st.session_state["file2_filetype"] = ""
    if "file2_section_list" not in st.session_state:
        st.session_state["file2_section_list"] = []

    if choice == "文件上传":
        st.subheader("文件上传")
        # choose input method of manual or upload file
        input_method = st.sidebar.radio("文件上传方式", ("手动输入", "上传文件"))

        if input_method == "手动输入":
            file_name = st.text_input("请输入文件名称")
            file_text = st.text_area("请输入文件内容")
            # save txt file as bytesio
            if file_name != "" and file_text != "":
                file_bytes = bytes(file_text, encoding="utf8")
                file_io = io.BytesIO(file_bytes)
                file_io.name = file_name + ".txt"
                # save button
                filesave = st.button("保存文件")
                if filesave:
                    save_uploadedfile(file_io)
                    # st.success('文件保存成功')
            else:
                st.error("请输入文件名称和内容")

        elif input_method == "上传文件":
            uploaded_file_ls = st.file_uploader(
                "选择新文件上传",
                type=["docx", "pdf", "txt", "xlsx"],
                accept_multiple_files=True,
                help="选择文件上传",
            )

            for uploaded_file in uploaded_file_ls:
                if uploaded_file is not None:

                    # Check File Type
                    if (
                        (
                            uploaded_file.type
                            == "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
                        )
                        | (uploaded_file.type == "application/pdf")
                        | (uploaded_file.type == "text/plain")
                    ):
                        save_uploadedfile(uploaded_file)

                        # if upload file is xlsx
                    elif (
                        uploaded_file.type
                        == "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    ):
                        # get sheet names list from excel file
                        xls = pd.ExcelFile(uploaded_file)
                        sheets = xls.sheet_names
                        # choose sheet name and click button
                        sheet_name = st.selectbox("选择表单", sheets)

                        # choose header row
                        header_row = st.number_input(
                            "选择表头行",
                            min_value=0,
                            max_value=10,
                            value=0,
                            key="header_row",
                        )
                        df = pd.read_excel(
                            uploaded_file, header=header_row, sheet_name=sheet_name
                        )
                        # filllna
                        df = df.fillna("")
                        # display the first five rows
                        st.write(df.astype(str))

                        # get df columns
                        cols = df.columns
                        # choose proc_text and audit_text column
                        proc_col = st.sidebar.selectbox("选择文本列", cols)

                        # get proc_text and audit_text list
                        proc_list = df[proc_col].tolist()

                        # get proc_list and audit_list length
                        proc_len = len(proc_list)

                        # if proc_list or audit_list is empty or not equal
                        if proc_len == 0:
                            st.error("文本列为空，请重新选择")
                            return
                        else:
                            # choose start and end index
                            start_idx = st.sidebar.number_input(
                                "选择开始索引", min_value=0, max_value=proc_len - 1, value=0
                            )
                            end_idx = st.sidebar.number_input(
                                "选择结束索引",
                                min_value=start_idx,
                                max_value=proc_len - 1,
                                value=proc_len - 1,
                            )
                            # get proc_list and audit_list
                            subproc_list = proc_list[start_idx : end_idx + 1]
                            # get basename of uploaded file
                            basename = uploaded_file.name.split(".")[0]
                            # save subproc_list to file using upload
                            savedf(subproc_list, basename)

                    else:
                        st.error("不支持文件类型")

        upload_list = get_uploadfiles(uploadfolder)
        upload_choice = st.sidebar.multiselect("选择已上传文件:", upload_list, [])

        if upload_choice == []:
            st.error("请选择文件")
            # return

        file_list = upload_choice

        # file choose button
        file_button = st.sidebar.button("选择待编码文件")
        if file_button:
            # copy file_list to filerawfolder
            copy_files(file_list, uploadfolder, filerawfolder)

        # enbedding button
        embedding = st.sidebar.button("重新生成模型")
        if embedding:
            # with st.spinner("正在生成问答模型..."):
            # generate embeddings
            try:
                build_index()
                st.success("问答模型生成完成")
            except Exception as e:
                st.error(e)
                st.error("问答模型生成失败，请检查文件格式")

        # add documnet button
        # add_doc = st.sidebar.button("模型添加文档")
        # if add_doc:
        #     # with st.spinner("正在添加文档..."):
        #     # generate embeddings
        #     try:
        #         add_to_index()
        #         st.success("文档添加完成")
        #     except Exception as e:
        #         st.error(e)
        #         st.error("文档添加失败，请检查文件格式")
        remove = st.button("删除待编码的文件")
        if remove:
            remove_uploadfiles(filerawfolder)
            st.success("删除成功")

        # submit = st.sidebar.button("文件编码")
        # if submit:
        #     with st.spinner("正在处理中..."):
        #         upload_data()
        #         st.success("文件编码完成")
        # display all policy
        st.write("已编码的文件：")
        uploadfilels = get_uploadfiles(filerawfolder)
        # st.write(uploadfilels)
        # display all upload files
        for uploadfile in uploadfilels:
            st.markdown(f"- {uploadfile}")

        remove = st.sidebar.button("删除已上传文件")
        if remove:
            remove_uploadfiles(uploadfolder)
            st.success("删除成功")

    elif choice == "文件选择":
        # choose radio for file1 or file2
        file_choice = st.sidebar.radio("选择文件", ["选择文件1", "选择文件2"])

        if file_choice == "选择文件1":
            # get current file1 value from session
            industry_choice = st.session_state["file1_industry"]
            rule_choice = st.session_state["file1_rulechoice"]
            filetype_choice = st.session_state["file1_filetype"]
            # section_choice = st.session_state["file1_section_list"]
        elif file_choice == "选择文件2":
            # get current file2 value from session
            industry_choice = st.session_state["file2_industry"]
            rule_choice = st.session_state["file2_rulechoice"]
            filetype_choice = st.session_state["file2_filetype"]
            # section_choice = st.session_state["file2_section_list"]
        # get preselected filetype index
        file_typels = ["监管制度", "上传文件"]
        if filetype_choice == "":
            filetype_index = 0
        else:
            filetype_index = file_typels.index(filetype_choice)
        # choose file type
        file_type = st.sidebar.selectbox(
            "选择文件类型",
            file_typels,
            index=filetype_index,
        )
        if file_type == "监管制度":
            industry_list = get_folder_list(rulefolder)
            # get preselected industry index
            # if industry_choice in industry_list:
            #     industry_index = industry_list.index(industry_choice)
            # else:
            industry_index = 0
            rule_choice = []
            # section_choice = []

            industry_choice = st.sidebar.selectbox(
                "选择行业:", industry_list, index=industry_index
            )

            rule_list = searchByIndustrysupa(industry_choice)
            rule_choice = st.sidebar.multiselect("选择匹配监管要求:", rule_list, rule_choice)
            rule_column_ls = []
            # rule_section_list = get_section_list(rule_val, rule_choice)
            # rule_column_ls = st.sidebar.multiselect(
            #     "选择章节:", rule_section_list, section_choice
            # )
            # if rule_column_ls == []:
            #     column_rule = ""
            # else:
            #     column_rule = "|".join(rule_column_ls)

            # if rule_choice != []:
            #     ruledf, rule_embeddings = get_rule_data(rule_choice, industry_choice)
            #     choosedf, _ = searchByItem(ruledf, rule_choice, column_rule, "")
            #     # get index of rule
            #     rule_index = choosedf.index.tolist()
            #     choose_embeddings = rule_embeddings[rule_index]
            # else:
            #     choosedf, choose_embeddings = None, None

        # elif file_type == "审计程序":

        #     industry_list = get_folder_list(auditfolder)

        #     # get preselected industry index
        #     # if industry_choice in industry_list:
        #     #     industry_index = industry_list.index(industry_choice)
        #     # else:
        #     industry_index = 0
        #     rule_choice = []
        #     section_choice = []

        #     industry_choice = st.sidebar.selectbox(
        #         "选择行业:", industry_list, index=industry_index
        #     )

        #     if industry_choice != "":
        #         name_text = ""
        #         searchresult, choicels = searchauditByName(name_text, industry_choice)

        #         rule_choice = st.sidebar.multiselect("选择监管制度:", choicels, rule_choice)

        #         if rule_choice == []:
        #             rule_choice = choicels
        #         section_list = get_section_list(searchresult, rule_choice)
        #         rule_column_ls = st.sidebar.multiselect(
        #             "选择章节:", section_list, section_choice
        #         )
        #         if rule_column_ls == []:
        #             column_text = ""
        #         else:
        #             column_text = "|".join(rule_column_ls)

        #         if rule_choice != []:
        #             ruledf, _ = searchauditByItem(
        #                 searchresult, rule_choice, column_text, "", "", ""
        #             )
        #             emblist = ruledf["监管要求"].unique().tolist()
        #             subsearchdf = get_sampleaudit(emblist, industry_choice)
        #             # fix index
        #             choosedf, _ = searchauditByItem(
        #                 subsearchdf, emblist, column_text, "", "", ""
        #             )
        #             # get index of the rule
        #             ruledf_index = choosedf.index.tolist()
        #             choosefolder = get_auditfolder(industry_choice)
        #             sentence_embeddings = get_embedding(choosefolder, rule_choice)
        #             # get sub-embedding by index
        #             choose_embeddings = sentence_embeddings[ruledf_index]
        #         else:
        #             choosedf, choose_embeddings = None, None

        elif file_type == "上传文件":
            if industry_choice != "":
                rule_choice = []
            upload_list = get_uploadfiles(filerawfolder)
            upload_choice = st.sidebar.multiselect("选择已上传文件:", upload_list, rule_choice)
            # if upload_choice != []:
            #     choosedf, choose_embeddings = get_upload_data(upload_choice)
            # else:
            #     choosedf, choose_embeddings = None, None
            industry_choice = ""
            rule_choice = upload_choice
            rule_column_ls = []

        # file choose button
        file_button = st.sidebar.button("选择文件")
        if file_button:
            if file_choice == "选择文件1":
                # file1df, file1_embeddings = choosedf, choose_embeddings
                file1_industry = industry_choice
                file1_rulechoice = rule_choice
                file1_filetype = file_type
                file1_section_list = rule_column_ls

                # if file1_filetype == "监管制度":
                #     file1df, file1_embeddings = searchByNamesupa(
                #         file1_rulechoice, file1_industry
                #     )
                # else:
                #     file1df, file1_embeddings = convert_index_to_df(file1_rulechoice)

                # st.session_state["file1df"] = file1df
                # st.session_state["file1_embeddings"] = file1_embeddings
                st.session_state["file1_industry"] = file1_industry
                st.session_state["file1_rulechoice"] = file1_rulechoice
                st.session_state["file1_filetype"] = file1_filetype
                st.session_state["file1_section_list"] = file1_section_list
                # file2df = st.session_state["file2df"]
                # file2_embeddings = st.session_state["file2_embeddings"]
                file2_industry = st.session_state["file2_industry"]
                file2_rulechoice = st.session_state["file2_rulechoice"]
                file2_filetype = st.session_state["file2_filetype"]
                file2_section_list = st.session_state["file2_section_list"]
            elif file_choice == "选择文件2":
                # file2df, file2_embeddings = choosedf, choose_embeddings
                file2_industry = industry_choice
                file2_rulechoice = rule_choice
                file2_filetype = file_type
                file2_section_list = rule_column_ls

                # if file2_filetype == "监管制度":
                #     file2df, file2_embeddings = searchByNamesupa(
                #         file2_rulechoice, file2_industry
                #     )
                # else:
                #     file2df, file2_embeddings = convert_index_to_df(file2_rulechoice)

                # st.write(file2df)
                # st.session_state["file2df"] = file2df
                # st.session_state["file2_embeddings"] = file2_embeddings
                st.session_state["file2_industry"] = file2_industry
                st.session_state["file2_rulechoice"] = file2_rulechoice
                st.session_state["file2_filetype"] = file2_filetype
                st.session_state["file2_section_list"] = file2_section_list
                # file1df = st.session_state["file1df"]
                # file1_embeddings = st.session_state["file1_embeddings"]
                file1_industry = st.session_state["file1_industry"]
                file1_rulechoice = st.session_state["file1_rulechoice"]
                file1_filetype = st.session_state["file1_filetype"]
                file1_section_list = st.session_state["file1_section_list"]
        else:
            # file1df = st.session_state["file1df"]
            # file2df = st.session_state["file2df"]
            # file1_embeddings = st.session_state["file1_embeddings"]
            # file2_embeddings = st.session_state["file2_embeddings"]
            file1_industry = st.session_state["file1_industry"]
            file2_industry = st.session_state["file2_industry"]
            file1_rulechoice = st.session_state["file1_rulechoice"]
            file2_rulechoice = st.session_state["file2_rulechoice"]
            file1_filetype = st.session_state["file1_filetype"]
            file2_filetype = st.session_state["file2_filetype"]
            file1_section_list = st.session_state["file1_section_list"]
            file2_section_list = st.session_state["file2_section_list"]

        # file choose reset
        file_reset = st.sidebar.button("重置文件")
        if file_reset:
            # file1df, file2df = None, None
            # file1_embeddings, file2_embeddings = None, None
            file1_industry, file2_industry = "", ""
            file1_rulechoice, file2_rulechoice = [], []
            file1_filetype, file2_filetype = "", ""
            file1_section_list, file2_section_list = [], []
            # st.session_state["file1df"] = file1df
            # st.session_state["file2df"] = file2df
            # st.session_state["file1_embeddings"] = file1_embeddings
            # st.session_state["file2_embeddings"] = file2_embeddings
            st.session_state["file1_industry"] = file1_industry
            st.session_state["file2_industry"] = file2_industry
            st.session_state["file1_rulechoice"] = file1_rulechoice
            st.session_state["file2_rulechoice"] = file2_rulechoice
            st.session_state["file1_filetype"] = file1_filetype
            st.session_state["file2_filetype"] = file2_filetype
            st.session_state["file1_section_list"] = file1_section_list
            st.session_state["file2_section_list"] = file2_section_list

        st.subheader("已选择的文件1：")
        # display file1 rulechoice
        if file1_rulechoice != []:
            # convert to string
            file1_rulechoice_str = "| ".join(file1_rulechoice)
            # display string
            st.warning("文件1：" + file1_rulechoice_str)
        else:
            st.error("文件1：无")
        # display file1 section
        if file1_section_list != []:
            # convert to string
            file1_section_str = "| ".join(file1_section_list)
            # display string
            st.info("章节1：" + file1_section_str)
        else:
            st.info("章节1：全部")

        st.subheader("已选择的文件2：")
        # display file2 rulechoice
        if file2_rulechoice != []:
            # convert to string
            file2_rulechoice_str = "| ".join(file2_rulechoice)
            # display string
            st.warning("文件2：" + file2_rulechoice_str)
        else:
            st.error("文件2：无")

        # display file2 section
        if file2_section_list != []:
            # convert to string
            file2_section_str = "| ".join(file2_section_list)
            # display string
            st.info("章节2：" + file2_section_str)
        else:
            st.info("章节2：全部")

    elif choice == "匹配分析":
        # file1df = st.session_state["file1df"]
        # file2df = st.session_state["file2df"]
        # file1_embeddings = st.session_state["file1_embeddings"]
        # file2_embeddings = st.session_state["file2_embeddings"]
        file1_rulechoice = st.session_state["file1_rulechoice"]
        file2_rulechoice = st.session_state["file2_rulechoice"]
        file1_industry = st.session_state["file1_industry"]
        file2_industry = st.session_state["file2_industry"]
        file1_filetype = st.session_state["file1_filetype"]
        file2_filetype = st.session_state["file2_filetype"]
        file1_section_list = st.session_state["file1_section_list"]
        file2_section_list = st.session_state["file2_section_list"]

        st.subheader("已选择的文件：")
        # display file1 rulechoice
        if file1_rulechoice != []:
            # convert to string
            file1_rulechoice_str = "| ".join(file1_rulechoice)
            # display string
            st.warning("文件1：" + file1_rulechoice_str)
        else:
            st.error("文件1：无")
        # display file1 section
        if file1_section_list != []:
            # convert to string
            file1_section_str = "| ".join(file1_section_list)
            # display string
            st.info("章节1：" + file1_section_str)
        else:
            st.info("章节1：全部")

        # display file2 rulechoice
        if file2_rulechoice != []:
            # convert to string
            file2_rulechoice_str = "| ".join(file2_rulechoice)
            # display string
            st.warning("文件2：" + file2_rulechoice_str)
        else:
            st.error("文件2：无")

        # display file2 section
        if file2_section_list != []:
            # convert to string
            file2_section_str = "| ".join(file2_section_list)
            # display string
            st.info("章节2：" + file2_section_str)
        else:
            st.info("章节2：全部")

        # if file1df is None
        # if file1df is None:
        #     st.error("请选择文件1")
        #     return
        # if file2df is None:
        #     st.error("请选择文件2")
        #     return

        # match_method = st.sidebar.radio("匹配方法选择", ["语义匹配"])
        # subruledf = file1df
        # subrule_embeddings = file1_embeddings
        # uploaddf = file2df[["条款", "结构", "监管要求"]]
        # upload_embeddings = file2_embeddings

        # if match_method == "语义匹配":
        # use expander
        # with st.sidebar.expander("参数设置"):

        querydf, _ = searchByNamesupa(file1_rulechoice, file1_industry)
        queries = querydf["条款"].tolist()
        # add new column of True or False
        matchdf = pd.DataFrame({"匹配状态": [False] * len(queries), "匹配条款": queries})
        st.data_editor(matchdf, key="matchdf")
        status = st.session_state["matchdf"]["edited_rows"]
        keys_with_true = [key for key, value in status.items() if value["匹配状态"]]
        # get query list by index
        query_list = [queries[i] for i in keys_with_true]

        if query_list == []:
            st.sidebar.error("请选择匹配条款")
            return
        else:
            st.markdown("#### 待匹配条款")
            for i in range(len(query_list)):
                st.markdown("###### " + str(i + 1) + ": " + query_list[i])

        # choose model
        model_name = st.sidebar.selectbox(
            "选择模型",
            [
                "gpt-35-turbo",
                "gpt-35-turbo-16k",
                "gpt-4",
                "gpt-4-32k",
                "gpt-4-turbo",
                "tongyi",
                "ERNIE-Bot-4",
                "ERNIE-Bot-turbo",
                "ChatGLM2-6B-32K",
                "Yi-34B-Chat",
                "gemini-pro",
            ],
        )
        top = st.sidebar.slider("匹配数量选择", min_value=1, max_value=10, value=4)

        # button for match
        match_button = st.sidebar.button("开始匹配")
        if match_button:
            with st.spinner("正在匹配中..."):
                # get match result
                answerls, sourcels = get_matchplc_batch(
                    query_list,
                    file2_filetype,
                    file2_industry,
                    file2_rulechoice,
                    top,
                    model_name,
                )

            for i in range(len(query_list)):
                st.markdown("#### 匹配结果")
                st.markdown("###### 待匹配条款 " + str(i + 1) + ": " + query_list[i])
                st.markdown("###### 分析 " + str(i + 1) + ": " + answerls[i])
                st.markdown("###### 来源 " + str(i + 1) + ": ")
                sourcedf = sourcels[i]
                st.table(sourcedf)

            exportdf = pd.DataFrame(
                {"待匹配条款": query_list, "分析": answerls, "来源": sourcels}
            )
            # analysis is done
            st.sidebar.success("分析完成")
            st.sidebar.download_button(
                label="下载结果",
                file_name="合规分析结果.csv",
                data=exportdf.to_csv(),
                mime="text/csv",
            )

    elif choice == "文件浏览":
        st.subheader("文件浏览")

        upload_list = get_uploadfiles(filerawfolder)
        upload_choice = st.sidebar.multiselect("选择已上传文件:", upload_list)
        if upload_choice == []:
            upload_choice = upload_list

        choosedf, choose_embeddings = convert_index_to_df(upload_choice)

        # st.write(choosedf)
        match = st.sidebar.radio("搜索方式", ("关键字搜索", "模糊搜索"))
        # initialize session value search_result
        if "search_result" not in st.session_state:
            st.session_state["search_result"] = None

        # placeholder
        placeholder = st.empty()

        if match == "关键字搜索":
            item_text = st.sidebar.text_input("按条文关键字搜索")

            if item_text != "":
                # convert upload full path
                uploadfullls = add_upload_folder(upload_choice)

                fullresultdf, total = searchByItem(
                    choosedf, uploadfullls, "", item_text
                )

                resultdf = fullresultdf[["监管要求", "结构", "条款"]]
                # update resultdf columns
                resultdf.columns = ["制度", "结构", "条款"]

                if resultdf.empty:
                    placeholder.text("没有搜索结果")
                else:
                    # reset index
                    resultdf = resultdf.reset_index(drop=True)
                    # placeholder.table(resultdf)

                    with placeholder.container():
                        # get columns value list of resultdf
                        mdf1 = resultdf[["制度", "结构"]]
                        # groupby col1 and convert col2 to list
                        mdf2 = mdf1.groupby("制度")["结构"].apply(list).reset_index()

                        plcmatch = mdf2["制度"].tolist()
                        colmatch = mdf2["结构"].tolist()
                        # display plc list
                        plcstr = "、".join(plcmatch)
                        st.warning("##### " + "匹配制度: " + plcstr)

                        for plc, col in zip(plcmatch, colmatch):
                            st.markdown("#### " + plc)

                            tab1, tab2 = st.tabs(["匹配结果", "制度浏览"])
                            with tab1:
                                disdf1 = resultdf[(resultdf["制度"] == plc)][["结构", "条款"]]
                                st.table(disdf1)

                            disdf = choosedf[["监管要求", "结构", "条款"]].reset_index(
                                drop=True
                            )
                            # st.write(disdf)
                            with tab2:
                                # st.write(col)

                                subdf = disdf[(disdf["监管要求"] == plc)][["结构", "条款"]]
                                # Subset your original dataframe with condition
                                df_ = subdf[(subdf["结构"].isin(col))]
                                # st.write(df_)

                                # Pass the subset dataframe index and column to pd.IndexSlice
                                slice_ = pd.IndexSlice[df_.index, df_.columns]
                                # st.write(slice_)
                                s = subdf.style.set_properties(
                                    **{"color": "red"}, subset=slice_
                                )
                                # display s in html format
                                st.table(s)

                    # search is done
                    # st.sidebar.success('搜索完成')
                    st.sidebar.success("共搜索到" + str(total) + "条结果")
                    st.sidebar.download_button(
                        label="下载搜索结果",
                        data=resultdf.to_csv(),
                        file_name="搜索结果.csv",
                        mime="text/csv",
                    )

            else:
                st.sidebar.warning("请输入搜索条件")
                resultdf = st.session_state["search_result"]

        elif match == "模糊搜索":
            search_text = st.sidebar.text_area("输入搜索条件")

            top = st.sidebar.slider("匹配数量选择", min_value=1, max_value=10, value=3)

            search = st.sidebar.button("搜索条款")

            if search:
                with st.spinner("正在搜索..."):
                    resultdf = searchupload(search_text, upload_choice, top)

                    # resultdf = fullresultdf[:top][["监管要求", "结构", "条款"]]

                    # reset index
                    resultdf.reset_index(drop=True, inplace=True)
                    placeholder.table(resultdf)
                    # search is done
                    # st.sidebar.success('搜索完成')
                    st.sidebar.success("共搜索到" + str(resultdf.shape[0]) + "条结果")
                    st.sidebar.download_button(
                        label="下载搜索结果",
                        data=resultdf.to_csv(),
                        file_name="搜索结果.csv",
                        mime="text/csv",
                    )
            else:
                st.sidebar.warning("请输入搜索条件")
                resultdf = st.session_state["search_result"]

    elif choice == "文件问答":
        st.subheader("文件问答")

        upload_list = get_uploadfiles(filerawfolder)
        upload_choice = st.sidebar.multiselect("选择已上传文件:", upload_list)
        if upload_choice == []:
            upload_choice = upload_list

        # choose chain type
        # chain_type = st.sidebar.selectbox(
        #     "选择链条类型", ["stuff", "map_reduce", "refine", "map_rerank"]
        # )
        chain_type = "stuff"
        # choose model
        model_name = st.sidebar.selectbox(
            "选择模型",
            [
                "gpt-35-turbo",
                "gpt-35-turbo-16k",
                "gpt-4",
                "gpt-4-32k",
                "gpt-4-turbo",
                "tongyi",
                "ERNIE-Bot-4",
                "ERNIE-Bot-turbo",
                "ChatGLM2-6B-32K",
                "Yi-34B-Chat",
                "gemini-pro",
            ],
        )

        # choose top_k
        top_k = st.sidebar.slider("选择top_k", 1, 10, 3)
        # question input
        question = st.text_area("输入问题")

        # answer button
        answer_btn = st.button("获取答案")
        if answer_btn:
            if question != "" and chain_type != "":
                with st.spinner("正在获取答案..."):
                    # get answer
                    # answer = gpt_answer(question,chain_type)
                    # docsearch = st.session_state["docsearch"]
                    answer, sourcedb = gpt_vectoranswer(
                        question,
                        upload_choice,
                        chain_type,
                        top_k=top_k,
                        model_name=model_name,
                    )
                    st.markdown("#### 答案")
                    st.write(answer)
                    with st.expander("查看来源"):
                        st.markdown("#### 来源")
                        st.table(sourcedb)
            else:
                st.error("问题或链条类型不能为空")


if __name__ == "__main__":
    main()
