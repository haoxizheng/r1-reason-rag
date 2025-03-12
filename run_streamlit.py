import streamlit as st
import subprocess
import asyncio
from datetime import datetime
from src.agent import QAAgent
import os
import glob
from collections import defaultdict
import re
from streamlit_mermaid import st_mermaid


st.set_page_config(layout="wide")
st.title('📖 Hithink Deep Research V3')

agent = QAAgent()

def clean():
    # 保持用户输入为空
    if "user_input" in st.session_state:
        st.session_state['user_input'] = []
    if "start_time" in st.session_state:
        st.session_state["start_time"] = ""
    if "message_queue" in st.session_state:
        st.session_state['message_queue'] = []
    
    # 使用rerun来刷新整个页面
    st.rerun()

def parse_file_names(file_names):
    problems = defaultdict(list)
    for file_name in file_names:
        parts = file_name.split('_')
        if len(parts) == 2:
            problem, time = parts[0], parts[1].split('.')[0]
            problem = problem[5:]
            problems[problem].append(time)
    return problems

def mermaid(self, report: str, add_str: str = ""):
        with st.chat_message("assistant"):
            st.markdown(f"==== FINAL REPORT ({add_str}) ====")
            if "```mermaid" in report:
                parts = re.split(r'(```mermaid.*?```)', report, flags=re.DOTALL)
                for part in parts:
                    if not part.strip():
                        continue
                    if part.strip().startswith('```mermaid'):
                        code = re.search(r'```mermaid\s*(.*?)\s*```', part, re.DOTALL).group(1)
                        st_mermaid(code, key=str(uuid.uuid4()))
                    else:
                        st.markdown(part)

            else:
                st.markdown(report)

def show_hist_log(show_file):
    try:
        with open(show_file, 'r', encoding='utf-8') as file:
            content = file.read()
            print(content)
            agent.mermaid(content)
    except FileNotFoundError:
        with st.chat_message("assistant"):
            content = f"记录 {show_file} 未找到。"
            st.markdown(content)
        return
    except Exception as e:
        with st.chat_message("assistant"):
            content = f"读取记录 {show_file} 时出错: {e}"
            st.markdown(content)
        return
    
    with st.chat_message("assistant"):
        st.markdown(f"正在查找历史记录")
        st.markdown(content)
        st.session_state['start_time'] =  datetime.now()
    
def main():
    if 'user_input' not in st.session_state:
        st.session_state['user_input'] = []
    if "start_time" not in st.session_state:
        st.session_state["start_time"] = ""
    if "message_queue" not in st.session_state:
        st.session_state['message_queue'] = []

    model = st.sidebar.selectbox('选择模型', ['deepseek-r1'])
    is_polish = st.sidebar.selectbox('是否润色', [False, True])
    polish_step = st.sidebar.slider('润色步数', min_value=2, max_value=8, value=4)
    max_outline_num = st.sidebar.slider('大纲数', min_value=1, max_value=8, value=8)
    max_loop = st.sidebar.slider('最大循环数', min_value=1, max_value=10, value=5)
    clear = st.sidebar.button("clear")
    if clear:
        clean()


    files = glob.glob(os.path.join(f"logs", '*'))
    problems = parse_file_names(files)

    
    if 'selected_problem' not in st.session_state:
        st.session_state.selected_problem = None
    if 'selected_time' not in st.session_state:
        st.session_state.selected_time = None

    st.sidebar.selectbox(
        "问题记录",
        options=[""] + list(problems.keys()),  
        key="selected_problem",
        on_change=lambda: st.session_state.update(selected_time=None)  
    )


    if st.session_state.selected_problem:
        st.sidebar.selectbox(
            "问题时间",
            options=[""] + problems[st.session_state.selected_problem],  # 添加空字符串作为默认选项
            key="selected_time"
    )


    show_history = st.sidebar.button("查询历史")
    if show_history:
        print(st.session_state.selected_problem+st.session_state.selected_time)
        show_hist_log(show_file=f"logs/{st.session_state.selected_problem}_{st.session_state.selected_time}.log")

    user_input = st.chat_input("Enter a question:")


    if user_input :

        with st.chat_message("user"):
            st.markdown(user_input)
        start_time = datetime.now()
        agent.run(user_input, polish=is_polish, polish_step=polish_step, max_outline_num=max_outline_num, max_loop=max_loop)

        st.session_state['user_input'].append(user_input)
        st.session_state['start_time'] = start_time
                  


if __name__ == '__main__':
    main()
