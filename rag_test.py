import autogen
from autogen.agentchat.contrib.web_surfer import WebSurferAgent
from autogen.agentchat.contrib.retrieve_user_proxy_agent import RetrieveUserProxyAgent
from autogen.agentchat.contrib.retrieve_assistant_agent import RetrieveAssistantAgent

import streamlit as st 

from dotenv import load_dotenv
import os

from chromadb.utils import embedding_functions

from openai import OpenAI

st.set_page_config(page_title="Report Generation", page_icon="🤖", layout="wide")

load_dotenv()

BING_KEY = os.environ.get("BING_API")

class CustomAssistantAgent(RetrieveAssistantAgent):
    def _process_received_message(self, message, sender, silent):
        formatted_message = ""  # Initialize formatted_message as an empty string
    
        # Handle the case when message is a dictionary
        if isinstance(message, dict):
            if 'content' in message and message['content'].strip():
                formatted_message = f"**{sender.name}**: {message['content']}"
                st.session_state.setdefault("displayed_messages", []).append(message['content'])
            else:
                return super()._process_received_message(message, sender, silent)
        # Handle the case when message is a string
        elif isinstance(message, str) and message.strip():
            formatted_message = f"**{sender.name}**: {message}"
            st.session_state.setdefault("displayed_messages", []).append(message)
        else:
            return super()._process_received_message(message, sender, silent)
    
        # Only format and display the message if the sender is not the manager
        if sender != manager and formatted_message:
        # if formatted_message:
            with st.chat_message(sender.name):
                st.markdown(formatted_message + "\n")
    
        return super()._process_received_message(message, sender, silent)

class CustomProxyRetrieveAgent(RetrieveUserProxyAgent):
    def _process_received_message(self, message, sender, silent):
        formatted_message = ""  # Initialize formatted_message as an empty string
    
        # Handle the case when message is a dictionary
        if isinstance(message, dict):
            if 'content' in message and message['content'].strip():
                formatted_message = f"**{sender.name}**: {message['content']}"
                st.session_state.setdefault("displayed_messages", []).append(message['content'])
            else:
                return super()._process_received_message(message, sender, silent)
        # Handle the case when message is a string
        elif isinstance(message, str) and message.strip():
            formatted_message = f"**{sender.name}**: {message}"
            st.session_state.setdefault("displayed_messages", []).append(message)
        else:
            return super()._process_received_message(message, sender, silent)
    
        # Only format and display the message if the sender is not the manager
        if sender != manager and formatted_message:
        # if formatted_message:
            with st.chat_message(sender.name):
                st.markdown(formatted_message + "\n")
    
        return super()._process_received_message(message, sender, silent)
    
class CustomWebSearchAgent(WebSurferAgent):
    def _process_received_message(self, message, sender, silent):
        formatted_message = ""  # Initialize formatted_message as an empty string
    
        # Handle the case when message is a dictionary
        if isinstance(message, dict):
            if 'content' in message and message['content'].strip():
                formatted_message = f"**{sender.name}**: {message['content']}"
                st.session_state.setdefault("displayed_messages", []).append(message['content'])
            else:
                return super()._process_received_message(message, sender, silent)
        # Handle the case when message is a string
        elif isinstance(message, str) and message.strip():
            formatted_message = f"**{sender.name}**: {message}"
            st.session_state.setdefault("displayed_messages", []).append(message)
        else:
            return super()._process_received_message(message, sender, silent)
    
        # Only format and display the message if the sender is not the manager
        if sender != manager and formatted_message:
        # if formatted_message:
            with st.chat_message(sender.name):
                st.markdown(formatted_message + "\n")
    
        return super()._process_received_message(message, sender, silent)

class CustomGroupChatManager(autogen.GroupChatManager):
    def _process_received_message(self, message, sender, silent):
        formatted_message = ""  # Initialize formatted_message as an empty string
        # Handle the case when message is a dictionary
        if isinstance(message, dict):
            if 'content' in message and message['content'].strip():
                formatted_message = f"**{sender.name}**: {message['content']}"
                st.session_state.setdefault("displayed_messages", []).append(message['content'])
            else:
                return super()._process_received_message(message, sender, silent)
        # Handle the case when message is a string
        elif isinstance(message, str) and message.strip():
            formatted_message = f"**{sender.name}**: {message}"
            st.session_state.setdefault("displayed_messages", []).append(message)
        else:
            return super()._process_received_message(message, sender, silent)
    
        # # Only format and display the message if the sender is not the manager
        if sender != manager and formatted_message:
        # if formatted_message:
            with st.chat_message(sender.name):
                st.markdown(formatted_message + "\n")

        filename = "chat_summary.txt"

        with open(filename, 'a') as f:
            f.write(formatted_message + "\n")
        return super()._process_received_message(message, sender, silent)
    


embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
    

llm_config = {
    "config_list": [
                {
                    "model": "l3custom", 
                    "api_key": "ollama", 
                    "base_url": "http://localhost:11434/v1",
                  
                }
            ],
            "cache_seed": None,
            
}

assistant = CustomAssistantAgent(
    name="assistant",
    system_message="You are a helpful assistant.",
    llm_config=llm_config,
    description="Assistant that can be assigned to create reports using provided information and search for additional information."
)          

searcher_agent = WebSurferAgent(
    name="Web Searcher",
    system_message=''' You are a web searcher who excels in finding data on the web for market research. You know that you must use a query with the search tools.''',
    llm_config=llm_config,
    summarizer_llm_config=llm_config,
    browser_config={"viewport_size": 4096, "bing_api_key": BING_KEY},
    description = "Can search the web and provide information otherwise unavailable to other agents. Can also summarize the information found."

)


retrieve_proxy = CustomProxyRetrieveAgent(
    name="RAG Proxy",
    human_input_mode="NEVER",
    max_consecutive_auto_reply=3,
    retrieve_config={
        "task": "qa",
        "vector_db": "chroma",
        "collection": "retail",
        "docs_path": [
            "./2023_Winter_Retail_Report.txt",
           "./Colliers_Spring_Retail_Report_2024.txt",
           "./YG_FMCG_CPG_Retail_Whitepaper.txt",
            os.path.join(os.path.abspath(""))
        ],
        "custom_text_types": ["non-existent-type"],
        "chunk_token_size": 1000,
        "model": "l3custom",
        "overwrite": True,
        "get_or_create": True
        
    },
    description="Assistant who has extra content retrieval power for solving difficult problems",
    code_execution_config=False,
    
)



groupchat = autogen.GroupChat(agents=[assistant, searcher_agent], messages=[], speaker_selection_method="random", max_round=20)
manager = CustomGroupChatManager(groupchat=groupchat, llm_config=llm_config)


with st.container(height=800):
    kickoff = st.button("Start Research")
    retail_problem = "Generate a detailed report on the current state of the retail market, opportunities in the market, trends in the retail market, and key players in the market. This report should be at least 500 words. Use our documents and web research to get a final report."
    if kickoff:
        assistant.reset()
        if "chat_initiated" not in st.session_state:
            st.session_state.chat_initiated = False
            if not st.session_state.chat_initiated:

                retrieve_proxy.initiate_chat(
                    manager,
                    message=retrieve_proxy.message_generator,
                    problem = retail_problem,
                    n_results = 4,
                llm_config=llm_config,
                )         
                st.session_state.chat_initiated = True

    report = st.button("Generate Report")
    if report:
        llm = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
        with open('./chat_summary.txt', 'r') as f:
            chat_summary = f.read()

        output = llm.chat.completions.create(
            messages = [
                {
                    "role": "system",
                    "content": f"""You are a researcher preparing a market report on the retail industry. Your assistants have gathered information from documents and web research.
                            The documents were provided to you as {chat_summary}.
                            Review this information and prepare a detailed report on the current state of the retail market, opportunities in the market, trends in the retail market, and key players in the market.
                            Your report should be well-organized and cite sources.
                            The report should be at least 500 words.
                            The report should be in markdown format."""
                }
            ],
            model = "l3custom",
        )


        response = output.choices[0].message.content
        st.markdown(response)
        st.markdown("### Report Generated")
        with open('./report.md', 'w') as f:
            f.write(response)
        st.markdown("### Report Saved")