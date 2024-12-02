from langchain import hub
from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.agents import AgentExecutor, create_react_agent, load_tools
from langchain_openai import OpenAI

from langchain_community.callbacks.streamlit import (
    StreamlitCallbackHandler,
)
from langchain_community.agent_toolkits.load_tools import load_tools
from langchain.agents.agent_toolkits import create_conversational_retrieval_agent
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_openai import ChatOpenAI
from langchain_chroma import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings
import openai
import os
from langchain.tools.retriever import create_retriever_tool

from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv()) # Read .env file
import streamlit as st

from langchain_community.document_loaders import PyPDFLoader

openai.api_key  = os.environ['OPENAI_API_KEY']

st.set_page_config(
    page_title="Travel Advice for Cameroon ",
    page_icon="🇨🇲",
    layout="wide",
    initial_sidebar_state="expanded"
)
st.header(' Welcome to VisitCameroon 🇨🇲!!')

loader = PyPDFLoader('data/police_assurance.pdf')
pages = loader.load()

for page in pages:
    page.page_content = page.page_content.replace("*", "").replace("###", "")

# Split documents
full_text = " ".join([page.page_content for page in pages]) # When there are multiple pages.
r_splitter = RecursiveCharacterTextSplitter(
separators=["\n\n", "\n", " ", ""],
chunk_size=300,
chunk_overlap=50
    
)
text_split = r_splitter.split_text(full_text)
# define embedding
embeddings = OpenAIEmbeddings()
# create vector database from data
embedding_model = OpenAIEmbeddings(model="text-embedding-3-large")
persist_directory = 'data/chroma/'
vectordb = Chroma(
    collection_name="insurance_collection",
    persist_directory=persist_directory, # Where to save data locally, remove if not necessary
    embedding_function=embedding_model
)
db = vectordb.from_texts(text_split, embedding_model)

# define retriever
retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 3})
tool = create_retriever_tool(
    retriever,
    "police_assurance",
    "Searches and returns texts regarding  police_assurance.pdf.",
)
tools = [tool]
prompt = hub.pull("hwchase17/openai-tools-agent")
# create a chatbot chain. Memory is managed externally.
llm_name = 'gpt-3.5-turbo'
llm = ChatOpenAI(model_name=llm_name, temperature=0)
# Memory
memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True
)
# tools = load_tools(["ddg-search"])
# prompt = hub.pull("hwchase17/react")
agent = create_conversational_retrieval_agent(llm, tools, memory_key='chat_history', verbose=True)
# agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home","AI Assistant", "General Information"])

# Default page
if 'page' not in st.session_state:
    st.session_state.page = "Home"

# Home Page
if page == "Home":
    st.title("Welcome to Travel Advice for Cameroon")
    st.write("Use the buttons on the left to navigate to different pages.")
    # st.image('reunif.png', caption='Monument de la réunification')

if page == "AI Assistant":
    st.title("AI assistant")
    st.write("This is the page for the LLM bot, your travel assistant in Cameroon with Internet access. What are you planning for your next trip?")
    # image = Image.open('reunif.png')
    # st.image(image, caption='Monument de la réunification',)

    #
    # Build prompt
    from langchain.prompts import PromptTemplate
    template = """Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer. Use three sentences maximum. Keep the answer as concise as possible. 
    {context}
    Question: {question}
    Helpful Answer:"""
    QA_CHAIN_PROMPT = PromptTemplate(input_variables=["context", "question"],template=template,)

    # Run chain
    from langchain.chains import RetrievalQA
    qa_chain = RetrievalQA.from_chain_type(llm,
                                        retriever=db.as_retriever(),
                                        return_source_documents=True,
                                        chain_type_kwargs={"prompt": QA_CHAIN_PROMPT})
    #

    question = st.text_input(
        "Votre question :",
        placeholder="Ask me anything!"

        )
    if "messages" not in st.session_state:
            st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]
    if "memory" not in st.session_state:
        st.session_state['memory'] = memory
    # Add your LLM bot code here
    if question:
        result = qa_chain({"query": question})
        st.session_state.messages.append({"role": "user", "content": question})
        st.chat_message("user").write(question)
        print(f'Question : {question }')
    with st.chat_message("assistant"):
        # result = qa_chain({"query": user_query})
        st_cb = StreamlitCallbackHandler(st.container())
        print(f'st_cb : {st_cb }')
        response = agent(question, callbacks=[st_cb])
        print(f'response : {response }')
        st.session_state.messages.append({"role": "assistant", "content": result["result"] })
        st.write(result["result"])
        print(f'Result : {result["result"] }')

elif page == "General Information":
    st.title("General Information")
    st.write("This page contains General travel information about Cameroon.")


    # destinations = {
    #     "Douala": "A vibrant city with a bustling seaport and lively markets.",
    #     "Yaoundé": "The capital city known for its museums and monuments.",
    #     "Kribi": "Famous for its beautiful beaches and the Chutes de la Lobé waterfalls.",
    #     "Waza National Park": "A great place to see elephants, giraffes, and other wildlife."
    # }
    # search = st.text_input("Search for a destination:")
    # if search:
    #     st.write(f"Results for {search}:")
    #     if search in destinations:
    #         st.write(destinations[search])
    #     else:
    #         st.write("Destination not found.")
    # tips = [
    #     "Best time to visit: November to February.",
    #     "Languages spoken: Ewondo, Bulu, Douala, Bassa, French, English.",
    #     "Currency: Central African CFA franc (XAF).",
    # ]
    # for tip in tips:
    #     st.write(f"- {tip}")
# qa = ConversationalRetrievalChain.from_llm(
#     llm=llm, 
#     retriever=retriever, 
#     memory=memory,
#     return_generated_question=True,
# )


# if prompt := st.chat_input():
#     st.chat_message("user").write(prompt)
#     with st.chat_message("assistant"):
#         st_callback = StreamlitCallbackHandler(st.container())
#         response = agent_executor.invoke(
#             {"input": prompt}, {"callbacks": [st_callback]}
#         )
#         st.write(response["output"])

# st.title("Cars Insurance Chatbot")

# USER_AVATAR = "👤"
# BOT_AVATAR = "🤖"
# client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# # Ensure openai_model is initialized in session state
# if "openai_model" not in st.session_state:
#     st.session_state["openai_model"] = "gpt-3.5-turbo"


# # Load chat history from shelve file
# def load_chat_history():
#     with shelve.open("chat_history") as db:
#         return db.get("messages", [])


# # Save chat history to shelve file
# def save_chat_history(messages):
#     with shelve.open("chat_history") as db:
#         db["messages"] = messages


# # Initialize or load chat history
# if "messages" not in st.session_state:
#     st.session_state.messages = load_chat_history()

# # Sidebar with a button to delete chat history
# with st.sidebar:
#     if st.button("Delete Chat History"):
#         st.session_state.messages = []
#         save_chat_history([])

# # Display chat messages
# for message in st.session_state.messages:
#     avatar = USER_AVATAR if message["role"] == "user" else BOT_AVATAR
#     with st.chat_message(message["role"], avatar=avatar):
#         st.markdown(message["content"])

# # Main chat interface
# if prompt := st.chat_input("How can I help?"):
#     st.session_state.messages.append({"role": "user", "content": prompt})
#     with st.chat_message("user", avatar=USER_AVATAR):
#         st.markdown(prompt)

#     with st.chat_message("assistant", avatar=BOT_AVATAR):
#         message_placeholder = st.empty()
#         full_response = ""
#         for response in client.chat.completions.create(
#             model=st.session_state["openai_model"],
#             messages=st.session_state["messages"],
#             stream=True,
#         ):
#             full_response += response.choices[0].delta.content or ""
#             message_placeholder.markdown(full_response + "|")
#         message_placeholder.markdown(full_response)
#     st.session_state.messages.append({"role": "assistant", "content": full_response})

# # Save chat history after each interaction
# save_chat_history(st.session_state.messages)