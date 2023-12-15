import streamlit as st
import pandas as pd
from langchain.chat_models import AzureChatOpenAI

import time


# Page configurations
st.set_page_config(page_title="GenAI-Bank App", page_icon="🚀", layout="wide")

# Function to render the home page
def home():
    st.title("Customer Landing Page")
    
    # Get user input
    user_input = st.text_input("Enter your unique customer ID:")
    if "messages" in st.session_state:
        st.session_state.messages = []
        st.session_state.messages.append({"role": "assistant", "content": "Hi, How can I Help you?"})
    # print(user_input, type(int(user_input)))
    # Check if the input is a valid number
    if user_input:
        fraud_classify = get_customer_preds(str(user_input))
        print(fraud_classify)
        # fraud_classify = user_input

        

        # Check if the number is even
        if fraud_classify == 'legit':
            st.session_state.user_input = user_input
            st.success(f"The given customer is classifed as: {fraud_classify}")
            st.button("Go to About Page", on_click=about)
        else:
            st.error(f"The given customer is classifed as: {fraud_classify}")

        # else:
        # st.error(f"You entered an odd number. Please enter an even number.")


# Function to render the about page
def about():
    st.title("Customer Profile")
    if hasattr(st.session_state, 'user_input'):
        # st.write(f"This is the About Page. You entered the number: {st.session_state.user_input}.")
        customer_details = get_customer_data(str(st.session_state.user_input))

        col1, col2 = st.columns(2)
        for key, value in customer_details.items():
            if key in ["name", "account number", "age", "country", "balance"]:
                with col1:
                    st.write(f"**{key}:**")
                with col2:
                    st.write(value)
        st.write(" ")
        st.write(" ")
        st.write(" ")
        st.write(" ")
        st.write(" ")
        st.write(" ")
        st.write(" ")
        st.write(" ")
        st.write(" ")
        st.write(" ")
        st.write(" ")
        st.write(" ")
        st.write(" ")
        st.write(" ")
        st.write(" ")
        st.write(" ")
        st.write(" ")
        st.write(" ")
        st.write(" ")
        st.write(" ")
        st.write(" ")
        st.write(" ")
        st.write(" ")
        st.write(" ")

    else:
        st.warning("Please go to the Home Page and enter a valid number first.")

# Function to render the contact page
def contact():
    import streamlit as st
    from langchain.llms import CTransformers
    # from langchain.tools import DuckDuckGoSearchRun
    # from pathlib import Path
    from langchain.llms import CTransformers
    from langchain.document_loaders import PyPDFLoader
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain.embeddings import HuggingFaceEmbeddings
    from langchain.vectorstores import FAISS
    from langchain.chains import RetrievalQA
    # from langchain.callbacks import StreamlitCallbackHandler
    # from langchain.chat_models import AzureChatOpenAI
    from dotenv import load_dotenv
    import os
    import re
    from langchain.chat_models import AzureChatOpenAI

    import time
    load_dotenv()


    # gpt_engine = os.environ.get('OPEN_AI_ENGINE')
    # openai_api_version = os.environ.get('OPEN_AI_VERSION')
    # gpt_api_base = os.environ.get('OPEN_AI_BASE')
    # gpt_api_key = os.environ.get('OPEN_AI_KEY')

    if 'embeddings' not in st.session_state:
        print('loading embeddings...')
        st.session_state['embeddings'] = HuggingFaceEmbeddings(model_name = 'BAAI/bge-large-en-v1.5')
        print('embeddings loaded')

    if 'LLM' not in st.session_state:
        print('loading llm...')
        st.session_state['LLM'] = CTransformers(model="TheBloke/Llama-2-7B-Chat-GGUF",
                    model_file="llama-2-7b-chat.Q4_K_M.gguf",
                    model_type="llama",
                    max_new_tokens=512,
                    temperature=0.1)
        #CTransformers(model= "local_models/llama-2-7b-chat.Q4_K_M.gguf") #HuggingFaceEmbeddings(model_name = 'local_models/embeddings-bge-large/')
       # st.session_state['LLM'] = AzureChatOpenAI(deployment_name=gpt_engine, temperature=0, openai_api_version=openai_api_version, openai_api_key=gpt_api_key,
                         # openai_api_base=gpt_api_base) #HuggingFaceEmbeddings(model_name = 'local_models/embeddings-bge-large/')
        
        print('llm loaded')

    def get_llm_model(selected_option):
        if selected_option == "LLama2":
            llm = CTransformers(model="TheBloke/Llama-2-7B-Chat-GGUF",
                    model_file="llama-2-7b-chat.Q4_K_M.gguf",
                    model_type="llama",
                    max_new_tokens=512,
                    temperature=0.1)
        elif selected_option == "ChatGPT":
            llm = ""
            #llm = AzureChatOpenAI(deployment_name=gpt_engine, temperature=0, openai_api_version=openai_api_version, openai_api_key=gpt_api_key,
                        #openai_api_base=gpt_api_base)
        return llm

    # llm = CTransformers(model= "local_models/llama-2-7b-chat.Q4_K_M.gguf")
    # embeddings = HuggingFaceEmbeddings(model_name = 'local_models/embeddings-bge-large/')
    def load_data(dir_path):
        files = os.listdir(dir_path)
        data = []
        for file in files:
            print(file)
            loader = PyPDFLoader(dir_path+file)
            pages = loader.load_and_split()
            data.extend(pages)
        return data

    def build_vector_db(data):
        text_splitter = RecursiveCharacterTextSplitter(
            # Set a really small chunk size, just to show.
            chunk_size = 300,
            chunk_overlap  = 30,
            length_function = len,
        )
        text_chunks = text_splitter.split_documents(data)
        print(len(text_chunks))
        docsearch = FAISS.from_documents(text_chunks, st.session_state.embeddings)
        docsearch.save_local('PMS_vector_db/PMS_index')
        return docsearch


    def get_vector_db(db_path):
        if os.path.exists(db_path):
            vector_db = FAISS.load_local(db_path, st.session_state.embeddings)
            print('loading from the existing vectorDB')
        else:
            data = load_data("PMS_pdfs/")
            vector_db = build_vector_db(data)
        return vector_db

    def get_response(prompt ):
        vector_db = get_vector_db('PMS_vector_db/PMS_index/')
        qa = RetrievalQA.from_chain_type(llm=st.session_state.LLM, chain_type='stuff',
                                    retriever = vector_db.as_retriever(),
                                    return_source_documents = True)
        response = qa({'query':prompt})
        response = response['result']
        return response

    if "messages" not in st.session_state:
        st.session_state.messages = []
        st.session_state.messages.append({"role": "assistant", "content": "Hi, How can I Help you?"})

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Accept user input
    if prompt := st.chat_input("What is up?"):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(prompt)

        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            # search = DuckDuckGoSearchRun()
            with st.spinner("loading..."):
                assistant_response = get_response(prompt)#search.run(prompt)
                assistant_response = assistant_response.replace('\n', '  \n')
            # Simulate stream of response with milliseconds delay
            for chunk in re.split(r'(\s+)', assistant_response):
                full_response += chunk + " "
                time.sleep(0.05)
                # Add a blinking cursor to simulate typing
                message_placeholder.markdown(full_response + "▌")
            message_placeholder.markdown(assistant_response)
            print(assistant_response)
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": full_response})


# Navigation
pages = {
    "Home": home,
    "About": about,
    "Chat": contact,
}


def get_customer_preds(customer_id):
    fraud_predictions = pd.read_csv('customer_data_updated.csv',encoding='ISO-8859-1')
    # print(fraud_predictions.shape)
    # fraud_predictions.head()
    fraud_classify = fraud_predictions[fraud_predictions['EEID'] == customer_id]['Label'].values[0]
    return fraud_classify

def get_customer_data(customer_id):
    fraud_predictions = pd.read_csv('customer_data_updated.csv',encoding='ISO-8859-1')
    # print(fraud_predictions.shape)
    # fraud_predictions.head()
    customer_name = fraud_predictions[fraud_predictions['EEID'] == customer_id]['Full Name'].values[0]
    acc_no = fraud_predictions[fraud_predictions['EEID'] == customer_id]['Account Number'].values[0]
    age = fraud_predictions[fraud_predictions['EEID'] == customer_id]['Age'].values[0]
    country = fraud_predictions[fraud_predictions['EEID'] == customer_id]['Country'].values[0]
    current_balance = fraud_predictions[fraud_predictions['EEID'] == customer_id]['Bank Balance'].values[0]
    # fraud_classify = fraud_predictions[fraud_predictions['EEID'] == customer_id]['Label'].values[0]
    output_response = {'name': customer_name,
                       'account number':  acc_no,
                       'age': age,
                       'country':country,
                       'balance': current_balance,
                       }
    return output_response

# Sidebar with page selection
selected_page = st.sidebar.selectbox("Select a page", list(pages.keys()))

# Render the selected page
pages[selected_page]()
