from langchain.llms import CTransformers
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
import time
import gradio as gr
import os

llm = CTransformers(model= "local_models/llama-2-7b-chat.Q4_K_M.gguf")
embeddings = HuggingFaceEmbeddings(model_name = 'local_models/embeddings-bge-large/')



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
    docsearch = FAISS.from_documents(text_chunks, embeddings)
    docsearch.save_local('PMS_vector_db/PMS_index')
    return docsearch

def get_vector_db(db_path):
    if os.path.exists(db_path):
        vector_db = FAISS.load_local(db_path, embeddings)
        print('loading from the existing vectorDB')
    else:
        data = load_data("PMS_pdfs/")
        vector_db = build_vector_db(data)
    return vector_db

def predict(prompt,history):
    vector_db = get_vector_db('PMS_vector_db/PMS_index/')
    qa = RetrievalQA.from_chain_type(llm=llm, chain_type='stuff',
                                retriever = vector_db.as_retriever(),
                                return_source_documents = True)
    response = qa({'query':prompt})
    response = response['result']
    for i in range(len(response)):
      time.sleep(0.05)
      yield response[:i+1]


gr.ChatInterface(predict).queue().launch()


