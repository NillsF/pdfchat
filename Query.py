import streamlit as st

from streamlit_chat import message
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import AzureOpenAIEmbeddings
#from langchain.chat_models import ChatOpenAI
from langchain_openai import AzureChatOpenAI
from langchain.schema import (HumanMessage)
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_mistralai.chat_models import ChatMistralAI

#from langchain.llms import AzureOpenAI


import openai

import dotenv
import os

# load environment variables
dotenv.load_dotenv()

AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AUZRE_ENDPOINT = AZURE_OPENAI_ENDPOINT
OPENAI_DEPLOYMENT_NAME = os.getenv("OPENAI_DEPLOYMENT_NAME")
OPENAI_MODEL_NAME = os.getenv("OPENAI_MODEL_NAME")
OPENAI_EMBEDDING_DEPLOYMENT_NAME = os.getenv("OPENAI_EMBEDDING_DEPLOYMENT_NAME")
OPENAI_EMBEDDING_MODEL_NAME = os.getenv("OPENAI_EMBEDDING_MODEL_NAME")
OPENAI_DEPLOYMENT_VERSION = os.getenv("OPENAI_DEPLOYMENT_VERSION")
MISTRAL_KEY = os.getenv("MISTRAL_KEY")
MISTRAL_ENDPOINT = os.getenv("MISTRAL_ENDPOINT")
#init Azure OpenAI
#openai.api_type = "azure"
#openai.api_version = "2023-05-15"  # subject to change
#openai.api_base = AZURE_OPENAI_ENDPOINT
#openai.api_key = AZURE_OPENAI_API_KEY


# use OpenAI Embeddings to generate embeddings for FAISS
embeddings = AzureOpenAIEmbeddings(azure_endpoint=AZURE_OPENAI_ENDPOINT,azure_deployment=OPENAI_EMBEDDING_DEPLOYMENT_NAME,model=OPENAI_EMBEDDING_MODEL_NAME,chunk_size = 1)

# define llm
#llm = ChatOpenAI(verbose=True, client=None, temperature=0)

# llm = AzureChatOpenAI(
#     azure_deployment=OPENAI_DEPLOYMENT_NAME,
#     model=OPENAI_MODEL_NAME, 
#     temperature=0.7, 
#     openai_api_version="2023-05-15"
# )
llm = ChatMistralAI(
    endpoint=MISTRAL_ENDPOINT,
    mistral_api_key=MISTRAL_KEY,
)


# memory for chatbot
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True, output_key="answer")

#@st.cache_data
def load_pdf(file):
    loader = PyPDFLoader(file)

    # split into pages where each page becomes a document
    docs = loader.load_and_split()

    return docs

#@st.cache_data
def init_faiss(_docs, pdf, pages):
    db = FAISS.from_documents(docs, embeddings)
    return db

def query(question):
	result = qa({"question": question})
	return result["answer"], result["source_documents"]

def get_text():
    input_text = st.text_input(key="input", label="Type a question and press Enter")
    return input_text
    

st.title("Ask questions about your PDFs")

## enumerate all the files in the uploads folder
files = os.listdir("uploads")

## create a dropdown menu of all the files
file = st.selectbox("Select a file", files)

## when file is selected, enable querying
if file:
    st.write("You selected", file)

    # load the file in the Uploads folder
    docs = load_pdf(os.path.join("uploads", file))

    # get FAISS db
    db = init_faiss(docs, file, len(docs))

    # init retrieval chain
    qa = ConversationalRetrievalChain.from_llm(llm, db.as_retriever(), memory=memory, return_source_documents=True)

    # keep track of generated responses
    if 'generated' not in st.session_state:
        st.session_state['generated'] = []

    if 'past' not in st.session_state:
        st.session_state['past'] = []

    user_input = get_text()

    if user_input:
        output, docs = query(user_input)

        sources = set()
        pages = set()

        for doc in docs:
            source = doc.metadata['source']
            page = str(doc.metadata['page']+1)
            sources.add(source)
            pages.add(page)

        unique_sources = list(sources)
        unique_pages = list(pages)

        # add the unique sources to the output with a line break between each source
        output += "\n\n" + "\n".join(unique_sources) + "\nPages: " + " ".join(unique_pages)

        st.session_state.past.append(user_input)
        st.session_state.generated.append(output)

    if st.session_state['generated']:
        for i in range(len(st.session_state['generated'])-1, -1, -1):
            message(st.session_state['past'][i], is_user=True, key=str(i) + '_user')
            message(st.session_state["generated"][i], key=str(i))




    


    
    
   