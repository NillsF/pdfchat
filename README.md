# PDF Chat

Chat with a PDF file using Azure OpenAI and Langchain. Requires Python. Install Python dependencies with `pip install -r requirements.txt`

If you do not have an OpenAI account, go to https://platform.openai.com. You need an account for both the embeddings and the calls to the chat model (gpt-3.5-turbo).

Add an `.env` file to the same folder that contains `Query.py ` with following contents:

```bash
AZURE_OPENAI_API_KEY = "xxx"
AZURE_OPENAI_ENDPOINT = "https://xxx.openai.azure.com/"
OPENAI_DEPLOYMENT_NAME = "nillsfgpt35turbo"
OPENAI_MODEL_NAME = "gpt-35-turbo"
OPENAI_EMBEDDING_DEPLOYMENT_NAME = "nillsf-embeddings"
OPENAI_EMBEDDING_MODEL_NAME = "text-embedding-ada-002"
OPENAI_DEPLOYMENT_VERSION = "0301"
MISTRAL_KEY="xxx"
MISTRAL_ENDPOINT="https://xxx.eastus2.inference.ai.azure.com"
```

Create an `uploads` folder in the same folder that contains `Query.py`.

To run, use `streamlit run Query.py`. A brower window should open with the app. Upload a PDF and start asking questions.

**Notes:** 
- uses in memory FAISS database as a vector store and only tested with a smaller PDFs (around 10MB) 
- max upload size in 200MB
- you can only query one PDF at a time in this version
- because embeddings are not saved, when you restart the app or clear the Streamlit cache, they have to be recreated which means $ 😃