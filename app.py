# import system libraries
import os

# import environment loading library
from dotenv import load_dotenv

# import IBMGen library 
from genai.model import Credentials
from genai.schemas import GenerateParams 
from genai.extensions.langchain import LangChainInterface

# import LangChain library
from langchain.chains import RetrievalQA
from langchain.vectorstores import Chroma
from langchain.document_loaders import PDFMinerLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings

# initialize variables
pdf_folder_path = './data'
db_folder_path = './db'
model_id = 'google/ul2'
db = None

# get GenAI credentials
def get_genai_creds():
    load_dotenv(".env")
    api_key = os.getenv("GENAI_KEY", None)
    api_url = os.getenv("GENAI_API", None)
    if api_key is None or api_url is None:
        print("Either api_key or api_url is None. Please make sure your credentials are correct.")
    creds = Credentials(api_key, api_url)
    return creds

# define embedding function
def initEmbedFunc():
    embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    return embedding_function

# populate chroma db
def generateDB():
    docs = []
    for root, dirs, files in os.walk(pdf_folder_path):
        for file in files:
            if file.endswith(".pdf"):
                print(f'Reading File: {file}')
                
                # read PDF
                loader = PDFMinerLoader(os.path.join(root, file))
                documents = loader.load()

                # load the document and split it into chunks
                text_splitter = RecursiveCharacterTextSplitter(
                                    chunk_size=1000, 
                                    chunk_overlap=100, 
                                    separators=["\n"]
                )
                temp = text_splitter.split_documents(documents)
                
                # append to docs
                docs += temp

    # create the open-source embedding function
    embedding_function = initEmbedFunc()

    # save to disk
    db = Chroma.from_documents(docs, embedding_function, persist_directory=db_folder_path)
    
    return db

# generate response
def generateResponse(query, qa):    
    generated_text = qa(query)
    answer = generated_text['result']
    return answer     

# *** START HERE ***

if [f for f in os.listdir(db_folder_path) if not f.startswith('.')] == []:
    print("Chroma DB is empty. Generating indexes...")
    
    # generate chroma db
    db = generateDB()
else:
    print("Chroma DB is not empty. Using existing indexes!")

    # create the open-source embedding function
    embedding_function = initEmbedFunc()

    # load from disk
    db = Chroma(persist_directory=db_folder_path, embedding_function=embedding_function)

# get credentials
creds = get_genai_creds()

# generate LLM params
params = GenerateParams(
    decoding_method="sample",
    max_new_tokens=250,
    min_new_tokens=1,
    stream=False,
    temperature=0.55,
    top_k=50,
    top_p=1,
    repetition_penalty=1.5
)

# create a langchain interface to use with retrieved content
langchain_model = LangChainInterface(model=model_id, params=params, credentials=creds)

# create retrieval QA chain
retriever = db.as_retriever()
qa = RetrievalQA.from_chain_type(
        llm=langchain_model,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True
)

# FLASK CODE
from flask import Flask

app = Flask(__name__)

@app.route('/')
def hello():
    return "Hello World!"

@app.route('/test/<value>')
def test(value):
    return 'Here is the value passed: %s' % value

@app.route('/qa/<query>')
def respond(query):
    print('Here is the query: %s' % query)
    response = generateResponse(query, qa)
    print(f'Here is the response: {response}')
    return response

if __name__ == '__main__':
    app.run(debug=True)