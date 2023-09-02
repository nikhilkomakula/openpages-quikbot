# import system libraries
import os

# import environment loading library
from dotenv import load_dotenv

# import IBMGen library 
from genai.model import Credentials
from genai.schemas import GenerateParams 
from genai.extensions.langchain import LangChainInterface

# import LangChain library
from langchain.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import PyPDFDirectoryLoader
from langchain.chains.question_answering import load_qa_chain
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings

# get GenAI credentials
def get_genai_creds():
    load_dotenv(".env")
    api_key = os.getenv("GENAI_KEY", None)
    api_url = os.getenv("GENAI_API", None)
    if api_key is None or api_url is None:
        print("Either api_key or api_url is None. Please make sure your credentials are correct.")
    creds = Credentials(api_key, api_url)
    return creds

# populate chroma db
def generateDB():
    # load PDFs from folder
    loader = PyPDFDirectoryLoader(pdf_folder_path)
    documents = loader.load()    

    # load the document and split it into chunks
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0, separator='\n')
    docs = text_splitter.split_documents(documents)

    # create the open-source embedding function
    embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

    # save to disk
    db = Chroma.from_documents(docs, embedding_function, persist_directory="./db")
    
    return db

# generate response
def generateResponse(query, db):
    
    # retrieve results from chroma db
    results = db.similarity_search(query)
    
    # generate the response
    response = chain({"input_documents": results, "question": query})
    
    return response["output_text"]    

# ** start from here **

# variables
# ibm/mpt-7b-instruct -> 3/5
# meta-llama/llama-2-7b -> 3/5
# ibm/granite-13b-sft -> 3/5
# google/ul2 -> 3.5/5
model_id = 'google/ul2'
pdf_folder_path = './data'
db_folder_path = './db'
db = None

if [f for f in os.listdir(db_folder_path) if not f.startswith('.')] == []:
    print("Chroma DB is empty. Populating it.")
    
    # generate chroma db
    db = generateDB()
else:
    print("Chroma DB is not empty.")

    # create the open-source embedding function
    embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

    # load from disk
    db = Chroma(persist_directory="./db", embedding_function=embedding_function)

# get credentials
credentials = get_genai_creds()

# generate LLM params
params = GenerateParams(
            decoding_method='greedy', 
            min_new_tokens=1,
            max_new_tokens=200,
            stream=False,
            temperature=0.7,
            repetition_penalty=2)

# create a langchain interface to use with retrieved content
langchain_model = LangChainInterface(model=model_id, params=params, credentials=credentials)

# create the chain
chain = load_qa_chain(langchain_model, chain_type="stuff")

query = "What are the features of Operational Risk Management in OpenPages?"
print(generateResponse(query, db))