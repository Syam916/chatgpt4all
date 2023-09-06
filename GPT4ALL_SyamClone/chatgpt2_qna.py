import datetime
from langchain import PromptTemplate, LLMChain
from langchain.llms import LlamaCpp
from langchain.llms import GPT4All
from langchain.callbacks.base import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
# function for loading only TXT files
from langchain.document_loaders import TextLoader
# text splitter for create chunks
from langchain.text_splitter import RecursiveCharacterTextSplitter
# to be able to load the pdf files
from langchain.document_loaders import UnstructuredPDFLoader
from langchain.document_loaders import PyPDFLoader
from langchain.document_loaders import DirectoryLoader
# Vector Store Index to create our database about our knowledge
from langchain.indexes import VectorstoreIndexCreator
# LLamaCpp embeddings from the Alpaca model
from langchain.embeddings import LlamaCppEmbeddings
# FAISS  library for similaarity search
from langchain.vectorstores.faiss import FAISS
import os  #for interaaction with the fil

# Paths for models, embeddings, and index
gpt4all_path = './models/gpt4all-converted.bin'
llama_path = './models/ggml-model-q4_0.bin'
embeddings_path = './embeddings'  # Path to store embeddings
index_path = './index'  # Path to store the FAISS index

# Callback manager for handling calls with the model
callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])

# Create the GPT4All llm object
llm = GPT4All(model=gpt4all_path, callback_manager=callback_manager, verbose=True)

# Function to create embeddings
def create_embeddings(chunks):
    texts = [doc.page_content for doc in chunks]
    metadatas = [doc.metadata for doc in chunks]

    # Create embeddings and save them to the specified path
    embeddings = LlamaCppEmbeddings(model_path=llama_path)
    embeddings.create_embeddings(texts, metadatas, embeddings_path)

    return embeddings

# Function to load embeddings
def load_embeddings(embeddings_path):
    return LlamaCppEmbeddings(model_path=embeddings_path)

# Split text
def split_chunks(sources):
    chunks = []
    splitter = RecursiveCharacterTextSplitter(chunk_size=256, chunk_overlap=32)
    for chunk in splitter.split_documents(sources):
        chunks.append(chunk)
    return chunks

# Create FAISS index
def create_index(chunks, index_path):
    texts = [doc.page_content for doc in chunks]
    metadatas = [doc.metadata for doc in chunks]

    # Create or load the FAISS index
    if os.path.exists(index_path):
        search_index = FAISS.load(index_path)
    else:
        search_index = FAISS.from_texts(texts, embeddings, metadatas=metadatas)
        search_index.save(index_path)

    return search_index

# Similarity search
def similarity_search(query, index):
    matched_docs = index.similarity_search(query, k=3)
    sources = []
    for doc in matched_docs:
        sources.append(
            {
                "page_content": doc.page_content,
                "metadata": doc.metadata,
            }
        )

    return matched_docs, sources

# Directory containing PDF files
pdf_folder_path = './docs1'

# List PDF files
doc_list = [s for s in os.listdir(pdf_folder_path) if s.endswith('.pdf') or s.endswith('.PDF')]
print(doc_list)
num_of_docs = len(doc_list)
print('Total number of documents are:', num_of_docs)

# Create a loader for the PDFs from the path
general_start = datetime.datetime.now()
print("Starting the loop...")
loop_start = datetime.datetime.now()

# Load or create embeddings
embeddings = load_embeddings(embeddings_path) if os.path.exists(embeddings_path) else None

# Initialize the index
if os.path.exists(index_path):
    db0 = FAISS.load(index_path)
else:
    db0 = None

for i in range(num_of_docs):
    print(doc_list[i])
    print(f"Loop position {i}")
    loader = PyPDFLoader(os.path.join(pdf_folder_path, doc_list[i]))
    start = datetime.datetime.now()
    docs = loader.load()
    chunks = split_chunks(docs)

    if embeddings is None:
        embeddings = create_embeddings(chunks)

    if db0 is None:
        db0 = create_index(chunks, index_path)
    else:
        dbi = create_index(chunks, index_path)
        db0.merge_from(dbi)

    end = datetime.datetime.now()
    elapsed = end - start
    print(f"Completed in {elapsed}")
    print("-----------------------------------")

print(f"All documents processed in {datetime.datetime.now() - loop_start}")
print(f"The database is done with {num_of_docs} subset of the DB index")
print("-----------------------------------")
print(f"Merging completed")
print("-----------------------------------")
print("Saving Merged Database Locally")
db0.save(index_path)
print("-----------------------------------")
print("Merged database saved as my_faiss_index")
general_end = datetime.datetime.now()
print(f"All indexing completed in {general_end - general_start}")
print("-----------------------------------")
