from langchain import PromptTemplate, LLMChain
from langchain.llms import LlamaCpp
from langchain.llms import GPT4All
from langchain.callbacks.base import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredPDFLoader
from langchain.document_loaders import PyPDFLoader
from langchain.document_loaders import DirectoryLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain.embeddings import LlamaCppEmbeddings
from langchain.vectorstores.faiss import FAISS
import os
import datetime

# Paths for the models
gpt4all_path = './models/gpt4all-converted.bin'
llama_path = './models/ggml-model-q4_0.bin'

# Callback manager
callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])

# Create the embeddings object (load only once)
embeddings = LlamaCppEmbeddings(model_path=llama_path)

# Split text
def split_chunks(sources):
    chunks = []
    splitter = RecursiveCharacterTextSplitter(chunk_size=256, chunk_overlap=32)
    for chunk in splitter.split_documents(sources):
        chunks.append(chunk)
    return chunks

# Create or load the language model (GPT-4)
def create_or_load_llm(llm_path):
    if os.path.exists(llm_path):
        llm = GPT4All(model=llm_path, callback_manager=callback_manager, verbose=True)
    else:
        llm = GPT4All(callback_manager=callback_manager, verbose=True)
        llm.save(llm_path)  # Save the newly created model
    return llm

# Create or load the embeddings index
def create_or_load_embeddings_index(embeddings, pdf_folder_path, doc_list):
    # Create an empty index or load an existing one
    if os.path.exists("my_faiss_index1"):
        search_index = FAISS.load_local("my_faiss_index1",embeddings)
    else:
        search_index = create_index(embeddings, pdf_folder_path, doc_list)
    return search_index

# Create the FAISS index
def create_index(embeddings, pdf_folder_path, doc_list):
    texts = []
    metadatas = []

    for doc_file in doc_list:
        loader = PyPDFLoader(os.path.join(pdf_folder_path, doc_file))
        docs = loader.load()
        chunks = split_chunks(docs)
        for chunk in chunks:
            texts.append(chunk.page_content)
            metadatas.append(chunk.metadata)

    search_index = FAISS.from_texts(texts, embeddings, metadatas=metadatas)
    return search_index

# Perform similarity search
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

# Get the list of PDF files
pdf_folder_path = './docs1'
doc_list = [s for s in os.listdir(pdf_folder_path) if s.endswith('.pdf') or s.endswith('.PDF')]
print(doc_list)
num_of_docs = len(doc_list)
print('Total no of documents are : ', num_of_docs)

# Create or load the language model
llm = create_or_load_llm(gpt4all_path)

# Create or load the embeddings index
search_index = create_or_load_embeddings_index(embeddings, pdf_folder_path, doc_list)

# Document processing loop (for new data)
for i in range(num_of_docs):
    print(doc_list[i])
    print(f"loop position {i}")
    loader = PyPDFLoader(os.path.join(pdf_folder_path, doc_list[i]))
    start = datetime.datetime.now()
    docs = loader.load()
    chunks = split_chunks(docs)
    dbi = create_index(embeddings, pdf_folder_path, [doc_list[i]])
    print("start merging with search_index...")
    search_index.merge_from(dbi)
    end = datetime.datetime.now()
    elapsed = end - start
    print(f"completed in {elapsed}")
    print("-----------------------------------")

# Save the merged database locally
search_index.save_local("my_faiss_index1")

print("-----------------------------------")
print("Merged database saved as my_faiss_index1")
