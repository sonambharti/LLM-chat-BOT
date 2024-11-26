import logging
import sys
import faiss
from llama_index.vector_stores.faiss import FaissVectorStore
from IPython.display import Markdown, display
from llama_index import (
    SimpleDirectoryReader,
    load_index_from_storage,
    VectorStoreIndex,
    StorageContext,
)


logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))



# dimensions of text-ada-embedding-002
d = 1536
faiss_index = faiss.IndexFlatL2(d)

documents = SimpleDirectoryReader("/Users/sonambharti/Documents/loksabha_rag_chatbot/data/lok_sabha/heading_wise_ls").load_data()

vector_store = FaissVectorStore(faiss_index=faiss_index)
storage_context = StorageContext.from_defaults(vector_store=vector_store)
index = VectorStoreIndex.from_documents(documents, storage_context=storage_context)

# save index to disk
index.storage_context.persist()

#load index from disk
vector_store = FaissVectorStore.from_persist_dir("./storage")
storage_context = StorageContext.from_defaults(
    vector_store=vector_store, persist_dir="./storage"
)
index = load_index_from_storage(storage_context=storage_context)

#set logging to debug
query_engine = index.as_query_engine()
response = query_engine.query("Thanks")
print(response)
