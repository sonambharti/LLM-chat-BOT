import os, json
from pprint import pprint
from haystack.utils import print_answers
from langchain.document_loaders import PyPDFLoader
from langchain.document_loaders import Docx2txtLoader
from langchain.document_loaders import TextLoader
import gradio as gr
import weaviate
from langchain.vectorstores.weaviate import Weaviate
from langchain.llms import HuggingFacePipeline
from langchain import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.document_loaders import JSONLoader
from pathlib import Path



doc_dir = "/Users/sonambharti/Documents/loksabha_rag_chatbot/data/lok_sabha/heading_wise_ls"
# meta_data = []
loaders = []

with open("/Users/sonambharti/Documents/loksabha_rag_chatbot/data/lok_sabha/structured_lok_sabha.json", "r") as f:
    lok_sabha_data = json.loads(f.read())
for heading, qa_pairs in lok_sabha_data.items():
    # sub_meta_data = {}
    # sub_meta_data["document"] = heading
    # meta_data.append(sub_meta_data)
    load_file = os.path.join(doc_dir, heading+".txt")
    loaders.append(load_file)
    
# print(meta_data)


# print(loaders)

documents = []

for file in loaders:
    if file.endswith(".pdf"):
        loader = PyPDFLoader(file)
        documents.extend(loader.load())
    elif file.endswith('.docx') or file.endswith('.doc'):
        loader = Docx2txtLoader(file)
        documents.extend(loader.load())
    elif file.endswith('.txt'):
        loader = TextLoader(file)
        documents.extend(loader.load())


client = weaviate.Client("http://localhost:8080")

# print(type(client))
client.schema.get()


# repo id address
repo_id = "google/flan-t5-base"


# defining llm model
llm = HuggingFacePipeline.from_model_id(
    model_id=repo_id,
    task="text2text-generation",
    model_kwargs={"temperature": 0, "max_length": 512, "do_sample":False},
)

chunk_size = 300
chunk_overlap = 100

r_splitter = RecursiveCharacterTextSplitter(
    chunk_size=chunk_size,
    chunk_overlap=chunk_overlap,
    separators=["Question"]
)
doc_splitter = r_splitter.split_documents(documents)

# print(doc_splitter)

model_kwargs = {'device': 'cpu'}
encode_kwargs = {'normalize_embeddings': False}
embeddings = HuggingFaceEmbeddings(
    model_name=repo_id,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)

db = Weaviate.from_documents(doc_splitter, embeddings, client=client, by_text=False)


# reteriving data
# retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 5})
retriever = db.as_retriever(search_type="similarity_score_threshold", search_kwargs={"score_threshold": 0.6, "k": 5})

# storing memory in cache
memory = ConversationBufferMemory(
            memory_key="history",
            input_key="question")



def ques_responses(question_template, history, system_prompt, token):
    question = question_template

    # template = """Use the following pieces of context to answer the question at the end. \
    #     You are a virtual parliamentary assistant, and your name is Meera. You are developed to assist people with the available documents.\
    #         If user greets you, answer by `Greetings, I am Meera. How can I help you?` \
            
    #             If you don't know the answer, just say that you don't know, don't try to make up an answer.\
    #     {context}
    #     {history}
    #     Question: {question}
    #     Answer:"""

    template = """Instruction: You are virtual assistant named Meera. \n\n \
        When out of context question is asked respond "I don't know."\n\n \
            User: Hi How are you
            Assistant: I am good. How are you? How can I help you today?
            User: I need to know about earth.
            Assistant: I don't know about it. How else I can help you with?
            
            User: {question}
            Assistant:"""
    
 
    # prompt = PromptTemplate(input_variables=["history", "context", "question"],template=template)
    prompt = PromptTemplate(input_variables=["question"],template=template)
    

    qa_chain = RetrievalQA.from_chain_type(llm, return_source_documents=True, retriever=retriever)

    # response = qa_chain.run({"query": question})
    response = qa_chain({"query": question})
    # print(response)
    print(response['source_documents'])
    return response["result"]



if __name__ == "__main__":
    gr.ChatInterface(ques_responses, additional_inputs=[
        gr.Textbox("You are an assistant AI chatbot.", label="System Prompt"),
            gr.Slider(10, 100) 
               ]). queue().launch()
