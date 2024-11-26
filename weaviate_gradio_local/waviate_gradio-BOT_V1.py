"""
This file is using Hugging Face Pipeline for generator and retriever.
We are using flan-t5-base model from google.
And integrated weaviate database in the
"""

import gradio as gr
import random
import os
import time
import weaviate
from langchain.vectorstores.weaviate import Weaviate
from langchain.llms import HuggingFacePipeline
from langchain import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
# from langchain.vectorstores import DocArrayInMemorySearch
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.document_loaders import JSONLoader
import json
from pathlib import Path
from pprint import pprint


client = weaviate.Client("http://localhost:8080")

# print(type(client))
client.schema.get()


# repo id address
repo_id = "google/flan-t5-base"
# repo_id = "deepset/xlm-roberta-base-squad2-distilled"

# defining llm model
llm = HuggingFacePipeline.from_model_id(
    model_id=repo_id,
    task="text2text-generation",
    model_kwargs={"temperature": 0.5, "do_sample":True, "max_length": 64},
)

# Read pdf from device
from langchain.document_loaders import PyPDFLoader
loader = PyPDFLoader("/Users/sonambharti/Documents/Assignments/weaviate-gradio/KT-documentation-by-Sonam.pdf")
data = loader.load()



# file_path='./data/lok_sabha/structured_lok_sabha.json'
# data = json.loads(Path(file_path).read_text())


# with open("./data/lok_sabha/structured_lok_sabha.json", "r") as f:
#     lok_sabha_data = json.loads(f.read())

# import  aspose.cells 
# from aspose.cells import Workbook


# workbook = Workbook("input.json")
# workbook.save("Output.docx")



# split the data
chunk_size =300
chunk_overlap = 40

# c_splitter = CharacterTextSplitter(
#     chunk_size=chunk_size,
#     chunk_overlap=chunk_overlap
# )

r_splitter = RecursiveCharacterTextSplitter(
    chunk_size=chunk_size,
    chunk_overlap=chunk_overlap
)


# model embeddings 
model_kwargs = {'device': 'cpu'}
encode_kwargs = {'normalize_embeddings': False}
embeddings = HuggingFaceEmbeddings(
    model_name=repo_id,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)

# doc_splitter = c_splitter.split_text(data)
# db = Weaviate.from_texts(doc_splitter, embeddings, client=client, by_text=False)
doc_splitter = r_splitter.split_documents(data)
db = Weaviate.from_documents(doc_splitter, embeddings, client=client, by_text=False)


# reteriving data
retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 5})

# storing memory in cache
memory = ConversationBufferMemory(
            memory_key="history",
            input_key="question")



def ques_responses(question_template, history, system_prompt, token):
    question = question_template

    template = """Use the following pieces of context to answer the question at the end. \
        You are Meera, a virtual parliamentary assistant. You are developed to assist people with the available documents.\
            If someone greets you, answer by `Greetings, I am Meera. How can I help you?` \
                If you don't know the answer, just say that you don't know, don't try to make up an answer.\
        {context}
        {history}
        Question: {question}
        Answer:"""
    
 
    prompt = PromptTemplate(input_variables=["history", "context", "question"],template=template)

    qa_chain = RetrievalQA.from_chain_type(llm, return_source_documents=True, retriever=retriever)

    # response = qa_chain.run({"query": question})
    response = qa_chain({"query": question})
    print(response)
    return response["result"]



if __name__ == "__main__":
    gr.ChatInterface(ques_responses, additional_inputs=[
        gr.Textbox("You are an assistant AI chatbot.", label="System Prompt"),
            gr.Slider(10, 100) 
               ]). queue().launch()


    
