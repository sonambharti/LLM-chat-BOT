import os, json
from langchain import PromptTemplate, LLMChain
from langchain.llms import GPT4All
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.document_loaders import PyPDFLoader
from langchain.document_loaders import Docx2txtLoader
from langchain.document_loaders import TextLoader
import gradio as gr
import weaviate
from langchain.vectorstores.weaviate import Weaviate
from langchain.llms import HuggingFacePipeline
from langchain import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import GPT4AllEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA




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
repo_id = "/Users/sonambharti/Documents/loksabha_rag_chatbot/models/orca-mini-3b.ggmlv3.q4_0.bin"


# defining llm model
# llm = HuggingFacePipeline.from_model_id(
#     model_id=repo_id,
#     task="text-generation",
#     model_kwargs={"temperature": 0, "max_length": 512, "do_sample":False},
# )

# Callbacks support token-wise streaming
callbacks = [StreamingStdOutCallbackHandler()]

# Verbose is required to pass to the callback manager
# llm = GPT4All(model=repo_id, callbacks=callbacks, verbose=True)

llm = GPT4All(model=repo_id, 
              callbacks=callbacks, 
              verbose=True,
              repeat_penalty = 1,
              temp=0,
              top_p=0.1,
              top_k = 4,)


chunk_size = 300
chunk_overlap = 50

r_splitter = RecursiveCharacterTextSplitter(
    chunk_size=chunk_size,
    chunk_overlap=chunk_overlap,
    separators=["Question"]
)
doc_splitter = r_splitter.split_documents(documents)

# print(doc_splitter)

embeddings = GPT4AllEmbeddings()

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
    #             If you don't know the answer, just say, "Sorry, I am not build to answer out of the context questions." And, don't try to make up an answer.\
    #     {context}
    #     {history}
    #     Question: {question}
    #     Answer:"""

    template = """
        Instruction: You are virtual assistant named Meera. \n\n
        You have to respond in positive sentiment.\n\n
        Whenever there is question from outside the context and you don't know the answer just don't try to make up an answer.\n\n
        Ask User, "Sorry, I am not build to answer out of context questions. Anything else I can help you with."\n\n

        #####
        Examples

        User: Hi, How are you
        Assistant: Hello, I am good. How are you? How can I help you today?

        User: Hi
        Assistant: Hi, I am Meera. How are you?

        User: I need to know about earth.
        Assistant: Sorry, I am not build to answer out of context questions. Anything else I can help you with.
        #####

        {context}
        {history}
        User: {question}
        
        Assistant:"""
    
 
    prompt = PromptTemplate(input_variables=["history", "context", "question"],template=template)
    
    

    qa_chain = RetrievalQA.from_chain_type(llm, return_source_documents=True, retriever=retriever)

    # response = qa_chain.run({"query": question})
    response = qa_chain({"query": question})
    # print(response)
    print(response['source_documents'])
    return response["result"]



if __name__ == "__main__":
    gr.ChatInterface(ques_responses, additional_inputs=[
        gr.Textbox("You are Meera, an assistant AI chatbot", label="System Prompt"),
            gr.Slider(10, 100) 
               ]). queue().launch()

