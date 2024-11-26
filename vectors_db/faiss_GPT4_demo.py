import os, json
from langchain import PromptTemplate, LLMChain
from langchain.llms import GPT4All
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.document_loaders import PyPDFLoader
from langchain.document_loaders import Docx2txtLoader
from langchain.document_loaders import TextLoader
import gradio as gr
from langchain.vectorstores import FAISS
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

loaders.append(os.path.join(doc_dir, "GREET.txt"))
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


chunk_size = 500
chunk_overlap = 0

r_splitter = RecursiveCharacterTextSplitter(
    chunk_size=chunk_size,
    chunk_overlap=chunk_overlap,
    separators=["Question"]
)
doc_splitter = r_splitter.split_documents(documents)

# print(doc_splitter)

embeddings = GPT4AllEmbeddings()

db = FAISS.from_documents(doc_splitter, embeddings)

# reteriving data
# retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 5})
# retriever = db.as_retriever(search_type="similarity_score_threshold", search_kwargs={"score_threshold": 0.6, "k": 5})


# storing memory in cache
memory = ConversationBufferMemory(
            memory_key="history",
            input_key="question")



def ques_responses(question_template, history, system_prompt, token):
    question = question_template
    retriever = db.similarity_search(question, k=1)
    instruction = '''1. If the user is greeting you then greet the user and tell your name.

                 2. If user is asking you a question then answer the question ={question}.

                 3. If you do not find any answer for any {question}, return "Sorry, I am not build to reply out of context question.

                 4. 

                 '''
   
    template = """ 
        {context}
        {instruction}
        
        Question: {question}

        Answer: """

 

    prompt = PromptTemplate(template=template, input_variables=["context","instruction","question"])
 
    # prompt = PromptTemplate(input_variables=["system", "context", "question"],template=template)
    # qa_chain = RetrievalQA.from_chain_type(llm, return_source_documents=True, retriever=retriever)
    qa_chain = LLMChain(llm=llm, prompt=prompt)

    # response = qa_chain.run({"query": question})
    response = qa_chain.predict(question = question, context = retriever, instruction = instruction)
    # # print(response)
    # if len(response['source_documents'])==0:
    #     return "Sorry, I am not build to answer out of context questions. Anything else I can help you with."
    return response



if __name__ == "__main__":
    gr.ChatInterface(ques_responses, additional_inputs=[
        gr.Textbox("You are Meera, an assistant AI chatbot", label="System Prompt"),
            gr.Slider(10, 100) 
               ]). queue().launch()

