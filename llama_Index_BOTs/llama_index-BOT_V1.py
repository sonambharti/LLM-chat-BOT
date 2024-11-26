from llama_cpp import Llama
# llm = Llama(model_path="/Users/sonambharti/Documents/loksabha_rag_chatbot/models/llama-2-7b-chat.Q2_K.gguf")

# output = llm("Q: Name the planets in the solar system? A: ", max_tokens=32, stop=["Q:", "\n"], echo=True)
# print(output)

from langchain.embeddings import HuggingFaceEmbeddings
from llama_index import (
    SimpleDirectoryReader,
    VectorStoreIndex,
    ServiceContext,
)
from llama_index.llms import LlamaCPP
from llama_index.llms.llama_utils import messages_to_prompt, completion_to_prompt
import gradio as gr


# model = "/Users/sonambharti/Documents/loksabha_rag_chatbot/models/llama-2-7b-chat.Q4_0.gguf"
model = "/Users/sonambharti/Documents/loksabha_rag_chatbot/models/llama-2-7b-chat.Q2_K.gguf"
llm = LlamaCPP(
    # You can pass in the URL to a GGML model to download it automatically
    # model_url="https://huggingface.co/TheBloke/Llama-2-7b-Chat-GGUF/resolve/main/llama-2-7b-chat.Q4_0.gguf",
    model_url="https://huggingface.co/TheBloke/Llama-2-7b-Chat-GGUF/resolve/main/llama-2-7b-chat.Q2_K.gguf",
    # optionally, you can set the path to a pre-downloaded model instead of model_url
    model_path=model,
    temperature=0.1,
    max_new_tokens=256,
    # llama2 has a context window of 4096 tokens, but we set it lower to allow for some wiggle room
    context_window=3900,
    # kwargs to pass to __call__()
    generate_kwargs={},
    # kwargs to pass to __init__()
    # set to at least 1 to use GPU
    # model_kwargs={"n_gpu_layers": 1},
    # transform inputs into Llama2 format
    messages_to_prompt=messages_to_prompt,
    completion_to_prompt=completion_to_prompt,
    verbose=True,
)


# use Huggingface embeddings
embed_model = HuggingFaceEmbeddings()

# create a service context
service_context = ServiceContext.from_defaults(
    llm=llm,
    embed_model=embed_model,
)

# load documents
documents = SimpleDirectoryReader(
    "/Users/sonambharti/Documents/loksabha_rag_chatbot/data/lok_sabha/heading_wise_ls"
).load_data()

# create vector store index
index = VectorStoreIndex.from_documents(documents, service_context=service_context)

# set up query engine
query_engine = index.as_query_engine()

response = query_engine.query("No. of Male MPs?")
print(response)



# if __name__ == "__main__":
#     gr.ChatInterface(ques_responses, additional_inputs=[
#         gr.Textbox("You are Meera, an assistant AI chatbot", label="System Prompt"),
#             gr.Slider(10, 100) 
#                ]). queue().launch()
