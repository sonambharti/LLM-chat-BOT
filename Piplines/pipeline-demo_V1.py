#https://github.com/jerryjliu/llama_index/blob/046183303da4161ee027026becf25fb48b67a3d2/docs/how_to/custom_llms.md#example-using-a-custom-llm-model

import torch
from langchain.llms.base import LLM
from llama_index import SimpleDirectoryReader, LangchainEmbedding, SummaryIndex, PromptHelper
from llama_index import LLMPredictor
from transformers import pipeline
from typing import Optional, List, Mapping, Any


# define prompt helper
# set maximum input size
max_input_size = 2048
# set number of output tokens
num_output = 256
# set maximum chunk overlap
max_chunk_overlap = 20
prompt_helper = PromptHelper(context_window=max_input_size, num_output=num_output, chunk_overlap_ratio=0.1)


class CustomLLM(LLM):
    
    # model_name = "TheBloke/Llama-2-7b-Chat-GGUF"
    model_name = "facebook/opt-iml-max-1.3b"
    # model_name = "TheBloke/Llama-2-7b-Chat-GGUF"
    pipeline = pipeline("text-generation", 
                        model=model_name, 
                        # device="cuda:0", 
                        # model_kwargs={"torch_dtype":torch.bfloat16}
                        )

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        prompt_length = len(prompt)
        response = self.pipeline(prompt, max_new_tokens=num_output)[0]["generated_text"]

        # only return newly generated tokens
        return response[prompt_length:]

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        return {"name_of_model": self.model_name}

    @property
    def _llm_type(self) -> str:
        return "custom"

# define our LLM
llm_predictor = LLMPredictor(llm=CustomLLM())

# Load the your data
documents = SimpleDirectoryReader('../data/lok_sabha/heading_wise_ls').load_data()

index = SummaryIndex(documents, llm_predictor=llm_predictor, prompt_helper=prompt_helper)

# Query and print response
response = index.from_documents("Hi How are you?")
print(response)