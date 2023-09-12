from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer
from .llm import LLM_Base
import torch
from torch import Tensor
import torch.nn.functional as F
import pynvml

EMBEDDING_SIZE=1024

class Embedding(LLM_Base):
    embedding_size=EMBEDDING_SIZE
    model_name="intfloat/multilingual-e5-large"
    tokenizer:AutoTokenizer=None
    model:AutoModel=None

    def average_pool(last_hidden_states: Tensor,attention_mask: Tensor) -> Tensor:
        last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
        return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]


    def get_model_name(self):
        return self.model_name
    def set_model_name(self,model_name):
        self.model_name=model_name
    

    def get_available_vram():
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        return Embedding.bytes_to_gb(mem_info.free)
    def bytes_to_gb(size_in_bytes):
        gb = size_in_bytes / (1024 ** 3)  # 1 GB = 1024^3 bytes
        return gb
    def have_available_gpu():
        if torch.cuda.is_available():
            return Embedding.get_available_vram()>1
        return False
    def get_embeddings(self,sentences:str|list[str]):
        origin_sentences_type=type(sentences)
        if self.model is None:
            self.tokenizer = AutoTokenizer.from_pretrained(self.get_model_name())
            if Embedding.have_available_gpu():
                self.model = AutoModel.from_pretrained(self.get_model_name(),device_map="auto")
            else:
                self.model = AutoModel.from_pretrained(self.get_model_name())
        if origin_sentences_type.__name__=="str":
            sentences=[sentences]
        batch_dict = self.tokenizer(sentences, max_length=512, padding=True, truncation=True, return_tensors='pt')
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 
        batch_dict.to(device)
        outputs = self.model(**batch_dict)
        embeddings = Embedding.average_pool(outputs.last_hidden_state, batch_dict['attention_mask'])
        if origin_sentences_type.__name__=="str":
            return embeddings.detach().numpy()[0]
        return embeddings.detach().numpy()
    
    def get_response(self,system,assistant,user):
        return
    def get_response_stream(self,system,assistant,user):
        return
    def get_conversation_stream(self,messages):
        return
    def get_conversation_response(self,messages):
        return
    def detect_if_tokens_oversized(self,e):
        return False
    def get_functions_response(self,messages,functions):
        return