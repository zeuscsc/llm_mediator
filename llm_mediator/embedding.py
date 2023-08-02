from transformers import AutoTokenizer, AutoModel
from .llm import LLM_Base
from torch import Tensor
import torch.nn.functional as F
class Embedding(LLM_Base):
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
    
    def get_embedding(self,sentences:str):
        if self.model is None:
            self.tokenizer = AutoTokenizer.from_pretrained(self.get_model_name())
            self.model = AutoModel.from_pretrained(self.get_model_name())
        batch_dict = self.tokenizer(sentences, max_length=512, padding=True, truncation=True, return_tensors='pt')
        outputs = self.model(**batch_dict)
        embeddings = Embedding.average_pool(outputs.last_hidden_state, batch_dict['attention_mask'])
        return embeddings
    
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