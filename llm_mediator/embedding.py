from transformers import AutoTokenizer, AutoModel
from .llm import LLM_Base
import torch
from torch import Tensor
import torch.nn.functional as F
class Embedding(LLM_Base):
    model_name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
    tokenizer:AutoTokenizer=None
    model:AutoModel=None

    def mean_pooling(model_output, attention_mask):
        token_embeddings = model_output[0] #First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)



    def get_model_name(self):
        return self.model_name
    def set_model_name(self,model_name):
        self.model_name=model_name
    
    def get_embedding(self,sentences:str):
        if self.model is None:
            self.tokenizer = AutoTokenizer.from_pretrained(self.get_model_name())
            self.model = AutoModel.from_pretrained(self.get_model_name())
        if isinstance(sentences,str):
            sentences=[sentences]
        for sentence in sentences:
            sentence=f"query: {sentence}"
        encoded_input = self.tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')
        with torch.no_grad():
            model_output = self.model(**encoded_input)
        sentence_embeddings = self.mean_pooling(model_output, encoded_input['attention_mask'])
        return sentence_embeddings
    
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