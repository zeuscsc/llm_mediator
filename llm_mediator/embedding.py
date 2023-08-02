from transformers import AutoTokenizer, AutoModel
from .llm import LLM_Base
class Embedding(LLM_Base):
    model_name="intfloat/multilingual-e5-large"
    tokenizer:AutoTokenizer=None
    model:AutoModel=None
    def get_model_name(self):
        return self.model_name
    def set_model_name(self,model_name):
        self.model_name=model_name
    
    def get_embedding(self,sentences:str):
        if self.model is None:
            self.tokenizer = AutoTokenizer.from_pretrained(self.get_model_name())
            self.model = AutoModel.from_pretrained(self.get_model_name())
        return self.model.encode(sentences)
    
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