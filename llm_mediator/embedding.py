from sentence_transformers import SentenceTransformer
from .llm import LLM_Base
model:SentenceTransformer=None
class Embedding(LLM_Base):
    model=None
    def get_model_name(self):
        return "Bert"
    def set_model_name(self,model_name):
        return
    
    def get_embedding(self,sentences:str):
        if self.model is None:
            self.model = SentenceTransformer('intfloat/multilingual-e5-large')
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