from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer
from .llm import LLM_Base
import torch
from torch import Tensor
import torch.nn.functional as F

EMBEDDING_SIZE=768

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
    
    def get_embeddings(self,sentences:str|list[str]):
        origin_sentences_type=type(sentences)
        if self.model is None:
            self.tokenizer = AutoTokenizer.from_pretrained(self.get_model_name())
            self.model = AutoModel.from_pretrained(self.get_model_name())
        if origin_sentences_type.__name__=="str":
            sentences=[sentences]
        batch_dict = self.tokenizer(sentences, max_length=512, padding=True, truncation=True, return_tensors='pt')
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

# class Embedding(LLM_Base):
#     model_name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
#     tokenizer:AutoTokenizer=None
#     model:AutoModel=None

#     def mean_pooling(model_output, attention_mask):
#         token_embeddings = model_output[0] #First element of model_output contains all token embeddings
#         input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
#         return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


#     def get_model_name(self):
#         return self.model_name
#     def set_model_name(self,model_name):
#         self.model_name=model_name
    
#     def get_embedding(self,sentences:str|list[str]):
#         # if self.model is None:
#         #     self.model = SentenceTransformer(self.get_model_name())
#         # embeddings = self.model.encode(sentences)
#         # # print(embeddings.shape)
#         # return embeddings
#         origin_sentences_type=type(sentences)
#         # print(origin_sentences_type)
#         if self.model is None:
#             self.tokenizer = AutoTokenizer.from_pretrained(self.get_model_name())
#             self.model = AutoModel.from_pretrained(self.get_model_name())
#         if origin_sentences_type.__name__=="str":
#             sentences=[sentences]
#         encoded_input = self.tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')
#         with torch.no_grad():
#             model_output = self.model(**encoded_input)
#         sentence_embeddings = Embedding.mean_pooling(model_output, encoded_input['attention_mask'])
#         if origin_sentences_type.__name__=="str":
#             # print(sentence_embeddings[0].shape)
#             return sentence_embeddings.detach().numpy()[0]
#         # print(sentence_embeddings.shape)
#         return sentence_embeddings.detach().numpy()
    
#     def get_response(self,system,assistant,user):
#         return
#     def get_response_stream(self,system,assistant,user):
#         return
#     def get_conversation_stream(self,messages):
#         return
#     def get_conversation_response(self,messages):
#         return
#     def detect_if_tokens_oversized(self,e):
#         return False

