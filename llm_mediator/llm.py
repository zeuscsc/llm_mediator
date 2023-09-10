import os
import json
import datetime
import hashlib
import glob
from abc import ABC, abstractmethod
from typing import Any, Callable, Type,Generator
from .folders import LLM_RESPONSE_CACHE_FOLDER,LLM_STREAM_RESPONSE_CACHE_FOLDER,LLM_CONVERSATION_STREAM_CACHE_FOLDER,LLM_CONVERSATION_CACHE_FOLDER
from time import sleep
import numpy as np

ON_TOKENS_OVERSIZED="on_tokens_oversized"
COMPLETED_STREAM_CACHE={
    "object": "chat.completion.chunk",
    "model": "gpt-4-32k",
    "choices": [
        {
            "index": 0,
            "finish_reason": "stop",
            "delta": {}
        }
    ],
    "usage": None
}

def calculate_md5(string:str):
    md5_hash = hashlib.md5(string.encode()).hexdigest()
    return md5_hash

class CallStack:
    system:str
    assistant:str
    user:str
    response:str
    def __init__(self,system,assistant,user,response) -> None:
        self.system=system
        self.assistant=assistant
        self.user=user
        self.response=response
        pass
    pass
class _LLM_Base(ABC):
    def __init__(self) -> None:
        self.model_name:str=None
        self.use_cache:bool=True
        self.on_each_response:Type[object]=None
        self.on_chunked:Type[object]=None
        pass
    def load_response_cache(model,system,assistant,user):
        try:
            hashed_request=calculate_md5(f"{model}{system}{assistant}{user}")
            # print(f"Loading response cache for {model} model with id: {hashed_request}...")
            matching_files = glob.glob(f"{LLM_RESPONSE_CACHE_FOLDER}/{hashed_request}/*.json")
            if len(matching_files)>0:
                with open(matching_files[-1], "r",encoding="utf8") as chat_cache_file:
                    chat_cache = json.load(chat_cache_file)
                    return chat_cache
        except Exception as e:
            print(e)
        return None
    def save_response_cache(model,system,assistant,user,chat_cache):
        hashed_request=calculate_md5(f"{model}{system}{assistant}{user}")
        messages=[{"role": "system","content": system},{"role": "user","content": user},{"role": "assistant","content": assistant}]
        _LLM_Base.save_request_log(model,hashed_request,messages,LLM_RESPONSE_CACHE_FOLDER)
        _LLM_Base.save_cache(model,hashed_request,chat_cache,LLM_RESPONSE_CACHE_FOLDER)
    def save_stream_response_cache(model,system,assistant,user,chat_cache,combined=False):
        hashed_request=calculate_md5(f"{model}{system}{assistant}{user}")
        messages=[{"role": "system","content": system},{"role": "user","content": user},{"role": "assistant","content": assistant}]
        _LLM_Base.save_request_log(model,hashed_request,messages,LLM_STREAM_RESPONSE_CACHE_FOLDER)
        _LLM_Base.save_cache(model,hashed_request,chat_cache,LLM_STREAM_RESPONSE_CACHE_FOLDER,combined=combined)
    def save_conversation_stream_cache(model,messages,chat_cache):
        hashed_request=calculate_md5(f"{model}{json.dumps(messages)}")
        _LLM_Base.save_request_log(model,hashed_request,messages,LLM_CONVERSATION_STREAM_CACHE_FOLDER)
        _LLM_Base.save_cache(model,hashed_request,chat_cache,LLM_CONVERSATION_STREAM_CACHE_FOLDER)
    def save_conversation_cache(model,messages,chat_cache):
        hashed_request=calculate_md5(f"{model}{json.dumps(messages)}")
        _LLM_Base.save_request_log(model,hashed_request,messages,LLM_CONVERSATION_CACHE_FOLDER)
        _LLM_Base.save_cache(model,hashed_request,chat_cache,LLM_CONVERSATION_CACHE_FOLDER)
    def save_cache(model,hashed_request,chat_cache,folder_path,combined=False):
        matching_files = glob.glob(f"{folder_path}/{hashed_request}/*.json")
        file_index=len(matching_files)
        if combined:
            file_index="combined"
        os.makedirs(f"{folder_path}/{hashed_request}", exist_ok=True)
        with open(f"{folder_path}/{hashed_request}/{file_index}.json", "w",encoding="utf8") as temp_file:
            json.dump(chat_cache, temp_file, ensure_ascii=False)
    def save_request_log(model,hashed_request,messages,folder_path):
        request={"messages":messages,"folder_path":folder_path,"hashed_request":hashed_request}
        os.makedirs(f"{folder_path}_requests_logs", exist_ok=True)
        matching_files = glob.glob(f"{folder_path}_requests_logs/{hashed_request}/*.json")
        file_index=len(matching_files)
        os.makedirs(f"{folder_path}_requests_logs/{hashed_request}", exist_ok=True)
        if file_index==0:
            with open(f"{folder_path}_requests_logs/{hashed_request}/{file_index}.json", "w",encoding="utf8") as temp_file:
                json.dump(request, temp_file, ensure_ascii=False)
    def delete_response_cache(model,system,assistant,user):
        _LLM_Base.delete_cache(model,system,assistant,user,LLM_RESPONSE_CACHE_FOLDER)
    def delete_stream_response_cache(model,system,assistant,user):
        _LLM_Base.delete_cache(model,system,assistant,user,LLM_STREAM_RESPONSE_CACHE_FOLDER)
    def delete_cache(model,system,assistant,user,folder_path):
        hashed_request=calculate_md5(f"{model}{system}{assistant}{user}")
        print(f"Deleting response cache for {model} model with id: {hashed_request}...")
        matching_files = glob.glob(f"{folder_path}/{hashed_request}/*.json")
        for file in matching_files:
            os.remove(file)
    def split_text_in_half_if_too_large(text:str,max_tokens=10000):
        words = text.split()
        results = []
        
        if len(words) <= max_tokens:
            results.append(' '.join(words))
        else:
            half = len(words) // 2
            results.extend(_LLM_Base.split_text_in_half_if_too_large(' '.join(words[:half])))
            results.extend(_LLM_Base.split_text_in_half_if_too_large(' '.join(words[half:])))
        return results
    def split_text_in_half(text:str):
        words = text.split()
        results = []
        half = len(words) // 2
        results.extend(_LLM_Base.split_text_in_half_if_too_large(' '.join(words[:half])))
        results.extend(_LLM_Base.split_text_in_half_if_too_large(' '.join(words[half:])))
        return results

    @abstractmethod
    def get_model_name(self):
        pass
    @abstractmethod
    def set_model_name(self,name):
        pass
    @abstractmethod
    def detect_if_tokens_oversized(self,e):
        pass
    @abstractmethod
    def get_response(self,system,assistant,user)->str:
        pass
    @abstractmethod
    def get_response_stream(self,system,assistant,user)->Generator[Any,Any,None]:
        pass
    @abstractmethod
    def get_conversation_stream(self,messages)->Generator[Any,Any,None]:
        pass
    @abstractmethod
    def get_conversation_response(self,messages)->str:
        pass
    @abstractmethod
    def get_embeddings(self,sentences:str|list[str])->np.ndarray:
        pass
    @abstractmethod
    def get_functions_response(self,messages:str|list[str],functions:list[dict]):
        pass
    def set_event_listener(self,event_name:str,func:Callable[[Any], Any]):
        if event_name=="on_chunked":
            self.on_chunked=func
        elif event_name=="on_each_response":
            self.on_each_response=func
        elif event_name=="on_tokens_oversized":
            self.on_tokens_oversized=func
    
    def is_incomplete_stream_cache(self,chat_cache):
        if "choices" in chat_cache:
            if len(chat_cache["choices"])>0:
                if "finish_reason" in chat_cache["choices"][0]:
                    if chat_cache["choices"][0]["finish_reason"]!="stop":
                        return True
        return False
    
    def on_tokens_oversized(self,e,system,assistant,user):
        if self.detect_if_tokens_oversized(e):
            print("Splitting text in half...")
            chunks = []
            chunks.extend(_LLM_Base.split_text_in_half(user))
            responses=""
            for chunk in chunks:
                try:
                    response=self.get_response(system,assistant,chunk)
                except Exception as e:
                    print(e)
                    continue
                if response is not None:
                    if self.on_chunked is None:
                        responses+=response
                    else:
                        responses=self.on_chunked(system,assistant,chunk,responses,response)
            return responses
    
    def have_stream_response_cache(self,model,system,assistant,user):
        hashed_request=calculate_md5(f"{model}{system}{assistant}{user}")
        return os.path.exists(f"{LLM_STREAM_RESPONSE_CACHE_FOLDER}/{hashed_request}")
    def load_stream_response_cache(self,model,system,assistant,user):
        hashed_request=calculate_md5(f"{model}{system}{assistant}{user}")
        # print(f"Loading response cache for {model} model with id: {hashed_request}...")
        if self.have_stream_response_cache(model,system,assistant,user):
            if os.path.exists(f"{LLM_STREAM_RESPONSE_CACHE_FOLDER}/{hashed_request}/combined.json"):
                with open(f"{LLM_STREAM_RESPONSE_CACHE_FOLDER}/{hashed_request}/combined.json", "r",encoding="utf8") as chat_cache_file:
                    chat_cache = json.load(chat_cache_file)
                    for chunk in chat_cache:
                        yield chunk
                    return
            matching_files = glob.glob(f"{LLM_STREAM_RESPONSE_CACHE_FOLDER}/{hashed_request}/*.json")
            matching_files=sorted(matching_files, key=lambda x: int(os.path.basename(x).split(".")[0]))
            for path in matching_files:
                with open(path, "r",encoding="utf8") as chat_cache_file:
                    chat_cache = json.load(chat_cache_file)
                    yield chat_cache
            if self.is_incomplete_stream_cache(chat_cache):
                chat_cache=COMPLETED_STREAM_CACHE
                yield chat_cache
                import shutil
                shutil.rmtree(f"{LLM_STREAM_RESPONSE_CACHE_FOLDER}/{hashed_request}")
        return
    
    def combine_stream_response_cache(self,system,assistant,user):
        model=self.get_model_name()
        hashed_request=calculate_md5(f"{model}{system}{assistant}{user}")
        if os.path.exists(f"{LLM_STREAM_RESPONSE_CACHE_FOLDER}/{hashed_request}/combined.json"):
            return
        if self.have_stream_response_cache(model,system,assistant,user):
            matching_files = glob.glob(f"{LLM_STREAM_RESPONSE_CACHE_FOLDER}/{hashed_request}/*.json")
            matching_files=sorted(matching_files, key=lambda x: int(os.path.basename(x).split(".")[0]))
            chat_caches=[]
            for path in matching_files:
                with open(path, "r",encoding="utf8") as chat_cache_file:
                    chat_cache = json.load(chat_cache_file)
                    chat_caches.append(chat_cache)
            if self.is_incomplete_stream_cache(chat_cache) is False:
                import shutil
                shutil.rmtree(f"{LLM_STREAM_RESPONSE_CACHE_FOLDER}/{hashed_request}")
                _LLM_Base.save_stream_response_cache(model,system,assistant,user,chat_caches,combined=True)

    
    def have_conversation_stream_cache(self,model,messages:list):
        hashed_request=calculate_md5(f"{model}{json.dumps(messages)}")
        return os.path.exists(f"{LLM_STREAM_RESPONSE_CACHE_FOLDER}/{hashed_request}")
    def load_conversation_stream_cache(self,model,messages:list):
        hashed_request=calculate_md5(f"{model}{json.dumps(messages)}")
        # print(f"Loading response cache for {model} model with id: {hashed_request}...")
        if self.have_conversation_stream_cache(model,messages):
            matching_files = glob.glob(f"{LLM_CONVERSATION_STREAM_CACHE_FOLDER}/{hashed_request}/*.json")
            matching_files=sorted(matching_files, key=lambda x: int(os.path.basename(x).split(".")[0]))
            for path in matching_files:
                with open(path, "r",encoding="utf8") as chat_cache_file:
                    chat_cache = json.load(chat_cache_file)
                    yield chat_cache
            if self.is_incomplete_stream_cache(chat_cache):
                chat_cache=COMPLETED_STREAM_CACHE
                yield chat_cache
                import shutil
                shutil.rmtree(f"{LLM_CONVERSATION_STREAM_CACHE_FOLDER}/{hashed_request}")
    def have_conversation_cache(self,model,messages:list):
        hashed_request=calculate_md5(f"{model}{json.dumps(messages)}")
        return os.path.exists(f"{LLM_CONVERSATION_CACHE_FOLDER}/{hashed_request}.json")
    def load_conversation_cache(self,model,messages:list):
        hashed_request=calculate_md5(f"{model}{json.dumps(messages)}")
        # print(f"Loading response cache for {model} model with id: {hashed_request}...")
        matching_files = glob.glob(f"{LLM_CONVERSATION_CACHE_FOLDER}/{hashed_request}/*.json")
        if len(matching_files)>0:
            with open(matching_files[-1], "r",encoding="utf8") as chat_cache_file:
                chat_cache = json.load(chat_cache_file)
                return chat_cache
        return None
    pass


class LLM_Base(_LLM_Base):
    embedding_size=None
    def __init__(self,instant:_LLM_Base) -> None:
        self.instant=instant
        pass
    pass
class LLM:
    
    def __init__(self,ModelClass:Type[LLM_Base],use_cache:bool=True,
                 on_each_response:Callable[[str,str,str,str,str], str]=None,
                 on_chunked:Callable[[str,str,str,str,str], str]=None) -> None:
        """
        Constructor for LLM class
        :param ModelClass: LLM_Base The class of the model to be used
        :param use_cache: bool Whether to use cache or not
        :param on_each_response: Callable[[str,str,str,str,str], str] A function to be called on each response
            - This function should take 5 string from get_response or in between on_tokens_oversized: 
                system,assistant,user,responses,response, and the return value should be a string
            - for get_response, the responses is None and the response is the current response.
            - for on_tokens_oversized, the responses is the concatenated string of previous responses and the response is the current response.
                because the input is too large, on tokens oversized is called recursively, 
                it has become complicated to get back the concatenated responses.
            - for system,assistant,user, it will send the string needed to generate current response
        """
        self.model_class=ModelClass(self)
        self.use_cache=use_cache
        self.on_each_response=on_each_response
        self.on_chunked=on_chunked

        self.model_class.use_cache=self.use_cache
        self.model_class.on_each_response=self.on_each_response
        self.model_class.on_chunked=self.on_chunked
        pass
    def get_model_name(self):
        return self.model_class.get_model_name()
    def set_model_name(self,name):
        return self.model_class.set_model_name(name)
    def get_response(self,system,assistant,user)->str:
        if self.on_each_response is None:
            return self.model_class.get_response(system,assistant,user)
        else:
            return self.on_each_response(system,assistant,user,None,self.model_class.get_response(system,assistant,user))
    def get_json_response(self,system,assistant,user)->dict:
        response=self.get_response(system,user,assistant)
        try:
            return json.loads(response)
        except Exception as e:
            print(e)
            print(response)
            return self.repair_json(response)
    def repair_json(self,json_data, schema=None,error=None)->dict:
        system="""Fix the following JSON string to make it valid.
I don't need any extra description in the JSON only give me the JSON.
"""
        if error is not None:
            system+="""Here is the error message during JSON validation:"""+str(error)
        if schema is not None:
            system+="""You can fix the JSON according to the following schema.  Here is the schema:
"""+json.dumps(schema,ensure_ascii=False)
        assistant="""{"""
        user=json_data
        response=self.get_response(system,assistant,user)
        try:
            return json.loads(response)
        except Exception as e:
            print(e)
            print(response)
            return self.repair_json(response, schema,e)
    
    def get_response_stream(self,system,assistant,user):
        response=self.model_class.get_response_stream(system,assistant,user)
        return response
    def combine_stream_response_cache(self,system,assistant,user):
        return self.model_class.combine_stream_response_cache(system,assistant,user)
    def get_conversation_stream(self,messages):
        return self.model_class.get_conversation_stream(messages)
    def on_tokens_oversized(self,e,system,assistant,user):
        return self.model_class.on_tokens_oversized(e,system,assistant,user)
    def set_event_listener(self,event_name:str,func:Callable[[Any], Any]):
        self.model_class.set_event_listener(event_name,func)
    def get_conversation_response(self,messages):
        return self.model_class.get_conversation_response(messages)
    def get_embeddings(self,sentences:str|list[str])->np.ndarray:
        return self.model_class.get_embeddings(sentences)
    def get_functions_response(self,messages:str|list[str],functions:list[dict]):
        return self.model_class.get_functions_response(messages,functions)
    pass