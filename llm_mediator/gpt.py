from typing import Any, Generator
from .llm import LLM_Base,ON_TOKENS_OVERSIZED,CallStack,calculate_md5
import openai
from openai.types.chat.chat_completion_chunk import ChatCompletionChunk,Choice
from openai.types.chat.chat_completion import ChatCompletion
from time import sleep
import time
import os
import re
from .parallel_tasks_queuer import build_and_execute_tasks
import numpy as np
import json
from abc import ABC, abstractmethod

GPT3_MODEL = "gpt-3.5-turbo"
GPT4_MODEL = "gpt-4"
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')
TECKY_API_KEY = os.environ.get('TECKY_API_KEY')
EMBEDDING_MODEL="text-embedding-ada-002"
EMBEDDING_SIZE=1536
DEFAULT_SIMPLE_TEXT_GENERATOR_EXTRACTING_PATH=["choices",0,"delta","content"]
DEFAULT_FUNCTIONAL_JSON_TEXT_GENERATOR_EXTRACTING_PATH=["choices",0,"delta","function_call","arguments"]

ON_RESULT_FILTERED="on_result_filtered"

def detect_if_result_filtered(e):
    return re.search(r"The response was filtered due to the prompt triggering Azure OpenAI’s content management policy.", str(e)) is not None
class GPT(LLM_Base):
    SIMPLE_TEXT_GENERATOR_EXTRACTING_PATH=DEFAULT_SIMPLE_TEXT_GENERATOR_EXTRACTING_PATH
    FUNCTIONAL_JSON_TEXT_GENERATOR_EXTRACTING_PATH=DEFAULT_FUNCTIONAL_JSON_TEXT_GENERATOR_EXTRACTING_PATH
    embedding_size=EMBEDDING_SIZE
    gpt_error_delay=2
    temperature=0

    def switch2tecky(api_key=None):
        if api_key is not None:
            openai.api_key = api_key
        else:
            openai.api_key = TECKY_API_KEY
        openai.base_url = "https://api.gpt.tecky.ai/v1/"
    def switch2openai(api_key=None):
        if api_key is not None:
            openai.api_key = api_key
        else:
            openai.api_key = OPENAI_API_KEY
        openai.base_url = "https://api.openai.com/v1/"

    def get_4k_model_instant():
        from .llm import LLM
        model=LLM(GPT)
        model.set_model_name("gpt-3.5-turbo")
        return model
    def get_8k_model_instant():
        from .llm import LLM
        model=LLM(GPT)
        model.set_model_name("gpt-4")
        return model
    def get_16k_model_instant():
        from .llm import LLM
        model=LLM(GPT)
        model.set_model_name("gpt-3.5-turbo-16k")
        return model
    def get_32k_model_instant():
        from .llm import LLM
        model=LLM(GPT)
        model.set_model_name("gpt-4-32k")
        return model

    def model_picker():
        if TECKY_API_KEY is not None and TECKY_API_KEY != "":
            return GPT4_MODEL
        elif OPENAI_API_KEY is not None and OPENAI_API_KEY != "":
            return GPT3_MODEL
        else:
            return None

    class BaseGeneratorExtractor(ABC):
        @abstractmethod
        def get_extraction_path(self,chunk):
            pass
        def get_choice0(self,chunk:ChatCompletionChunk|dict):
            if hasattr(chunk,"choices") and len(chunk.choices)>0:
                return chunk.choices[0]
            elif "choices" in chunk and len(chunk["choices"])>0:
                return chunk["choices"][0]
        def extract_text_from_generator_chunk(self,chunk:ChatCompletionChunk|dict):
            node=self.get_choice0(chunk)
            extraction_path=self.get_extraction_path(chunk)
            if extraction_path is None:
                return ""
            if isinstance(chunk,dict):
                return self.extract_text_from_dict_chunk(chunk,extraction_path)
            for key in extraction_path:
                if hasattr(node,key) and getattr(node,key) is not None:
                    node = getattr(node,key)
                else:
                    return ""
            return node
        def append_text_into_generator_chunk(self,chunk:ChatCompletionChunk,text):
            node=self.get_choice0(chunk)
            extraction_path=self.get_extraction_path(chunk)
            if extraction_path is None:
                return chunk
            if isinstance(chunk,dict):
                return self.append_text_into_dict_chunk(chunk,text,extraction_path)
            for key in extraction_path:
                if hasattr(node,key) and getattr(node,key) is not None:
                    parent_node=node
                    node = getattr(node,key)
            if isinstance(node,str) and isinstance(text,str):
                setattr(parent_node,extraction_path[-1],getattr(parent_node,extraction_path[-1])+text)
            return chunk
        
        def extract_text_from_dict_chunk(self,chunk,generator_extracting_path):
            node=self.get_choice0(chunk)
            for key in generator_extracting_path:
                try:
                    node = node[key]
                except KeyError:
                    return ""
            return node
        def append_text_into_dict_chunk(self,chunk,text,generator_extracting_path):
            node=self.get_choice0(chunk)
            for key in generator_extracting_path:
                try:
                    parent_node=node
                    node = node[key]
                except KeyError:
                    return chunk
            if isinstance(node,str) and isinstance(text,str):
                parent_node[generator_extracting_path[-1]]+=text
            return chunk
        pass
    class AutoGeneratorExtractor(BaseGeneratorExtractor):
        @staticmethod
        def get_extraction_path(chunk:ChatCompletionChunk|dict):
            text_path=["delta","content"]
            functional_path=["delta","function_call","arguments"]
            if isinstance(chunk,dict):
                if "choices" in chunk and len(chunk["choices"])>0:
                    choice=chunk["choices"][0]
                    if "delta" in choice and "content" in choice["delta"] and choice["delta"]["content"] is not None:
                        return text_path
                    elif "delta" in choice and "function_call" in choice["delta"]:
                        function_call=choice["delta"]["function_call"]
                        if function_call is not None:
                            if "arguments" in choice["delta"]["function_call"]:
                                return functional_path
            if isinstance(chunk,ChatCompletionChunk):
                if hasattr(chunk,"choices") and len(chunk.choices)>0:
                    choice:Choice=chunk.choices[0]
                    if hasattr(choice,"delta") and hasattr(choice.delta,"content") and choice.delta.content is not None:
                        return text_path
                    elif hasattr(choice,"delta") and hasattr(choice.delta,"function_call") and hasattr(choice.delta.function_call,"arguments") and choice.delta.function_call.arguments is not None:
                        return functional_path
            return None
        pass
    
    def extract_text_from_chat_completion_chunk(chunk,completion_extractor:BaseGeneratorExtractor=AutoGeneratorExtractor,**_):
        return completion_extractor().extract_text_from_generator_chunk(chunk)
    def append_text_into_chat_completion_chunk(chunk,text,completion_extractor:BaseGeneratorExtractor=AutoGeneratorExtractor,**_):
        return completion_extractor().append_text_into_generator_chunk(chunk,text)
    
    def get_embeddings(self,sentences:str|list[str]):
        origin_sentences_type=type(sentences)
        if origin_sentences_type.__name__=="str":
            return openai.embeddings.create(input = [sentences], model=EMBEDDING_MODEL)['data'][0]['embedding']
        embeddings=[]
        def split_list(input_list, n):
            return [input_list[i:i + n] for i in range(0, len(input_list), n) if all(item is not None for item in input_list[i:i + n])]
        def get_embeddings_parallel(sentences):
            for embedding in openai.embeddings.create(input = sentences, model=EMBEDDING_MODEL)['data']:
                embeddings.append(embedding['embedding'])
        sentences_chunks=split_list(sentences,16)
        params=[]
        for sentences_chunk in sentences_chunks:
            params.append([sentences_chunk])
        build_and_execute_tasks(get_embeddings_parallel,params)
        return embeddings

    
    def get_model_name(self):
        if self.model_name is None:
            self.model_name = GPT.model_picker()
        return self.model_name
    def set_model_name(self, name):
        self.model_name = name
    def get_chat_completion_from_openai(self,*args,**kwargs):
        model=self.get_model_name()
        openai_kwargs=kwargs.copy()
        openai_kwargs.pop("print_chunk",None)
        openai_kwargs.pop("completion_extractor",None)
        response:ChatCompletion = openai.chat.completions.create(*args,model=model,**openai_kwargs)
        hashed_request=self.get_request_hash(model,*args,**kwargs)
        self.save_chat_completion_cache(model,hashed_request,response.model_dump())
        return response
    def get_chat_completion_from_openai_stream(self,*args,**kwargs):
        stream=kwargs["stream"]
        model=self.get_model_name()
        openai_kwargs=kwargs.copy()
        openai_kwargs.pop("print_chunk",None)
        openai_kwargs.pop("completion_extractor",None)
        response:ChatCompletionChunk|ChatCompletion = openai.chat.completions.create(*args,model=model,**openai_kwargs)
        if not stream:
            return response
        chunks:list[ChatCompletionChunk]=[]
        for chunk in response:
            yield chunk
            chunks.append(chunk)
            if "print_chunk" in kwargs and kwargs["print_chunk"] is True:
                print(GPT.extract_text_from_chat_completion_chunk(chunk,**kwargs),end="")
                pass
            pass
        first_chunk=None
        last_chunk=None
        for chunk in chunks:
            text=GPT.extract_text_from_chat_completion_chunk(chunk,**kwargs)
            if first_chunk is None and text is not None and text != "":
                first_chunk=chunk
            elif first_chunk is not None and text is not None and text != "":
                first_chunk=GPT.append_text_into_chat_completion_chunk(first_chunk,text,**kwargs)
            last_chunk=chunk
            pass
        if first_chunk is not None and last_chunk is not None:
            chunks=[first_chunk,last_chunk]
        hashed_request=self.get_request_hash(model,*args,**kwargs)
        output_chunks=[json.loads(chunk.model_dump_json()) for chunk in chunks]
        self.save_chat_completion_cache_stream(model,hashed_request,output_chunks)
    def get_chat_completion_from_cache_stream(self,hashed_request=None,*args,**kwargs):
        model=self.get_model_name()
        if hashed_request is None:
            hashed_request=self.get_request_hash(model,*args,**kwargs)
        if self.have_chat_completion_cache_stream(hashed_request):
            cache=self.load_chat_completion_cache_stream(hashed_request)
            return cache
        else:
            return None
    def get_chat_completion_from_cache(self,hashed_request=None,*args,**kwargs):
        model=self.get_model_name()
        if hashed_request is None:
            hashed_request=self.get_request_hash(model,*args,**kwargs)
        if self.have_chat_completion_cache(hashed_request):
            cache=self.load_chat_completion_cache(hashed_request)
            return cache
        else:
            return None
    def get_chat_completion(self,stream=False,completion_extractor=AutoGeneratorExtractor,*args,**kwargs):
        kwargs["completion_extractor"]=completion_extractor
        kwargs["stream"]=stream
        model=self.get_model_name()
        if model is None:
            raise Exception("No API key found for OpenAI or Tecky")
        generator=None
        hashed_request=self.get_request_hash(model,*args,**kwargs)
        if stream:
            if self.have_chat_completion_cache_stream(hashed_request,*args,**kwargs) and self.use_cache:
                generator = self.get_chat_completion_from_cache_stream(hashed_request,*args,**kwargs)
            else:
                generator = self.get_chat_completion_from_openai_stream(*args,**kwargs)
            return generator
        else:
            if self.have_chat_completion_cache(hashed_request,*args,**kwargs) and self.use_cache:
                response = self.get_chat_completion_from_cache(hashed_request,*args,**kwargs)
            else:
                response = self.get_chat_completion_from_openai(*args,**kwargs)
            return response
    def detect_if_tokens_oversized(self,e):
        return (re.search(r"This model's maximum context length is", str(e)) is not None and \
            re.search(r"tokens", str(e)) is not None and \
            re.search(r"Please reduce the length of the messages.", str(e)) is not None) or \
            (re.search(r"HTTP code 413 from API", str(e)) is not None and \
                re.search(r"PayloadTooLargeError: request entity too large", str(e)) is not None)
    def get_response(self,system,assistant,user):
        model=self.get_model_name()
        if model is None:
            raise Exception("No API key found for OpenAI or Tecky")
        response_cache=LLM_Base.load_response_cache(model,system,assistant,user)
        if response_cache is not None and self.use_cache:
            if "choices" in response_cache and len(response_cache["choices"])>0 and "message" in response_cache["choices"][0] and \
                "content" in response_cache["choices"][0]["message"]:
                response_content=response_cache["choices"][0]["message"]["content"]
                return response_content
            elif ON_TOKENS_OVERSIZED in response_cache:
                e=response_cache[ON_TOKENS_OVERSIZED]
                return self.instant.on_tokens_oversized(e,system,assistant,user)
            elif ON_RESULT_FILTERED in response_cache:
                return None
            else:
                if (len(response_cache["choices"])==0 or
                    "message" not in response_cache["choices"][0] or
                    "content" not in response_cache["choices"][0]["message"]):
                    LLM_Base.delete_response_cache(model,system,assistant,user)
        # print(f"Connecting to {model} model...")
        try:
            completion = openai.chat.completions.create(
                model=model,
                messages=[
                        {"role": "system","content": system},
                        {"role": "user","content": user},
                        {"role": "assistant","content": assistant}
                    ],
                temperature=self.temperature
                )
            LLM_Base.save_response_cache(model,system,assistant,user,completion)
            if len(completion.choices)==0:
                raise Exception("Invalid Output: No choices found in completion")
            elif "message" not in completion.choices[0]:
                raise Exception("Invalid Output: No message found in completion")
            elif "content" not in completion.choices[0].message:
                raise Exception("Invalid Output: No content found in completion")
            return completion.choices[0].message.content
        except Exception as e:
            print(e)
            if self.detect_if_tokens_oversized(e):
                LLM_Base.save_response_cache(model,system,assistant,user,{ON_TOKENS_OVERSIZED:str(e)})
                return self.instant.on_tokens_oversized(e,system,assistant,user)
            elif detect_if_result_filtered(e):
                LLM_Base.save_response_cache(model,system,assistant,user,{ON_RESULT_FILTERED:str(e)})
                return None
            elif re.search(r"Invalid Output: ", str(e)) is not None:
                return None
            else:
                print(f"Retrying in {self.gpt_error_delay} seconds...")
                sleep(self.gpt_error_delay)
                self.gpt_error_delay=self.gpt_error_delay*2
                return self.instant.get_response(system,assistant,user)
    def get_response_stream_from_openai(self, system, assistant, user):
        model=self.get_model_name()
        try:
            response = openai.chat.completions.create(
                model=model,
                messages=[
                        {"role": "system","content": system},
                        {"role": "user","content": user},
                        {"role": "assistant","content": assistant}
                    ],
                temperature=self.temperature,
                stream=True
            )
            for chunk in response:
                yield chunk
                LLM_Base.save_stream_response_cache(model,system,assistant,user,chunk)
        except Exception as e:
            print(e)
            pass
    def get_response_stream(self, system, assistant, user):
        model=self.get_model_name()
        """
        Get a streaming response from the GPT model
        :param system: The system message
        :param assistant: The assistant message
        :param user: The user message
        :return: The response
        """
        if model is None:
            raise Exception("No API key found for OpenAI or Tecky")
        if self.have_stream_response_cache(model,system,assistant,user):
            response_cache=self.load_stream_response_cache(model,system,assistant,user)
            return response_cache
        else:
            response=self.get_response_stream_from_openai(system,assistant,user)
            return response
    
    def get_conversation_stream_from_openai(self, messages):
        model=self.get_model_name()
        try:
            response = openai.chat.completions.create(
                model=model,
                messages=messages,
                temperature=self.temperature,
                stream=True
            )
            for chunk in response:
                yield chunk
                LLM_Base.save_conversation_stream_cache(model,messages,chunk)
        except Exception as e:
            print(e)
            pass
    def get_conversation_stream(self, messages) -> Generator[Any, Any, None]:
        model=self.get_model_name()
        """
        Get a streaming conversation from the GPT model
        :param messages: The messages for all system, assistant and user
        :return: The response
        """
        if model is None:
            raise Exception("No API key found for OpenAI or Tecky")
        if self.have_conversation_stream_cache(model,messages):
            response_cache=self.load_conversation_stream_cache(model,messages)
            return response_cache
        else:
            response=self.get_conversation_stream_from_openai(messages)
            return response
    
    def get_conversation_response(self,messages) -> str:
        model=self.get_model_name()
        if model is None:
            raise Exception("No API key found for OpenAI or Tecky")
        response_cache=self.load_conversation_cache(model,messages)
        if response_cache is not None and self.use_cache:
            if "choices" in response_cache and len(response_cache["choices"])>0 and "message" in response_cache["choices"][0] and \
                "content" in response_cache["choices"][0]["message"]:
                response_content=response_cache["choices"][0]["message"]["content"]
                return response_content
        # print(f"Connecting to {model} model...")
        try:
            completion = openai.chat.completions.create(
                model=model,
                messages=messages,
                temperature=self.temperature
                )
            LLM_Base.save_conversation_cache(model,messages,completion)
            if len(completion.choices)==0:
                raise Exception("Invalid Output: No choices found in completion")
            elif "message" not in completion.choices[0]:
                raise Exception("Invalid Output: No message found in completion")
            elif "content" not in completion.choices[0].message:
                raise Exception("Invalid Output: No content found in completion")
            return completion.choices[0].message.content
        except Exception as e:
            print(e)
            if detect_if_result_filtered(e):
                LLM_Base.save_conversation_cache(model,messages,{ON_RESULT_FILTERED:str(e)})
                return None
            elif re.search(r"Invalid Output: ", str(e)) is not None:
                return None
            elif "Insufficient balance" in str(e):
                raise Exception("Insufficient balance")
            else:
                print(f"Retrying in {self.gpt_error_delay} seconds...")
                sleep(self.gpt_error_delay)
                self.gpt_error_delay=self.gpt_error_delay*2
                return self.instant.get_conversation_response(messages)

    def get_functions_response(self,messages:str|list[str],functions:list[dict]):
        model=self.get_model_name()
        try:
            response = openai.chat.completions.create(
                model=model,
                messages=messages,
                functions=functions,
                temperature=self.temperature
            )
            return response
        except Exception as e:
            print(e)
            pass
        return
    
    def combine_stream_response_cache(self,system,assistant,user):
        from .llm import LLM_STREAM_RESPONSE_CACHE_FOLDER,calculate_md5
        import glob
        import json

        model=self.get_model_name()
        hashed_request=calculate_md5(f"{model}{system}{assistant}{user}")
        if os.path.exists(f"{LLM_STREAM_RESPONSE_CACHE_FOLDER}/{hashed_request}/combined.json"):
            return
        if self.have_stream_response_cache(model,system,assistant,user):
            matching_files = glob.glob(f"{LLM_STREAM_RESPONSE_CACHE_FOLDER}/{hashed_request}/*.json")
            matching_files=sorted(matching_files, key=lambda x: int(os.path.basename(x).split(".")[0]))
            first_chunk=None
            for path in matching_files:
                with open(path, "r",encoding="utf8") as chat_cache_file:
                    chat_cache = json.load(chat_cache_file)
                    text=GPT.extract_text_from_generator_chunk(chat_cache)
                    if first_chunk is None and text is not None:
                        first_chunk=chat_cache
                    first_chunk=GPT.append_text_into_generator_chunk(first_chunk,text)
            chat_caches=[first_chunk,chat_cache]
            if self.is_incomplete_stream_cache(chat_cache) is False:
                import shutil
                shutil.rmtree(f"{LLM_STREAM_RESPONSE_CACHE_FOLDER}/{hashed_request}")
                LLM_Base.save_stream_response_cache(model,system,assistant,user,chat_caches,combined=True)
    pass