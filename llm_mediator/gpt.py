from typing import Any, Generator
from .llm import LLM_Base,ON_TOKENS_OVERSIZED,CallStack
import openai
from time import sleep
import time
import os
import re
from .parallel_tasks_queuer import build_and_execute_tasks
import numpy as np

GPT3_MODEL = "gpt-3.5-turbo"
GPT4_MODEL = "gpt-4"
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')
TECKY_API_KEY = os.environ.get('TECKY_API_KEY')
EMBEDDING_MODEL="text-embedding-ada-002"
EMBEDDING_SIZE=1536

ON_RESULT_FILTERED="on_result_filtered"

def detect_if_result_filtered(e):
    return re.search(r"The response was filtered due to the prompt triggering Azure OpenAI’s content management policy.", str(e)) is not None

class GPT(LLM_Base):
    gpt_error_delay=2
    temperature=0

    def switch2tecky():
        openai.api_key = TECKY_API_KEY
        openai.api_base = "https://api.gpt.tecky.ai/v1"
    def switch2openai():
        openai.api_key = OPENAI_API_KEY
        openai.api_base = "https://api.openai.com/v1"

    def model_picker():
        if TECKY_API_KEY is not None and TECKY_API_KEY != "":
            return GPT4_MODEL
        elif OPENAI_API_KEY is not None and OPENAI_API_KEY != "":
            return GPT3_MODEL
        else:
            return None
    def get_embeddings(self,sentences:str|list[str]):
        origin_sentences_type=type(sentences)
        if origin_sentences_type.__name__=="str":
            return openai.Embedding.create(input = [sentences], model=EMBEDDING_MODEL)['data'][0]['embedding']
        embeddings=[]
        def split_list(input_list, n):
            return [input_list[i:i + n] for i in range(0, len(input_list), n)]
        def get_embeddings_parallel(sentences):
            for embedding in openai.Embedding.create(input = sentences, model=EMBEDDING_MODEL)['data']:
                embeddings.append(embedding['embedding'])
        sentences_chunks=split_list(sentences,16)
        params=[]
        for sentences_chunk in sentences_chunks:
            params.append([sentences_chunk])
        build_and_execute_tasks(get_embeddings_parallel,params)
        return np.array(embeddings)

    
    def get_model_name(self):
        if self.model_name is None:
            self.model_name = GPT.model_picker()
        return self.model_name
    def set_model_name(self, name):
        self.model_name = name
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
        print(f"Connecting to {model} model...")
        try:
            completion = openai.ChatCompletion.create(
                model=model,
                messages=[
                        {"role": "system","content": system},
                        {"role": "user","content": user},
                        {"role": "assistant","content": assistant}
                    ],
                temperature=self.temperature,
                # max_tokens=2048
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
            response = openai.ChatCompletion.create(
                model=model,
                messages=[
                        {"role": "system","content": system},
                        {"role": "user","content": user},
                        {"role": "assistant","content": assistant}
                    ],
                temperature=0,
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
            response = openai.ChatCompletion.create(
                model=model,
                messages=messages,
                temperature=0,
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
        print(f"Connecting to {model} model...")
        try:
            completion = openai.ChatCompletion.create(
                model=model,
                messages=messages,
                temperature=self.temperature,
                # max_tokens=2048
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
            else:
                print(f"Retrying in {self.gpt_error_delay} seconds...")
                sleep(self.gpt_error_delay)
                self.gpt_error_delay=self.gpt_error_delay*2
                return self.instant.get_conversation_response(messages)
        
    pass