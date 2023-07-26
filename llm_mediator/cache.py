from .llm import LLM_Base,ON_TOKENS_OVERSIZED,CallStack

class Cache(LLM_Base):
    def get_model_name(self):
        return self.model_name
    def set_model_name(self,name):
        self.model_name=name
    def detect_if_tokens_oversized(self,e):
        return False
    def get_response(self,system,assistant,user):
        model=self.get_model_name()
        if model is None:
            raise Exception("Model name is None")
        response_cache=LLM_Base.load_response_cache(model,system,assistant,user)
        if response_cache is not None:
            if "choices" in response_cache and len(response_cache["choices"])>0 and "message" in response_cache["choices"][0] and \
                "content" in response_cache["choices"][0]["message"]:
                response_content=response_cache["choices"][0]["message"]["content"]
                return response_content
            elif ON_TOKENS_OVERSIZED in response_cache:
                e=response_cache[ON_TOKENS_OVERSIZED]
                return self.instant.on_tokens_oversized(e,system,assistant,user)
            else:
                if (len(response_cache["choices"])==0 or
                    "message" not in response_cache["choices"][0] or
                    "content" not in response_cache["choices"][0]["message"]):
                    return None
        return None
    pass