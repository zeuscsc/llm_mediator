import os
def reset_root_folder(path:str):
    global ROOT_FOLDER,\
        LLM_CACHES_FOLDER,\
        LLM_RESPONSE_CACHE_FOLDER,\
        LLM_STREAM_RESPONSE_CACHE_FOLDER,\
        LLM_CONVERSATION_STREAM_CACHE_FOLDER,\
        LLM_CONVERSATION_CACHE_FOLDER,\
        LLM_CHAT_COMPLETION_FOLDER,\
        LLM_MODELS_FOLDER
    ROOT_FOLDER=path
    LLM_CACHES_FOLDER = os.path.join(ROOT_FOLDER,"llm_caches")
    LLM_RESPONSE_CACHE_FOLDER = os.path.join(LLM_CACHES_FOLDER,"llm_response_cache")
    LLM_STREAM_RESPONSE_CACHE_FOLDER = os.path.join(LLM_CACHES_FOLDER,"llm_stream_response_cache")
    LLM_CONVERSATION_STREAM_CACHE_FOLDER = os.path.join(LLM_CACHES_FOLDER,"llm_conversation_stream_cache")
    LLM_CONVERSATION_CACHE_FOLDER = os.path.join(LLM_CACHES_FOLDER,"llm_conversation_cache")
    LLM_CHAT_COMPLETION_FOLDER = os.path.join(LLM_CACHES_FOLDER,"llm_chat_completion")
    LLM_MODELS_FOLDER = os.path.join(ROOT_FOLDER,"llm_models")

reset_root_folder(os.getcwd())