import os
def reset_root_folder(path:str):
    global ROOT_FOLDER,\
        CACHES_FOLDER,\
        LLM_RESPONSE_CACHE_FOLDER,\
        LLM_STREAM_RESPONSE_CACHE_FOLDER,\
        LLM_CONVERSATION_STREAM_CACHE_FOLDER,\
        LLM_FOLDER
    ROOT_FOLDER=path
    CACHES_FOLDER = f"{ROOT_FOLDER}llm_caches"
    LLM_RESPONSE_CACHE_FOLDER = os.path.join(CACHES_FOLDER,"llm_response_cache")
    LLM_STREAM_RESPONSE_CACHE_FOLDER = os.path.join(CACHES_FOLDER,"llm_stream_response_cache")
    LLM_CONVERSATION_STREAM_CACHE_FOLDER = os.path.join(CACHES_FOLDER,"llm_conversation_stream_cache")
    LLM_FOLDER = f"{ROOT_FOLDER}llm"

reset_root_folder("./")