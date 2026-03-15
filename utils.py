import os

def get_openai_api_key():
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("Set OPENAI_API_KEY in your environment.")
    return key
def GEMINI_API_KEY():
    key=os.getenv("GEMINI_API_KEY")
    if not key:
        raise RuntimeError("Set gemini epi key in terminal")
    return key
def serper_api_key():
    key=os.getenv("SERPER_API_KEY")
    if not key:
        raise RuntimeError("Set SERPER epi key in terminal")
    return key
