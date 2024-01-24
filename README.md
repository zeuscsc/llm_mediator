# LLM Mediator
Just a simple mediator for different LLM models
Will cache the response for the same input text during debug and save money for you.

## Features
- [x] Cache
- [x] GPT-3.5
- [x] GPT-3.5-16k
- [x] GPT-4
- [x] GPT-4-32k
- [ ] GPT-4-vision
- [ ] DeepSeek-Gradio-API (Chinese LLM Gradio API)
- [ ] DeepSeek (Chinese LLM)

## Quick Usage for
Install:
~~~shell
pip install LLM-Mediator
# Install llm_mediator from github
pip install git+https://github.com/zeuscsc/llm_mediator.git
~~~
Usage:
~~~python
model_name="GPT-4-32k"
model=LLM(GPT)
model.model_class.set_model_name(model_name)
response=model.get_response(system,assistant,user)
~~~
Where `system`, `assistant`, `user` are the input text, and `response` is the output text.
Or you can just follow the docs from OpenAi:
~~python
generator=model.get_chat_completion(messages=messages,functions=functions,function_call=function_call,stream=True,temperature=0,completion_extractor=GPT.AutoGeneratorExtractor,print_chunk=False)
~~~
## Set Environment Variables
Unix:
~~~shell Unix
export OPENAI_API_KEY=your openai key (Nessary for GPT)
export TECKY_API_KEY=your tecky key (Nessary for GPT)
~~~
Windows:
~~~shell Windows
$ENV:OPENAI_API_KEY="your openai key" (Nessary for GPT)
$ENV:TECKY_API_KEY="your tecky key" (Nessary for GPT)
~~~
Python:
Create a 
~~~python
from llm_mediator import gpt
gpt.OPENAI_API_KEY="your openai key" (Nessary for GPT)
gpt.TECKY_API_KEY = "your tecky key" (Nessary for GPT)
~~~