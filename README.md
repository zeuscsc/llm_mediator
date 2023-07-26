# LLM Picker
Just a simple mediator for different LLM models

## Features
- [x] Cache
- [x] GPT-3.5
- [x] GPT-4
- [x] GPT-4-32k
- [ ] LLaMA
- [ ] Falcon

## Quick Usage
Install:
~~~shell
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