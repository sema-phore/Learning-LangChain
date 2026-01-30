"""
LLMs are old 
So don't use them to build agents or model...
"""

from langchain_openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# Declare your model
llm = OpenAI(model = 'gpt-3.5-turbo-instruct')

# Communicate with model using invoke
result = llm.invoke("What is semaphore stands for in Operating System")

print(result)