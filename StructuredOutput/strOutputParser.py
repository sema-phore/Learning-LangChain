from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
load_dotenv()

# Declare and Load model
llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation"
)
model = ChatHuggingFace(llm = llm)

# prompt1 - detailed report
template1 = PromptTemplate(
    template= "Write a detailed report on {topic}",
    input_variables=['topic']
)

# prompt2 - summary
template2 = PromptTemplate(
    template="Write a 5 point summary on following text. \n {text}",
    input_variables=['text']
)

# Creating parser
parser = StrOutputParser()

# Creating pipeline
chain = template1 | model | parser | template2 | model | parser
result = chain.invoke({'topic':'black hole'})

print(result)


# parser help to make everythin in same chain
"""
prompt1 = template1.invoke({'topic':'black hole'})
temp_result = model.invoke(prompt1)

prompt2 = template2.invoke({'text': temp_result.content})
result = model.invoke(prompt2)

print(result.content)
"""

