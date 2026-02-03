"""
Sequential chain of runnables
Output of previous automatically become input for next stage
"""

from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence
from dotenv import load_dotenv

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="google/gemma-2-9b-it",
    task="text-genertion"
)
model = ChatHuggingFace(llm=llm)

prompt1 = PromptTemplate(
    template="Write a joke about {topic}",
    input_variables=['topic']
)

prompt2 = PromptTemplate(
    template="Explain the following joke {text}",
    input_variables=['text']
)

parser = StrOutputParser()

chain = RunnableSequence(prompt1, model, parser, prompt2, model, parser)

result = chain.invoke({'topic': 'Marriage'})
print(result)