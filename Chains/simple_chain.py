"""
It's a way to create "PIPELINES".
Output of previous step, automatically become Input for next step.
~ Sequential Processing, Parallel Processing, Conditional Procession
"""
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="google/gemma-2-9b-it",
    task="text-generation"
)
model = ChatHuggingFace(llm=llm)

prompt = PromptTemplate(
    template="Generate 5 interestin facts about {topic}",
    input_variables=['topic']
)

parser = StrOutputParser()

chain = prompt | model | parser
result = chain.invoke({'topic':'Cricket'})
print(result)

chain.get_graph().print_ascii()