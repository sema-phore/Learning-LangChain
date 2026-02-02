"""
It is a structured output parser, uses Pydantic Models to enforce schema validation when precession LLM responses

- Strict schema enforcement
- type safety
- easy validation
- seamless integration
"""

from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
from pydantic import BaseModel, Field

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="google/gemma-2-9b-it",
    task="text-generation"
)
model = ChatHuggingFace(llm=llm)

class Person(BaseModel):
    name: str = Field(description="Name of the person")
    age: int = Field(gt=18, description="Age of the person")
    city: str = Field(description=" name of the city where perosn belongs to")

parser = PydanticOutputParser(pydantic_object=Person)

template = PromptTemplate(
    template="Generate the name, age and city of a friction {place} person \n {format_instruction}",
    input_variables=['place'],
    partial_variables= {'format_instruction': parser.get_format_instructions()}
)

# prompt = template.invoke({'place':'India'})
# print(prompt)


chain = template | model | parser

result = chain.invoke({'place':'India'})
print(result)