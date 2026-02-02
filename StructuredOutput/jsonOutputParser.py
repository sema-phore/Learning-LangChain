from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="google/gemma-2-9b-it",
    task="text-generation"
)
model = ChatHuggingFace(llm=llm)

parser = JsonOutputParser()

template = PromptTemplate(
    template="Give me the name, age and city of a frictional person \n {format_instruction}",
    input_variables=[],
    partial_variables={"format_instruction": parser.get_format_instructions()} # What kind of o/p we need form llm or model
)


chain = template | model | parser

final_result = chain.invoke({})

print(final_result)

"""
prompt = template.format()
result = model.invoke(prompt)
final_result = parser.parse(result.content)


sending blank dict inside chain.invoke for this error coz it need and input
- TypeError: RunnableSequence.invoke() missing 1 required positional argument: 'input'

Flaw - JsonOutputParser doesn't provide any Json Schema 🥲
"""