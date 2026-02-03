from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser, PydanticOutputParser
from pydantic import BaseModel, Field
from typing import Literal
from dotenv import load_dotenv
from langchain_core.runnables import RunnableBranch, RunnableLambda

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="google/gemma-2-9b-it",
    task="text-generation"
)
model= ChatHuggingFace(llm=llm)

# string parser
str_parser = StrOutputParser()

# Pydantic Parser
class Feedback(BaseModel):

    sentiment: Literal['positive', 'negative'] = Field(description="Give the sentiment of the feedback")

pydantic_parser = PydanticOutputParser(pydantic_object=Feedback)



prompt1 = PromptTemplate(
    template="Classify the sentiment of the following feedback text into positive or negative. \n {feedback} \n {format_instruction}",
    input_variables=['feedback'],
    partial_variables={'format_instruction': pydantic_parser.get_format_instructions()}
)

prompt2 = PromptTemplate(
    template='Write an appropriate response to this positive feedback \n {feedback}',
    input_variables=['feedback']
)

prompt3 = PromptTemplate(
    template='Write an appropriate response to this negative feedback \n {feedback}',
    input_variables=['feedback']
)


classifier_chain = prompt1 | model | pydantic_parser


branch_chain = RunnableBranch(
    (lambda x: x.sentiment == 'positive', prompt2 | model | str_parser),
    (lambda x:x .sentiment == 'negative', prompt3 | model | str_parser),
    RunnableLambda(lambda x: "Could not find sentiment")
)

chain = classifier_chain | branch_chain

result = chain.invoke({'feedback': 'This is a beautiful place'})
print(result)

chain.get_graph().print_ascii()