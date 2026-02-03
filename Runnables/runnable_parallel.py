"""
Allows multiple runnables to execute parallel.

Each runnable receives the same input, processes it independently and generate a dictionary of output.
"""

from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnableSequence
from dotenv import load_dotenv

load_dotenv()

llm1 = HuggingFaceEndpoint(
    repo_id="google/gemma-2-9b-it",
    task="text-genertion"
)
model1 = ChatHuggingFace(llm=llm1)

llm2 = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation"
)
model2 = ChatHuggingFace(llm = llm2)

prompt1 = PromptTemplate(
    template="Generate a tweet about topic {topic}",
    input_variables=['topic']
)

prompt2 = PromptTemplate(
    template="Generate a Linkedin post about {topic}",
    input_variables=['topic']
)

str_parser = StrOutputParser()

parallel_chain = RunnableParallel(
    {
        "Tweet": RunnableSequence(prompt1, model1, str_parser),
        "Linkedin": RunnableSequence(prompt2, model1, str_parser)
    }
)

result = parallel_chain.invoke({'topic': 'AI'})

print(result)

parallel_chain.get_graph().print_ascii()