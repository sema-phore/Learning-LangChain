"""
Conditional chain
"""

from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnableSequence, RunnablePassthrough, RunnableLambda, RunnableBranch
from dotenv import load_dotenv

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="google/gemma-2-9b-it",
    task="text-genertion"
)
model = ChatHuggingFace(llm=llm)

prompt1 = PromptTemplate(
    template='Write a detailed report on {topic}',
    input_variables=['topic']
)

prompt2 = PromptTemplate(
    template='Summarize the following text \n {text}',
    input_variables=['text']
)

parser = StrOutputParser()

reprot_gen_chain = RunnableSequence(prompt1, model, parser)

branch_chain = RunnableBranch(
    (lambda x: len(x.split()) >300, prompt2 | model | parser),
    RunnablePassthrough()
)

final_chain = RunnableSequence(reprot_gen_chain, branch_chain)

result = final_chain.invoke({'topic': 'Love and Lust'})
print(result)