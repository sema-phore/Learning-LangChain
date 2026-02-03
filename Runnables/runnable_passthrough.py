"""
Simply return the input as output🤔

usefull:
    - Want to show intermediate outputs in sequential chain
"""

from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnableSequence, RunnablePassthrough
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

joke_gen_chain = RunnableSequence(prompt1, model, parser)

parallel_chain = RunnableParallel(
    {
        'Joke': RunnablePassthrough(),
        'Explanation': RunnableSequence(prompt2, model, parser)
    }
)

final_chain = RunnableSequence(joke_gen_chain, parallel_chain)
result = final_chain.invoke({'topic': 'Marriage'})
print(result)