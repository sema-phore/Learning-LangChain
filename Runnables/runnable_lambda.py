"""

"""

from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnableSequence, RunnablePassthrough, RunnableLambda
from dotenv import load_dotenv

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="google/gemma-2-9b-it",
    task="text-genertion"
)
model = ChatHuggingFace(llm=llm)

prompt = PromptTemplate(
    template='Write a joke about {topic}',
    input_variables=['topic']
)

parser = StrOutputParser()

joke_gen_chain = RunnableSequence(prompt, model, parser)

def word_count(text):
    return len(text.split())

parallel_chain = RunnableParallel(
    {
        'joke':RunnablePassthrough(),
        'word_count': RunnableLambda(word_count)
    }
)

final_chain = RunnableSequence(joke_gen_chain, parallel_chain)

result = final_chain.invoke({'topic': 'Cricket'})

final_result = """{} \n word count - {}""".format(result['joke'], result['word_count'])

print(final_result)