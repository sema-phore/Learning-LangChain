from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="google/gemma-2-9b-it",
    task="text-generation"
)
model = ChatHuggingFace(llm=llm)

prompt1 = PromptTemplate(
    template='Generate a detailed report on {topic}',
    input_variables=['topic']
)

prompt2 = PromptTemplate(
    template='Generate a 5 pointer summary from the following text \n {text}',
    input_variables=['text']
)

parser = StrOutputParser()

seq_chain = prompt1 | model | parser | prompt2 | model | parser

result = seq_chain.invoke({'topic': 'Lang chain'})

print(result)

seq_chain.get_graph().print_ascii()