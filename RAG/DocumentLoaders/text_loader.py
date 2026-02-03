from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader

load_dotenv()

# Model
llm = HuggingFaceEndpoint(
    repo_id="google/gemma-2-9b-it",
    task="text-genertion"
)
model = ChatHuggingFace(llm=llm)

prompt = PromptTemplate(
    template='Write a summary for the following poem - \n {poem}',
    input_variables=['poem']
)

parser = StrOutputParser()

# loader with txt file path
loader = TextLoader('cricket.txt', encoding='utf-8')
# Load as a document
docs = loader.load()

docs_content = docs[0].page_content
docs_metadata = docs[0].metadata

chain = prompt | model | parser

print(chain.invoke({'poem': docs_content}))