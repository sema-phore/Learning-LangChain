"""
Predefined chunck size - either by charecters or by tokens
"""

from langchain_text_splitters import CharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader, PyPDFLoader

# url = "https://docs.langchain.com/oss/python/langchain/overview"
# web_loader = WebBaseLoader(url)

pdf_loader = PyPDFLoader('dl-curriculum.pdf')

splitter = CharacterTextSplitter(
    chunk_size = 300,
    chunk_overlap = 0, # Helps to retain the context by overlapping splits, 10-20% is good for RAG based application
    separator=''
)

text = """ Over the past two decades Machine Learning has become one of the mainstays of information technology and with that, a rather central, albeit usually
hidden, part of our life. With the ever increasing amounts of data becoming
available there is good reason to believe that smart data analysis will become
even more pervasive as a necessary ingredient for technological progress.
The purpose of this chapter is to provide the reader with an overview over
the vast range of applications which have at their heart a machine learning
problem and to bring some degree of order to the zoo of problems. After
that, we will discuss some basic tools from statistics and probability theory,
since they form the language in which many machine learning problems must
be phrased to become amenable to solving. Finally, we will outline a set of
fairly basic yet effective algorithms to solve an important problem, namely
that of classification. More sophisticated tools, a discussion of more general
problems and a detailed analysis will follow in later parts of the book.

"""

docs = pdf_loader.load()

result = splitter.split_documents(docs)

print(result[0].page_content)