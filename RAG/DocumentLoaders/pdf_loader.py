"""
Load content form PDF, convert each page into a document object

limitation - it uses PyPDF library - not great for scanned pdfs and complex layouts

PDFPlumberLoader - Tabular
UnstructuredPDFLoader - Scanned

https://docs.langchain.com/oss/python/integrations/document_loaders
"""

from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader('dl-curriculum.pdf')

docs = loader.load()

print(len(docs))
print(docs[0].page_content)
print(docs[0].metadata)