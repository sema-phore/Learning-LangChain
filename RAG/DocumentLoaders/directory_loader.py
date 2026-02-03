from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader

loader = DirectoryLoader(
    path='Books',
    glob= '*.pdf', # What kind of files you want to load from path
    loader_cls= PyPDFLoader # loader class for glob/ documents
)

docs = loader.load()
# docs = loader.lazy_load()
"""
lazy_load()

- return a 'generator' of document object
- used for large documents or lot's of file
- Instade of loading all the documents at a time in the memory, It loads documents one by one, perform operation, remove it then load another.(Loads on Demand)
"""

print(docs[1].page_content)
print(len(docs))