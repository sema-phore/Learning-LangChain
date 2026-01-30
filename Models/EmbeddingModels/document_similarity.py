from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity

load_dotenv()

model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

documents = [
    "Virat Kohli is an Indian cricketer known for his aggressive batting and leadership.",
    "MS Dhoni is a former Indian captain famous for his calm demeanor and finishing skills.",
    "Sachin Tendulkar, also known as the 'God of Cricket', holds many batting records.",
    "Rohit Sharma is known for his elegant batting and record-breaking double centuries.",
    "Jasprit Bumrah is an Indian fast bowler known for his unorthodox action and yorkers."
]

query = "Tell me about Jasprit Bumrah"

# Embeddings
document_embedding = model.embed_documents(documents)
query_embedding = model.embed_query(query)

# calculate similarity score between vectors using cosine_similarity

score = cosine_similarity([query_embedding], document_embedding)[0]

"""
score = [0.70334049 0.40582648 0.51636065 0.5736942  0.36735796]
enumerate(score) = [(0, 0.70334049), (1, 0.40582648) . . . ]
- here we are preserving the similarity with out document vector
sort this based on score value so we can extract out similarity vector index
"""
index, score = sorted(list(enumerate(score)), key=lambda x:x[1])[-1]

print(query)
print(documents[index])
print(f"Similarity score: {score}")