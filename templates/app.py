from sentence_transformers import SentenceTransformer
import numpy as np
# import faiss  # for efficient similarity search (optional)

# Load a pre-trained sentence transformer model
model = SentenceTransformer('all-MiniLM-L6-v2')  # You can use other models as well

# Sample documents
documents = [
    "The Eiffel Tower is in Paris.",
    "The capital of France is Paris.",
    "I love the culture and food of Italy.",
    "Python is a programming language."
]

# Step 1: Encode the documents into embeddings
doc_embeddings = model.encode(documents)

# Example search query
query = "Where is the Eiffel Tower located?"

# Step 2: Encode the query
query_embedding = model.encode([query])

# Step 3: Compute similarities between the query and the documents
# Using cosine similarity:
from sklearn.metrics.pairwise import cosine_similarity

cos_similarities = cosine_similarity(query_embedding, doc_embeddings)

# Step 4: Rank the documents based on similarity to the query
# Get the index of the most similar document
top_n = 3  # number of top results to return
top_indices = np.argsort(cos_similarities[0])[::-1][:top_n]

print("Query:", query)
print("\nTop matching documents:")
for idx in top_indices:
    print(f"- {documents[idx]} (Similarity: {cos_similarities[0][idx]:.4f})")
