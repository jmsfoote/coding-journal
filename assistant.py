import os
import lancedb
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

# ✅ Load API key from .env
load_dotenv()
api_key = os.getenv("LANCEDB_API_KEY")

# ✅ Load model + connect to LanceDB
model = SentenceTransformer("intfloat/e5-base")

db = lancedb.connect(
    uri="db://learning-project-dwwvuw",
    api_key=api_key,
    region="us-east-1"
)

table = db.open_table("journal_embeddings")


# ✅ Assistant function
def search_journal(query: str, top_k: int = 3):
    query_vector = model.encode(["passage: " + query])[0]
    results = table.search(query_vector).limit(top_k).to_pandas()

    for i, row in results.iterrows():
        similarity = 1 - row["_distance"]
        print(f"\n--- Match {i+1} (relevance: {similarity:.3f}) ---")
        print(row["text"])


# ✅ Simple command-line loop
if __name__ == "__main__":
    print("🧠 Coding Journal Assistant Ready!")
    while True:
        user_query = input("\n🔎 What would you like to search for? (or type 'exit'): ")
        if user_query.lower() in ["exit", "quit"]:
            print("Goodbye! 👋")
            break
        search_journal(user_query)