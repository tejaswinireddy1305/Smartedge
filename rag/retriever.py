import os
import glob
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DOCUMENT_DIR = os.path.join(BASE_DIR, "rag", "documents")


class SmartEdgeRetriever:

    def __init__(self):
        self.documents = []
        self.embeddings = None
        self.index = None
        self.use_sentence_transformer = False

        # Try to use Sentence Transformer, fall back to TF-IDF
        try:
            from sentence_transformers import SentenceTransformer
            import faiss
            self.model = SentenceTransformer("all-MiniLM-L6-v2")
            self.use_sentence_transformer = True
        except Exception as e:
            print(f"Warning: Could not load Sentence Transformer ({e}). Using TF-IDF fallback.")
            self.use_sentence_transformer = False
            self.vectorizer = TfidfVectorizer(stop_words='english')

        self.load_documents()
        self.build_index()

    def load_documents(self):

        files = glob.glob(
            os.path.join(DOCUMENT_DIR, "*.txt")
        )

        for file_path in files:

            with open(
                file_path,
                "r",
                encoding="utf-8"
            ) as file:

                text = file.read()

            self.documents.append({
                "source": os.path.basename(file_path),
                "text": text
            })

    def build_index(self):

        if not self.documents:
            return

        texts = [
            document["text"]
            for document in self.documents
        ]

        if self.use_sentence_transformer:
            import faiss
            self.embeddings = self.model.encode(
                texts,
                convert_to_numpy=True
            )

            dimension = self.embeddings.shape[1]

            self.index = faiss.IndexFlatL2(
                dimension
            )

            self.index.add(
                self.embeddings.astype("float32")
            )
        else:
            # TF-IDF fallback
            self.embeddings = self.vectorizer.fit_transform(texts)

    def search(self, query, top_k=2):

        if self.use_sentence_transformer:
            if self.index is None:
                return []

            query_embedding = self.model.encode(
                [query],
                convert_to_numpy=True
            )

            distances, indices = self.index.search(
                query_embedding.astype("float32"),
                top_k
            )

            results = []

            for distance, index in zip(
                distances[0],
                indices[0]
            ):

                if index < len(self.documents):

                    results.append({
                        "source": self.documents[index]["source"],
                        "text": self.documents[index]["text"],
                        "distance": float(distance)
                    })

            return results
        else:
            # TF-IDF fallback
            if self.embeddings is None:
                return []

            query_vector = self.vectorizer.transform([query])
            similarities = cosine_similarity(query_vector, self.embeddings)[0]

            # Get top-k indices
            top_indices = np.argsort(similarities)[-top_k:][::-1]

            results = []

            for index in top_indices:
                if index < len(self.documents):
                    results.append({
                        "source": self.documents[index]["source"],
                        "text": self.documents[index]["text"],
                        "distance": float(1 - similarities[index])  # Convert similarity to distance
                    })

            return results


if __name__ == "__main__":

    retriever = SmartEdgeRetriever()

    results = retriever.search(
        "Why does SmartEdge perform better than LFU?"
    )

    for result in results:

        print("\nSOURCE:")
        print(result["source"])

        print("\nCONTENT:")
        print(result["text"])

        print("\nDISTANCE:")
        print(result["distance"])