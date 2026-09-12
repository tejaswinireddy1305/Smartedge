import os

from rag.retriever import SmartEdgeRetriever
import ollama


class CacheAnalystAgent:

    def __init__(self):

        print("Initializing SmartEdge Cache Analyst Agent...")

        self.retriever = SmartEdgeRetriever()

        self.model = os.getenv(
            "OLLAMA_MODEL",
            "qwen2.5:0.5b"
        )

        self.ollama_available = True

        # Test Ollama availability
        try:
            ollama.list()
        except Exception as e:
            print(f"Warning: Ollama not available ({e}). Running in fallback mode.")
            self.ollama_available = False

    def analyze(self, question):

        # --------------------------------
        # STEP 1: Retrieve relevant evidence
        # --------------------------------

        results = self.retriever.search(
            question,
            top_k=2
        )

        if not results:

            return {
                "question": question,
                "answer": "No relevant SmartEdge evidence was found.",
                "evidence": []
            }

        # --------------------------------
        # STEP 2: Build context
        # --------------------------------

        context_parts = []

        evidence = []

        for result in results:

            context_parts.append(
                f"Source: {result['source']}\n"
                f"{result['text']}"
            )

            evidence.append({
                "source": result["source"],
                "distance": result["distance"]
            })

        context = "\n\n".join(context_parts)

        # --------------------------------
        # STEP 3: Send evidence to LLM (if available)
        # --------------------------------

        if not self.ollama_available:
            # Fallback: return retrieved evidence directly
            return {
                "question": question,
                "answer": f"AI service unavailable. Based on retrieved evidence:\n\n{context}",
                "evidence": evidence,
                "fallback": True
            }

        try:
            prompt = f"""
You are the SmartEdge Cache Analyst.

SmartEdge is an ML-driven predictive edge caching
system.

Answer the user's question using ONLY the evidence
provided below.

Do not invent benchmark numbers.

If the evidence does not contain enough information,
clearly say that the available evidence is insufficient.

Explain the answer clearly and technically but in
simple language.

USER QUESTION:
{question}

RETRIEVED EVIDENCE:
{context}

Provide a concise answer.
"""

            response = ollama.chat(
                model=self.model,
                options={"num_ctx": 512},
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            )

            answer = response["message"]["content"]

            # --------------------------------
            # STEP 4: Return grounded result
            # --------------------------------

            return {
                "question": question,
                "answer": answer,
                "evidence": evidence
            }
        except Exception as e:
            print(f"Ollama error: {e}")
            return {
                "question": question,
                "answer": f"AI service error: {str(e)}. Based on retrieved evidence:\n\n{context}",
                "evidence": evidence,
                "fallback": True
            }


if __name__ == "__main__":

    agent = CacheAnalystAgent()

    question = input(
        "\nAsk SmartEdge a question: "
    )

    result = agent.analyze(question)

    print("\n==============================")
    print("SMARTEDGE CACHE ANALYST AGENT")
    print("==============================")

    print("\nQUESTION:")
    print(result["question"])

    print("\nAI ANSWER:")
    print(result["answer"])

    print("\nRETRIEVED EVIDENCE:")

    for item in result["evidence"]:

        print(
            f"- {item['source']} "
            f"(distance: {item['distance']:.4f})"
        )