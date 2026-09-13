import os
import pandas as pd
import ollama


BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

RESULT_FILE = os.path.join(
    BASE_DIR,
    "rag",
    "data",
    "adaptive_threshold_results_v3.csv"
)


class TuningAgent:

    def __init__(self):

        print("Initializing SmartEdge Tuning Agent...")

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

        if not os.path.exists(RESULT_FILE):

            raise FileNotFoundError(
                f"Adaptive benchmark file not found: {RESULT_FILE}"
            )

        self.data = pd.read_csv(
            RESULT_FILE
        )

    def analyze(self, question):

        data = self.data.copy()

        # --------------------------------
        # Find best overall SmartEdge result
        # --------------------------------

        best_result = data.loc[
            data["smartedge_hit_rate"].idxmax()
        ]

        evidence = []

        evidence.append(
            f"Best overall SmartEdge hit rate: "
            f"{best_result['smartedge_hit_rate']:.2f}%"
        )

        evidence.append(
            f"Capacity: "
            f"{int(best_result['capacity'])}"
        )

        evidence.append(
            f"Threshold: "
            f"{best_result['threshold']}"
        )

        evidence.append(
            f"LRU hit rate: "
            f"{best_result['lru_hit_rate']:.2f}%"
        )

        evidence.append(
            f"LFU hit rate: "
            f"{best_result['lfu_hit_rate']:.2f}%"
        )

        evidence.append(
            f"SmartEdge improvement over LFU: "
            f"{best_result['vs_lfu']:.2f} percentage points"
        )

        evidence.append(
            f"SmartEdge improvement over LRU: "
            f"{best_result['vs_lru']:.2f} percentage points"
        )

        # --------------------------------
        # Best threshold for every capacity
        # --------------------------------

        evidence.append(
            "Best threshold by cache capacity:"
        )

        for capacity in sorted(
            data["capacity"].unique()
        ):

            subset = data[
                data["capacity"] == capacity
            ]

            best = subset.loc[
                subset["smartedge_hit_rate"].idxmax()
            ]

            evidence.append(
                f"Capacity {int(capacity)}: "
                f"threshold {best['threshold']}, "
                f"SmartEdge hit rate "
                f"{best['smartedge_hit_rate']:.2f}%, "
                f"LFU {best['lfu_hit_rate']:.2f}%, "
                f"vs LFU {best['vs_lfu']:.2f} pp"
            )

        # --------------------------------
        # Threshold behavior at capacity 100
        # --------------------------------

        capacity_100 = data[
            data["capacity"] == 100
        ]

        if not capacity_100.empty:

            evidence.append(
                "Threshold behavior at capacity 100:"
            )

            for _, row in capacity_100.iterrows():

                evidence.append(
                    f"Threshold {row['threshold']}: "
                    f"SmartEdge hit rate "
                    f"{row['smartedge_hit_rate']:.2f}%, "
                    f"admissions "
                    f"{int(row['admissions'])}, "
                    f"evictions "
                    f"{int(row['evictions'])}"
                )

        evidence_text = "\n".join(
            f"- {item}"
            for item in evidence
        )

        # --------------------------------
        # LLM prompt (if available)
        # --------------------------------

        if not self.ollama_available:
            # Fallback: return evidence directly
            return {
                "question": question,
                "recommendation": f"AI service unavailable. Based on experimental evidence:\n\n{evidence_text}",
                "evidence": evidence,
                "fallback": True
            }

        try:
            prompt = f"""
You are the SmartEdge Tuning Agent.

Your task is to analyze historical SmartEdge
threshold experiments and recommend useful
future experiments.

IMPORTANT RULES:

1. Use ONLY the supplied experimental evidence.
2. Never invent benchmark numbers.
3. Clearly distinguish observed results from
   recommendations.
4. Do NOT claim that a proposed threshold has
   already been validated.
5. Do NOT change SmartEdge configuration.
6. Recommendations must be tested through a
   benchmark before adoption.

USER QUESTION:

{question}

EXPERIMENTAL EVIDENCE:

{evidence_text}

Answer using this structure:

OBSERVED RESULT:
State what the experiments actually show.

RECOMMENDATION:
Suggest the next useful experiment.

REASON:
Explain why that experiment is useful.

CAUTION:
Explain that the recommendation must be benchmarked
before being used in production.
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

            recommendation = response[
                "message"
            ]["content"]

            return {
                "question": question,
                "recommendation": recommendation,
                "evidence": evidence
            }
        except Exception as e:
            print(f"Ollama error: {e}")
            return {
                "question": question,
                "recommendation": f"AI service error: {str(e)}. Based on experimental evidence:\n\n{evidence_text}",
                "evidence": evidence,
                "fallback": True
            }


if __name__ == "__main__":

    agent = TuningAgent()

    question = input(
        "\nAsk SmartEdge Tuning Agent: "
    )

    result = agent.analyze(
        question
    )

    print("\n================================")
    print("SMARTEDGE TUNING AGENT")
    print("================================")

    print("\nQUESTION:")
    print(result["question"])

    print("\nAI RECOMMENDATION:")
    print(result["recommendation"])

    print("\nEXPERIMENTAL EVIDENCE:")

    for item in result["evidence"]:

        print(
            f"- {item}"
        )