import os
import pandas as pd
import ollama


BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

BENCHMARK_FILE = os.path.join(
    BASE_DIR,
    "rag",
    "data",
    "final_cache_comparison.csv"
)


class PerformanceAgent:

    def __init__(self):

        print("Initializing SmartEdge Performance Agent...")

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

        if not os.path.exists(BENCHMARK_FILE):

            raise FileNotFoundError(
                f"Benchmark file not found: {BENCHMARK_FILE}"
            )

        self.data = pd.read_csv(
            BENCHMARK_FILE
        )

    def analyze(self, question):

        data = self.data.copy()

        # --------------------------------
        # Calculate performance findings
        # --------------------------------

        smartedge = data[
            data["algorithm"].str.lower()
            == "smartedge"
        ]

        lru = data[
            data["algorithm"].str.lower()
            == "lru"
        ]

        lfu = data[
            data["algorithm"].str.lower()
            == "lfu"
        ]

        findings = []

        if not smartedge.empty:

            smartedge_row = smartedge.iloc[0]

            findings.append(
                f"SmartEdge hit rate: "
                f"{smartedge_row['hit_rate']:.3f}%"
            )

            findings.append(
                f"SmartEdge requests: "
                f"{int(smartedge_row['requests'])}"
            )

            findings.append(
                f"SmartEdge hits: "
                f"{int(smartedge_row['hits'])}"
            )

            findings.append(
                f"SmartEdge misses: "
                f"{int(smartedge_row['misses'])}"
            )

            findings.append(
                f"SmartEdge evictions: "
                f"{int(smartedge_row['evictions'])}"
            )

            findings.append(
                f"SmartEdge proactive insertions: "
                f"{int(smartedge_row['proactive_insertions'])}"
            )

        if not lru.empty:

            lru_row = lru.iloc[0]

            findings.append(
                f"LRU hit rate: "
                f"{lru_row['hit_rate']:.3f}%"
            )

        if not lfu.empty:

            lfu_row = lfu.iloc[0]

            findings.append(
                f"LFU hit rate: "
                f"{lfu_row['hit_rate']:.3f}%"
            )

        if not smartedge.empty and not lfu.empty:

            improvement_lfu = (
                smartedge.iloc[0]["hit_rate"]
                -
                lfu.iloc[0]["hit_rate"]
            )

            findings.append(
                f"SmartEdge difference versus LFU: "
                f"{improvement_lfu:.3f} percentage points"
            )

        if not smartedge.empty and not lru.empty:

            improvement_lru = (
                smartedge.iloc[0]["hit_rate"]
                -
                lru.iloc[0]["hit_rate"]
            )

            findings.append(
                f"SmartEdge difference versus LRU: "
                f"{improvement_lru:.3f} percentage points"
            )

        findings_text = "\n".join(
            f"- {item}"
            for item in findings
        )

        # --------------------------------
        # Ask Llama to explain findings (if available)
        # --------------------------------

        if not self.ollama_available:
            # Fallback: return findings directly
            return {
                "question": question,
                "answer": f"AI service unavailable. Based on benchmark data:\n\n{findings_text}",
                "benchmark_source": os.path.basename(BENCHMARK_FILE),
                "findings": findings,
                "fallback": True
            }

        try:
            prompt = f"""
You are the SmartEdge Performance Analyst.

Analyze the benchmark information below.

IMPORTANT:
Use ONLY the supplied numerical evidence.
Do not invent numbers.
Do not change any numerical values.

Explain the performance clearly for a technical
project report.

USER QUESTION:
{question}

BENCHMARK EVIDENCE:
{findings_text}

Give:
1. Direct answer
2. Important numerical comparison
3. Short interpretation
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

            return {
                "question": question,
                "answer": answer,
                "benchmark_source": os.path.basename(
                    BENCHMARK_FILE
                ),
                "findings": findings
            }
        except Exception as e:
            print(f"Ollama error: {e}")
            return {
                "question": question,
                "answer": f"AI service error: {str(e)}. Based on benchmark data:\n\n{findings_text}",
                "benchmark_source": os.path.basename(BENCHMARK_FILE),
                "findings": findings,
                "fallback": True
            }


if __name__ == "__main__":

    agent = PerformanceAgent()

    question = input(
        "\nAsk SmartEdge Performance Agent: "
    )

    result = agent.analyze(
        question
    )

    print("\n================================")
    print("SMARTEDGE PERFORMANCE AGENT")
    print("================================")

    print("\nQUESTION:")
    print(result["question"])

    print("\nAI ANALYSIS:")
    print(result["answer"])

    print("\nNUMERICAL EVIDENCE:")

    for finding in result["findings"]:

        print(
            f"- {finding}"
        )