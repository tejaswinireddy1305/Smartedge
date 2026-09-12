from agents.cache_agent import CacheAnalystAgent
from agents.performance_agent import PerformanceAgent
from agents.tuning_agent import TuningAgent
from agents.smartedge_tools import get_live_context


class SmartEdgeOrchestrator:

    def __init__(self):

        print("Initializing SmartEdge Multi-Agent System...")

        self.cache_agent = CacheAnalystAgent()
        self.performance_agent = PerformanceAgent()
        self.tuning_agent = TuningAgent()

    def select_agent(self, question):

        q = question.lower()

        # --------------------------------
        # Tuning Agent keywords
        # --------------------------------

        tuning_keywords = [
            "threshold",
            "tune",
            "tuning",
            "recommend",
            "recommendation",
            "optimize",
            "optimization",
            "next experiment",
            "what should i test"
        ]

        # --------------------------------
        # Performance Agent keywords
        # --------------------------------

        performance_keywords = [
            "compare",
            "comparison",
            "performance",
            "hit rate",
            "lfu",
            "lru",
            "benchmark",
            "capacity",
            "eviction",
            "admission"
        ]

        # --------------------------------
        # Cache Analyst keywords
        # --------------------------------

        cache_keywords = [
            "cache",
            "cached",
            "caching",
            "request",
            "prediction",
            "predicted",
            "reject",
            "rejected",
            "decision",
            "reuse"
        ]

        # --------------------------------
        # Select appropriate agent
        # --------------------------------

        if any(
            keyword in q
            for keyword in tuning_keywords
        ):
            return "tuning"

        if any(
            keyword in q
            for keyword in performance_keywords
        ):
            return "performance"

        if any(
            keyword in q
            for keyword in cache_keywords
        ):
            return "cache"

        # Default agent
        return "cache"

    def ask(self, question):

        # --------------------------------
        # Get current SmartEdge state
        # --------------------------------

        live_context = get_live_context()

        # --------------------------------
        # Select agent
        # --------------------------------

        agent_type = self.select_agent(
            question
        )

        # --------------------------------
        # Tuning Agent
        # --------------------------------

        if agent_type == "tuning":

            result = self.tuning_agent.analyze(
                question
            )

            return {
                "agent": "Tuning Agent",
                "result": result,
                "live_context": live_context
            }

        # --------------------------------
        # Performance Agent
        # --------------------------------

        if agent_type == "performance":

            result = self.performance_agent.analyze(
                question
            )

            return {
                "agent": "Performance Agent",
                "result": result,
                "live_context": live_context
            }

        # --------------------------------
        # Cache Analyst Agent
        # --------------------------------

        result = self.cache_agent.analyze(
            question
        )

        return {
            "agent": "Cache Analyst Agent",
            "result": result,
            "live_context": live_context
        }


if __name__ == "__main__":

    orchestrator = SmartEdgeOrchestrator()

    question = input(
        "\nAsk SmartEdge: "
    )

    result = orchestrator.ask(
        question
    )

    print("\n================================")
    print("SMARTEDGE MULTI-AGENT SYSTEM")
    print("================================")

    print("\nSELECTED AGENT:")
    print(result["agent"])

    print("\nRESULT:")

    agent_result = result["result"]

    # --------------------------------
    # Display Cache Analyst result
    # --------------------------------

    if "answer" in agent_result:

        print(
            agent_result["answer"]
        )

    # --------------------------------
    # Display Tuning Agent result
    # --------------------------------

    elif "recommendation" in agent_result:

        print(
            agent_result["recommendation"]
        )

    # --------------------------------
    # Fallback
    # --------------------------------

    else:

        print(
            agent_result
        )

    # --------------------------------
    # Display live SmartEdge context
    # --------------------------------

    print("\n================================")
    print("LIVE SMARTEDGE CONTEXT")
    print("================================")

    print(
        result["live_context"]
    )