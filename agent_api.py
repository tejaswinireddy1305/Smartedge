from fastapi import APIRouter
from pydantic import BaseModel
import traceback

from agents.orchestrator import SmartEdgeOrchestrator


router = APIRouter(
    prefix="/agent",
    tags=["AI Agent"]
)


# Lazy initialization to avoid startup errors
orchestrator = None


def get_orchestrator():
    global orchestrator
    if orchestrator is None:
        try:
            orchestrator = SmartEdgeOrchestrator()
        except Exception as e:
            print(f"Error initializing orchestrator: {e}")
            print(traceback.format_exc())
            return None
    return orchestrator


class AgentRequest(BaseModel):
    question: str


@router.post("/ask")
def ask_agent(request: AgentRequest):

    try:
        orch = get_orchestrator()
        if orch is None:
            return {
                "status": "error",
                "message": "Failed to initialize agent system. Check Ollama is running with: ollama serve"
            }

        result = orch.ask(
            request.question
        )

        return {
            "status": "success",
            "question": request.question,
            "selected_agent": result["agent"],
            "result": result["result"],
            "live_context": result["live_context"]
        }
    except Exception as e:
        print(f"Error in agent API: {e}")
        print(traceback.format_exc())
        return {
            "status": "error",
            "message": str(e)
        }