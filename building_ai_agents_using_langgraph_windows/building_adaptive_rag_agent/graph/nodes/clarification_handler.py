from typing import Dict, Any

from ..chains.question_refiner import question_spewer_chain

def clarify_questions(state: AgentState) -> Dict[str, Any]:
    question = state["question"]

    question_s = question_spewer_chain.invoke{
                        "question": question
                        }


    return {"question": question_s}