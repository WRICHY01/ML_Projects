from typing import Dict, Any

from ..state import AgentState
from ..chains.generated_answer_grader import generated_answer_chain 


def check_answer_relevance(state: AgentState) -> Dict[str, Any]:
    """
    Evaluates whether the generated answer answers the user's question.
    
    Args:
        state(AgentState): The current graph state containing user's question and  generated_answer

    Returns:
        ####################
    """
    print("---CHECKING FOR GENRATED ANSWER CORRECTNESS---")
    
    question = state["question"]
    print(f"question in answer_relevance node is: {question}")
    generated_answer = state["generated_answer"]

    generated_answer_check = generated_answer_chain.invoke({"question": question,"generated_answer": generated_answer})


    return {"generated_answer_grader_status": generated_answer_check.binary_score}