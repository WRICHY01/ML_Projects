from typing import Dict, Any

from ..state import AgentState
from ..chains.hallucination_grader import hallucination_chain

def check_hallucination(state: AgentState) -> Dict[str, Any]:
    """
    Evaluates whether  generated answer is grounded in the provided documents.

    Args:
        state(AgentState): The current graph state containing documents and  generated_answer

    Returns:
        ######################################
    """
    print("---CHECKING FOR HALLUCINATIONS.---")
    # question = state["question"]
    
    knowledge_source = state["knowledge_source"]
    documents = state["documents"]
    generated_answer = state["generated_answer"]

    print(f"KNOWLEDGE_SOURCE IS THUS: {knowledge_source}")


    print(f"the content of documents in the hallucination checker is {documents}")
    # allow_external_sources = state["allow_external_sources"]

    # print("done with adding the needed variables... ")
    # if not generation and not allow_external_sources:
    #     if something:
    #         llm
    #     else:
    #         web_search
    hallucination_check = hallucination_chain.invoke({"knowledge_source": knowledge_source,
                                                      "generated_answer": generated_answer,
                                                      "documents": documents})
    # if hallucination_check.binary_score.lower() == "yes":
    #     print("---DECISION: ANSWER IS GROUNDED---")
    #     return "grounded"
    # else:
    #     print("---DECISION: ANSWER IS HALLUCINATED---")
    #     return "hallucinated"
    return {"hallucination_status": hallucination_check.binary_score}