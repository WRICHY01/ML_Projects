from typing import Any, Dict

from ..state import AgentState


def generate_final_response(state: AgentState) -> Dict[str, Any]: #Node name needs renaming since we wont be using llm api to generate but instead just print out the output in a formatted and structured form,
    """
    
    """
    retrieval_evaluations = state["retrieval_evaluations"]
    all_retrieved_query_evaluations = state["all_retrieved_query_evaluations"]
    formatted_segment = []
    for i, retrieved_query_evaluation in enumerate(retrieval_evaluations):
        segment = f"Question {i}: {retrieved_query_evaluation.extracted_questions}\n\n{retrieved_query_evaluation.generated_answer}"
        formatted_segment.append(segment)

    formatted_final_response = "\n\n--\n\n".join(formatted_segment)

    return {"formatted_final_response": formatted_final_response}