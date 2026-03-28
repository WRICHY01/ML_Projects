from typing import Dict, Any

from ..state import AgentState
from ..chains.hallucination_grader import hallucination_chain


class HallucinationSubmission(BaseModel):
    question_id: str
    retrieved_docs: list[Document]
    generated_answer: list[Document]


def check_hallucination(state: AgentState) -> Dict[str, Any]:
    """
    Evaluates whether  generated answer is grounded in the provided documents.

    Args:
        state(AgentState): The current graph state containing documents and  generated_answer

    Returns:
        ######################################
    """
    print("---CHECKING FOR HALLUCINATIONS.---")
    
    
    all_retrieved_query_evaluations = state["all_retrieved_query_evaluations"]
    # # allow_external_sources = state["allow_external_sources"]
    hallucination_submissions = []

    # for retrieved_evaluation in all_retrieved_query_evaluations:

    #     hallucination_check = hallucination_chain.invoke({
    #         "generated_answer": retrieved_evaluation.generated_answer,
    #         "documents": retrieved_evaluation.retrieved_documents
    #     })
        
    #     retrieved_evaluation.hallucination_status = hallucination_check

    for retrieved_query_evaluation in all_retrieved_query_evaluations:
        hallucination_submission = HallucinationSubmission(
            question_id=retrieved_query_evaluation.query_id,
            # question=retrieved_query_evaluation.extracted_question,
            retrieved_docs=retrieved_query_evaluation.retrieved_docs,
            generated_answer=retrieved_query_evaluation.generated_answer
        )
        
        hallucination_submissions.append(hallucination_submission)
        # retrieved_query_evaluation.generated_answer_grader_status = generated_answer_check
    grounding_results = hallucination_chain.invoke(
        {
            "submission": hallucination_submissions
        }
    )

    no_of_grounded_results = len(grounding_results)
    for i in range(no_of_grounded_results):
        



    return {"all_retrieved_query_evaluations": all_retrieved_query_evaluations}