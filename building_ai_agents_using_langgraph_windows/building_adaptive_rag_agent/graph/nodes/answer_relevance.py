from typing import Dict, Any

from ..state import AgentState
from ..chains.generated_answer_grader import generated_answer_chain 


class AnswerRelevanceSubmission(BaseModel):
    question_id: int
    question: str
    generated_answer: list[Document]

def check_answer_relevance(state: AgentState) -> Dict[str, Any]:
    """
    Evaluates whether the generated answer answers the user's question.
    
    Args:
        state(AgentState): The current graph state containing user's question and  generated_answer

    Returns:
        ####################
    """
    print("---CHECKING FOR GENERATED ANSWER CORRECTNESS---")
    
    question = state["question"]
    print(f"question in answer_relevance node is: {question}")
    generated_answer = state["generated_answer"]
    all_retrieved_query_evaluations = state["all_retrieved_query_evaluations"]
    answer_relevance_submissions = []

    for retrieved_query_evaluation in all_retrieved_query_evaluations:
        answer_relevance_submission = AnswerRelevanceSubmission(
            question_id=retrieved_query_evaluation.query_id,
            question=retrieved_query_evaluation.extracted_question,
            generated_answer=retrieved_query_evaluation.generated_answer
        )
        # generated_answer_check = generated_answer_chain.invoke({
        #     "question": retrieved_query_evaluation.extracted_questions,
        #     "generated_answer": retrieved_query_evaluation.generated_answer
        # })
        answer_relevance_submissions.append(answer_relevance_submission)
        # retrieved_query_evaluation.generated_answer_grader_status = generated_answer_check
    relevance_results = generated_answer_chain.invoke(
        {
            "submission": answer_relevance_submissions
        }
    )

    for i in range(len(relevance_results)):
        pass


    return {"all_retrieved_query_evaluations": all_retrieved_query_evaluations}