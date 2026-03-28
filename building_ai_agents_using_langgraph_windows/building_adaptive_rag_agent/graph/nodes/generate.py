from typing import Dict, Any, Literal, Optional

from ..state import AgentState
from ..chains.answer_generation import answer_generation_chain

class QueryInferenceInput(BaseModel):
    question: str
    context: Optional[list[Document]] = None
    history: Optional[list[Document]] = None
    question_status: Literal["VALID", "AMBIGUOUS", "INVALID"]


def generate_response(state: AgentState) -> Dict[str, Dict]:
    """
    Generate an answer using the retrieved documents and the original question.

    This node takes  the current state, extract the accumulated documents 
    and the user's qustion, and passes them  to the generation chain. 
    the resulting generation is then added to the state.

    Args:
        state(GraphState): The current state of the graph containing the user's input query,
        and a list of retrieved documents for context.

    Returns:
        Dict[str, Any]: A dictionary updating the state with the  'generation' key
    """

    print("---GENERATING ANSWER USING RETRIEVED DOCUMENTS---")
    
    structured_query = state.get("structured_query", [])
    retrieval_evaluations = state["retrieval_evaluations"]
    ineligible_query_evaluations = state["ineligible_query_evaluations"]
    # ineligible_queries = state["ineligible_queries"]
    # if-else condition incase the user input is all gibberish
    if not structured_query.sub_queries:
        # Incase the user inputs gibberish all through and no reasonable/sensible question
        all_retrieved_query_evaluations = ineligible_query_evaluations
    else:
        all_retrieved_query_evaluations = retrieval_evaluations + ineligible_query_evaluations
    # all_retrieved_evaluations = retrieval_evaluations + ineligible_queries
    # all_retrieved_query_evaluations = state["all_retrieved_query_evaluations"]
    query_contexts = []
    # Cant afford for other field which isnt used here influence its outcome, hence my reason for iterating and filtering the ones will be using to feed into the llm
    for retrieved_query_evaluation in all_retrieved_query_evaluations:
        query_context = QueryInferenceInput(
            question=retrieved_query_evaluation.extracted_questions,
            context=retrieved_query_evaluation.retrieved_documents,
            history=retrieved_query_evaluation.summarized_generated_answer,
            question_status=retrieved_query_evaluation.question_status
        )
        query_contexts.append(query_context)
    # all_questions = [all_retrieved_query.extracted_questions for all_retrieved_query in all_retrieved_query_evaluations]
    # Refactored this to compile all the query object in a list so the llm is being called once instead of multiple times
    generated_answers = answer_generation_chain.invoke(
        {"query_contexts", query_contexts # wrong key value pair nomenclature, the value should be a list of object
         }
    )

    # for generated_answer in generated_answer:
    # if len(all_retrieved_query_evaluations) == len(generated_answers):
    for i in range(len(generated_answers)):
        print("single object in rq_container being selected using query index: ", query_contexts[generated_answers[i].query_index])
        print("single object in rq_evaluations being selected using query index: ", all_retrieved_query_evaluations[generated_answers[i].query_index])
        print("all object in rq_container being selected using no query index: ", query_contexts)
        
        all_retrieved_query_evaluations[generated_answers[i].query_index].generated_answer = generated_answers[i].generated_answer
        all_retrieved_query_evaluations[generated_answers[i].query_index].summarized_generated_answer = generated_answers[i].summarized_generated_answer
        
        print("singel object in rq_evaluations after adding extra 2 fields: ", all_retrieved_query_evaluations[generated_answers[i].query_index])


    # for retrieved_evaluation in all_retrieved_query_evaluations:
    #     generated_answer = answer_generation_chain.invoke(
    #                     {
    #                         "question": retrieved_evaluation.extracted_questions,
    #                         "history": retrieved_evaluation.summarized_generated_answer,
    #                         "context": retrieved_evaluation.retrieved_documents,
    #                         "question_status": retrieved_evaluation.question_status
    #                     }
    #                 )
    #     retrieved_evaluation.generated_answer = generated_answer.generated_answer
    #     retrieved_evaluation.summarized_generated_answer = generated_answer.summarized_generated_answer
        
        
    return {
        "all_retrieved_query_evaluations": all_retrieved_query_evaluations,
        "retrieval_evaluations": retrieval_evaluations
        }