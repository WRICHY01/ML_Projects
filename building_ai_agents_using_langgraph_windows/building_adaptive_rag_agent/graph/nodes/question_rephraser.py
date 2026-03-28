from typing import Dict, Any

from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

from ..chains.question_remodifier import structured_query_chain
# from  ...llm.llm_model import llm_model
# from .grade_documents import documents_grader
from ..state import AgentState, IneligibleQuery



def question_transformer(state: AgentState) -> Dict[str, Any]:
    """

    """
    print("processing questions...")
    questions = state["questions"]
    # ineligible_queries = state["ineligible_queries"]
    # current_questions = original_questions #########################
#     questions = state[""]
    print(f"questions in question_rephraser node is: {questions}")
    is_initial_queries = state.get("is_initial_queries", False)
    should_rewrite = state.get("should_rewrite")
    rewrite_queries = state.get("rewrite_queries", [])
    # retrieval_evaluations = state["retrieval_evaluations"]
    valid_questions = []
    ineligible_queries = []
    current_questions = []
    query_history = []
    # print("why are my getting empty list of questions thus: ", questions)
    # retrieval_evaluations = state["retrieval_evaluations"]
    
    
    # should_rewrite_count += 1
    # needs_preprocessing = state.get("needs_preprocessing", None)
    # print("getting value")

    # The goal here is that questions repharaser node takes in 3 types of objects which are:
    # -list of string (in the case where its the first input)
    # -an object containing past query history and the current questions (in the case where there is a follow up questions)
    # -a list of objects that needs rephrasing(in case there is need for rephrasing from the doc grader node)
    # but in all case the output is the same which is a structured_query object

    if not rewrite_queries:
        if not is_initial_queries:
            print("---PROCESSING INITIAL MESSAGE---")
            structured_queries = structured_query_chain.invoke({
                "questions": questions, #this prompt needs further refining because it needs to accept structured query object if needed i.e if "should_rewrite" flag is set to true
                                                    #    "needs_preprocessing": needs_preprocessing,
            })
            is_initial_queries = True
        else:
            print("---PROCESSING FOLLOW UP MESSAGE TOGETHER WITH PRIOR MESSAGE HISTORY")
            # Need to make this an object of query history and questions
            full_query_context = query_history + questions #history is placeholder for the history object that's gonna be generated in the generation node.
            structured_queries = structured_query_chain.invoke({
                "questions": full_query_context
            })
    else:
        print("---REPHRASING QUERIES THAT WAS SENT BACK FROM THE DOCS GRADER NODE--")
        structured_queries = structured_query_chain.invoke({
            "questions": rewrite_queries
        })
    # should_rewrite_count += 1
    print(")0"*300)
    print("structured_query looks thus", structured_queries, "and its type is: ", type(structured_queries))
    print("()0"*300)
    # valid_questions = [structured_query for structured_query in structured_queries.sub_queries if structured_query.question_status.lower() == "valid"]
    # invalid_questions = [structured_query for structured_query in structured_queries.sub_queries if structured_query.question_status.lower() == "invalid"]
    for structured_query in structured_queries.sub_queries:
        current_questions.append(HumanMessage(content=structured_query.rephrased_question))
        if structured_query.question_status in ["AMBIGUOUS", "INVALID"]:
            # This will be merged later on in the generate node with the valid questions to generate appropriate response together with the valid questions
            ineligible_query = IneligibleQuery(
                reasoning = structured_query.reasoning,
                extracted_questions = structured_query.rephrased_question,
                question_status = structured_query.question_status,
            )
            ineligible_queries.append(ineligible_query)
        else:
            valid_questions.append(structured_query)

    query_history.append(current_questions)   
    # invalid_questions = [structured_query.extract]
    structured_queries.sub_queries = valid_questions
    # return {"rewritten_questions":.rewritten_questions,
    #         # "needs_preprocessing": needs_preprocessing,
    #         "should_rewrite": should_rewrite}
    #         # "should_rewrite_count": should_rewrite_count}
    # print("\n\nThis is the outcome of how the structured query looks like: ", structured_queries.sub_queries, " and how the invalid questions looks like: ", invalid_questions, "\n\n")
    return {
        # "invalid_questions":  invalid_questions,
        "is_initial_queries": is_initial_queries,
        "structured_query": structured_queries,
        "ineligible_queries": ineligible_queries}



# print(question_transformer({"questions": "How many countries are there in a continent?", "rewrite_flag": True}))