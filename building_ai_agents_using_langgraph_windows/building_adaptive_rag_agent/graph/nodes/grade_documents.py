from typing import Any, Dict

from langchain_core.documents import Document

from .llm_knowledge import generate_using_llm_knowledge
from .web_search import fetch_from_web_search 
from ..state import AgentState, RetrievalEvaluation, RewriteQuery
from ..chains.doc_retrieval_grader import doc_grader


def documents_grader(state: AgentState) -> Dict[str, Any]:
    """
    Filters retrieved documents for relevance to the user question.

    This function evaluates each document in the current state. If a document is deemed irrelevant by the doc_grader,
    it is excluded from the returned list, If at least one document is irrelevant, the 'web_search' changes to True.

    Args:
        state(AgentState): The current graph state.

    Returns:
        Dict[str, Any]: A dictionary containing the filterd documents and the websearch flag.
    """
    print("---CHECKING DOCUMENTS RELEVANCE TO QUESTION---")

    current_questions = state["current_questions"]
    extracted_questions = state["extracted_questions"]

    retrieved_documents = state.get("retrieved_documents", [])
    documents = state.get("documents", [])
    all_query_contexts = state["all_query_contexts"]
    ineligible_query_evaluations = state["ineligible_query_evaluations"]
    print("All query context at the very beginning of document grader is: ?????????????????????????????????????????", all_query_contexts)
    should_rewrite = False
    # should_rewrite_count = state.get("should_rewrite_count", 0)
    # is_relevant = state.get("is_relevant", None)
    # vector_store_attempt_count = 0
    # llm_knowledge_attempt_count = 0
    # web_search_attempt_count = 0

    filtered_docs = []
    rewrite_queries = []
    all_graded_doc_evaluations = []
    # web_search = False #possible change it to source_external 
    # allow_llm_knowledge
    # use_web_search
    # allow_external_sources = False
    
    docs_length = len(all_query_contexts)
    print("No of all query context is thus: ", docs_length, "context query objects")
    # if something:
    #     
    #     all_query_contexts = RetrievalEvaluation
    # for n in range(docs_length):
    for query_context in all_query_contexts:

        # print(f"in the for loop and the content of documents is thus;  {retrieved_documents[n]}")
        print("query_context before being verified is thus: ", query_context)
        score = doc_grader.invoke({"question": query_context.extracted_questions,
                                   "retrieved_documents": query_context.retrieved_documents,})
                                #    "documents": retrieved_documents[n].page_content})
        
        is_relevant = score.binary_score.lower()

        #in order not to waste tokens it better to just create a new object instead of telling it to fill in the questions and retrieved documents in the gradedocument object
        relevant = []
        irrelevant = []

        # I might comment this if section and only leave the else section i.e make it look thus if is_relevant == "no"
        if is_relevant == "yes":
            # print("---DOCUMENTS IS RELEVANT TO QUESTION---")
            print("query_context in if statement is thus: ", query_context)
            relevant.append(query_context)
            # remove any duplicates
            # if documents[n] not in filtered_docs:
            # print(f"content of a single document is {retrieved_documents[n]}")
            # print(f"content of the single document is {retrieved_documents[n].page_content}")
            # filtered_docs.append(retrieved_documents[n])
            print(f"filtered_docs now looks like: {filtered_docs}")
            # print(f"but looks like this when put in a list thus: {[filtered_docs]}")
#*******************WORK NEEDS TO BE DONE HERE TO ALSO ALLOW FOR EXTERNAL KNOWLEDGE(LLM OR WEB_SEARCH*************************************#
        if is_relevant == "no":
            # print("---DOCUMENTS NOT RELEVANT QUESTION---")
            # current_questions.append(query_context.extracted_questions)
            print("query_context in the else statement is thus: ", query_context)

            should_rewrite = True
            rewrite_query = RewriteQuery(
                question=query_context.extracted_questions,
                should_rewrite=should_rewrite
            )
            rewrite_queries.append(rewrite_query)

            if query_context.data_source == "vector_store":
                query_context.vector_store_attempt_count += 1
            elif query_context.data_source == "llm_knowledge":
                query_context.llm_knowledge_attempt_count += 1
            else:
                query_context.web_search_attempt_count += 1
            
            print("retrieved documents field before being set to an empty list is ", query_context.retrieved_documents)
            # sets the retrieved_documents field to an empty list, this would be needed in the generate node
            # to give a response that the question could not be answered
            query_context.retrieved_documents = []

            # if 

            print("retrieved_documents field after being set to an empty list is ", query_context.retrieved_documents)
            print("The full current query_context object now looks thus: ", query_context)

            should_rewrite_count += 1
            
        graded_doc_evaluation = RetrievalEvaluation(
            extracted_questions=query_context.extracted_questions,
            data_source=query_context.data_source,
            retrieved_documents=query_context.retrieved_documents,
            is_relevant=score.binary_score,
            should_rewrite=should_rewrite
        )


        all_graded_doc_evaluations.append(graded_doc_evaluation)

    print("The contents of all graded docs looks thus: ", all_graded_doc_evaluations)

    # all_retrieved_query_evaluations = graded_doc_evaluation + ineligible_query_evaluations 
            # route to llm or web if it has rephrased twice and no reasonable answer was retrieved
            # else: 


    return {
        # "question": question, 
        "retrieved_documents": filtered_docs,
        "is_relevant": is_relevant,
        "should_rewrite": should_rewrite,
        "should_rewrite_count": should_rewrite_count,
        # "should_route_to_web": should_route_to_web,
        # "vector_store_attempt_count": vector_store_attempt_count,
        # "llm_knowledge_attempt_count": llm_knowledge_attempt_count,
        # "web_search_attempt_count": web_search_attempt_count,
        "rewrite_queries": rewrite_queries,
        "retrieval_evaluations": all_graded_doc_evaluations,
        # "all_retrieved_query_evaluations": all_retrieved_query_evaluations,
        # "should_route_to_llm_or_web_count": should_route_to_llm_or_web_count
        }


# print(documents_grader({"question": "what's a weird thing about insects", "documents": [Document(page_content="Somw insects gives birth and then die, while some insect kills their mates")]}))  



