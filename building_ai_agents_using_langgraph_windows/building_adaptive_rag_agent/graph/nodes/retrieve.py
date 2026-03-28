from typing import Dict, Any

from ..state import AgentState, QueryContext
from ...data_ingestion.ingestion import retriever


def retrieve_from_vectorstore(state: AgentState) -> Dict[str, Any]:
    """
    Takes the state, retrieves docs based on similarity relative the question,
    and returns the updated state.

    Args:
        state (AgentState): The current graph state containing the user question.

    Returns:
        dict: A dictionary containing the list of retrieved documents to update the state.
    """

    print("---RETRIEVING DOCUMENTS---")
    question = state["question"]
    query_router = state["query_router"]
    retrieved_documents = state.get("retrieved_documents", [])
    extracted_questions = state.get("extracted_questions", [])
    vs_query_contexts = []
    # print("Retrieved_documents at the very beginning of vector_store node is: ???????????????????????????????", retrieved_documents)
    # documents = state["documents"]
    print("query_router in retrieve node looks thus: ", query_router.routes)
    for qr_obj in query_router.routes:
        print("now in the loop and the current object being verified is :",  qr_obj)
        if qr_obj.data_source == "vector_store":
            extracted_questions.append(qr_obj.extracted_question)
            print("checking if on the right track, the extracted question is thus: ", qr_obj.extracted_question)
            print("checking to see if it'd move to the next node here...")
            retrieved_document = retriever.invoke(qr_obj.extracted_question)
            print(f"\n\n<<<<<<<<<<<<<<<<<<<<<<<<<<<The retrieved doc for {qr_obj.extracted_question} is {retrieved_document}>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>\n\n")
            # consolidated into a class for the purpose of the document grader node
            vs_query_context = QueryContext(
                                extracted_questions=qr_obj.extracted_question,
                                retrieved_documents=retrieved_document,
                                data_source=qr_obj.data_source
                            )
            # append all the Query context object to the vs(vector_store)_query_contexts list
            vs_query_contexts.append(vs_query_context)
            retrieved_documents.append(retrieved_document)
            print(f"{extracted_questions}\n{vs_query_contexts}")
            print("\n############################retrieved_documents now looks like this: ", retrieved_documents)

    all_retrieved_documents = [retrieved_documents]
    
    print("\nretrieved_documents in vector_store look thus: <<<<<<<<<<<>>>>>>>>>>", all_retrieved_documents)
    
    return {"extracted_questions": extracted_questions,
            "documents": retrieved_documents,
            "retrieved_documents": retrieved_documents,
            "all_query_contexts": vs_query_contexts}



