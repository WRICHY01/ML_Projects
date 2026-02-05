from typing import Any, Dict

from langchain_core.documents import Document

from .llm_knowledge import generate_using_llm_knowledge
from .web_search import fetch_from_web_search 
from ..state import AgentState
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

    question = state["question"]
    print(f"question in document_grader node is: {question}")
    documents = state["documents"]
    should_rewrite = state["should_rewrite"]
    should_rewrite_count = state.get("should_rewrite_count", 0)
    # is_relevant = state.get("is_relevant", None)

    filtered_docs = []
    # web_search = False #possible change it to source_external 
    # allow_llm_knowledge
    # use_web_search
    # allow_external_sources = False
    should_route_to_llm_or_web = state.get("should_route_to_llm_or_web")
    should_route_to_llm_or_web_count = state.get("should_route_to_llm_or_web_count", 0)
    print(f"documents contains: {documents}")
    if should_rewrite_count > 5: # This would be taken care of in one of the conditional edges in graph.py
        raise RuntimeError("Internal Error, rewrite count exceeded!") #we still need to verify tho
    

    docs_length = len(documents)
    
    for n in range(docs_length):
        print(f"in the for loop and the content of documents is thus;  {documents[n]}")
        score = doc_grader.invoke({"question": question, "documents": documents[n].page_content})
        
        is_relevant = score.binary_score.lower()

        if is_relevant == "yes":
            print("---DOCUMENTS IS RELEVANT TO QUESTION---")
            # remove any duplicates
            # if documents[n] not in filtered_docs:
            print(f"content of a single document is {documents[n]}")
            print(f"content of the single document is {documents[n].page_content}")
            filtered_docs.append(documents[n])
            print(f"filtered_docs now looks like: {filtered_docs}")
            # print(f"but looks like this when put in a list thus: {[filtered_docs]}")
#*******************WORK NEEDS TO BE DONE HERE TO ALSO ALLOW FOR EXTERNAL KNOWLEDGE(LLM OR WEB_SEARCH*************************************#
        else: 
            print("---DOCUMENTS NOT RELEVANT  QUESTION---")
            # if the docs is indeed available in the vector database and proper rephrasing is needed
            # should_route_to_llm_or_web_count += 1
            should_rewrite = True
            if should_rewrite_count >= 2: # use the if else condition in the conditonal edge
                # print("---ROUTING TO QUESTION REPHRASER--")
                # should_rewrite = True
                should_route_to_llm_or_web = True
                should_route_to_llm_or_web_count += 1

            should_rewrite_count += 1
            
            continue

                
            # route to llm or web if it has rephrased twice and no reasonable answer was retrieved
            # else: 
                
            

                

        # least no of docs to be used answer a question should be 50 should incase some docs answers the question
        # while dome doesnt
        # if is_relevant == "yes":
        # if len(documents) < 50:
        #     print("---DOCUMENTS NOT ENOUGH TO ANSWER QUESTION---")
        #     should


    return {
        "question": question, 
        "documents": filtered_docs,
        "is_relevant": is_relevant,
        "should_rewrite": should_rewrite,
        "should_rewrite_count": should_rewrite_count,
        "should_route_to_llm_or_web": should_route_to_llm_or_web,
        "should_route_to_llm_or_web_count": should_route_to_llm_or_web_count
        }


# print(documents_grader({"question": "what's a weird thing about insects", "documents": [Document(page_content="Somw insects gives birth and then die, while some insect kills their mates")]}))  



