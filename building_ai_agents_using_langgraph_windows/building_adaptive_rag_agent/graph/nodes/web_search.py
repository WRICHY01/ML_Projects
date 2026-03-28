from typing import Dict, Any

from langchain_tavily import TavilySearch
from langchain_core.documents import Document

from ..state import AgentState, SubQueryRouter, QueryContext
from dotenv import load_dotenv

load_dotenv()

web_search_tool = TavilySearch(max=3)
def fetch_from_web_search(state: AgentState) -> Dict[str, Any]:
    """
    Performs a web search based on the question stored in the current graph.

    This function uses Tavily search engine to retrieve the top 3 relevant results,
    formats them into a single Langchain Document, and appends them to the existing document collection in the state.
    Args:
        state (AgentState): The current state of the graph.

    Returns:
        Dict[str, Any]: An updated state dictionary containing the original search query, 
        and the list of updated search result documents
    """
    question = state["question"]
    extracted_questions = state.get("extracted_questions", [])
    retrieved_documents = state.get("retrieved_documents", [])
    # print("Retrieved_documents at the very beginning of web_search node is: ?????????????????????????????", retrieved_documents)
    query_router = state["query_router"]
    print(f"question in web_search node is: {question}")
    # documents = state.get("documents", [])
    web_search_count = state.get("web_search_count", 0)
    all_retrieved_query_evaluations = state["all_retrieved_query_evaluations"]

    tavily_web_search_results = None #Look into this soon!
    ws_query_contexts = []
    for qr_obj in query_router.routes:
        # The object could be from the router node or from the conditional edge function it doesnt matter since both of them have the same field(field of importance needed for the node to carry out its task)
        if qr_obj.data_source == "web_search":
            # use an if statement here to separate the instance
            # qr_obj.routed_to_web_search = True
            extracted_questions.append(qr_obj.extracted_question)
            tavily_web_search_results = web_search_tool.invoke({"query": qr_obj.extracted_question})["results"]
            print(f"tavily_web_search_results after fetching from the internet/web is thus: {tavily_web_search_results}")

            formatted_tavily_web_search_results = [Document(page_content=tavily_web_search_result["content"]) for tavily_web_search_result in tavily_web_search_results]
            ws_query_context = QueryContext(
                extracted_questions=qr_obj.extracted_question,
                retrieved_documents=formatted_tavily_web_search_results,
                data_source=qr_obj.data_source
            )
            # retrieved_documents.append(formatted_tavily_web_search_results)
            ws_query_contexts.append(ws_query_context)

            print("The content of extracted_questions and the ws_query_contexts in the web_search node is: ", extracted_questions, "\n", ws_query_contexts)

    
    web_search_count += 1

    return {
        "extracted_questions": extracted_questions,
        "retrieved_documents": retrieved_documents,
        "web_search_count": web_search_count, 
        "all_query_contexts": ws_query_contexts
    }