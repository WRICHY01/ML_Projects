from typing import Dict, Any

from langchain_tavily import TavilySearch
from langchain_core.documents import Document

from ..state import AgentState
from dotenv import load_dotenv

load_dotenv()

web_search_tool = TavilySearch(max=5)
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
    print(f"question in web_search node is: {question}")
    documents = state.get("documents", [])
    web_search_count = state.get("web_search_count", 0)

    if web_search_count > 2:
        raise RuntimeError("Sorry no response could be provided..")

    # print(f"documents content in the websearch before being invoked: {documents})")
    print(f"reformed question looks thus: {question}")
    print(f"the last question is thus {question[-1]}")
    tavily_web_search_results = web_search_tool.invoke({"query": question[-1].content})["results"]
    # print(web_search_tool.invoke({"query": question}))
    if not tavily_web_search_results:
        raise ValueError("web_search result is empty")
    formatted_tavily_web_search_results = '\n'.join(
        [tavily_web_search_result["content"] for tavily_web_search_result in tavily_web_search_results]
    )
    
    web_search_results = Document(page_content=formatted_tavily_web_search_results)

    # #**************************************There is still work needed to be done here****************************************#
    if documents:
        documents.append(web_search_results) # I dont think there is a need for an if-else statement, since the document would always be empty since it wasnt populated when performing  retriever_grader operation
    else:
        documents = [web_search_results]

    print(f"document in websearch is thus; {documents}")
    web_search_count += 1

    return {
        "question": question,
        "documents": documents,
        "web_search_count": web_search_count
    }


# if __name__ == "__main__":
    # print(web_search_node({"question": "agent_memory", "documents": None}))

# print(fetch_from_web_search({"question": "what are the types of adversarial attack"}))

