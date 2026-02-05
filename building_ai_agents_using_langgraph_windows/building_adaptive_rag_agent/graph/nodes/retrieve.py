from typing import Dict, Any

from ..state import AgentState
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
    documents = state["documents"]

    retrieved_documents = retriever.invoke(question)
    if not retrieved_documents:
        raise ValueError("Internal Error, Vector_store response is empty!")
    print(f"documents content in the retrieved section is thus: {documents}")
    
    return {"question": question,
            "documents": retrieved_documents}



