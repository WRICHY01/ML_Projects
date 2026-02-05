from typing import Dict, Any

from ..state import AgentState
from ..chains.answer_generation import answer_generation_chain


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
    
    question = state["question"]
    print(f"question in generate node is: {question}")
    documents = state["documents"]

    generation  = answer_generation_chain.invoke(
                    {
                        "question": question,
                        "context": documents,
                    }
                )
    
    return {
        "question": question, 
        "documents": documents, 
        "generated_answer": generation
        }