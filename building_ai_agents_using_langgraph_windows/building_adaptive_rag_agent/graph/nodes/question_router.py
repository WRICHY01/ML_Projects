from typing import Any, Dict

from langchain_core.messages import AIMessage

from ..state import AgentState
# from ...llm import llm_model
from ..chains.router_ import question_router_chain

def route_question(state: AgentState) -> Dict[str, Any]:
    """
    
    """
    
    question = state["question"]
    print(f"question in question_router node is: {question}")
    should_rewrite = state["should_rewrite"]
    # knowledge_source = state.get("knowledge_source") ##################### need to remove this later on...
    routing_target = state.get("routing_target") ##################### need to remove this later on...
    last_rephrased_question = question[-1]
    print("last_rephrased_question looks like: ", last_rephrased_question)

    if isinstance(last_rephrased_question, AIMessage) \
    and last_rephrased_question.content in ["[AMBIGUOUS]", "[INVALID]"
                                            "[ambiguous]", "[invalid]"]:
        return {"knowledge_source": knowledge_source}
    # chat_model = llm_model.chat_model
    # query_router_chain = router.get_route_chain(chat_model, should_rewrite)
    query_router = question_router_chain.invoke({"question": question})
    
    knowledge_source = query_router.data_source

    

    # if should_rewrite: ################## and this ....
    #     routing_target = query_router.data_source ################### with this too.......

    return {"knowledge_source": knowledge_source,
            "routing_target": routing_target}