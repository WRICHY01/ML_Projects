from pathlib import Path
from typing import Dict, Any, Literal

from langgraph.graph import END, StateGraph
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv

from .state import AgentState
from .chains import (router,
                    doc_retrieval_grader,
                    generated_answer_grader,
                    hallucination_grader,
                    answer_generation)
from .nodes import (question_rephraser,
                    question_router,
                    retrieve,
                    llm_knowledge,
                    web_search,
                    grade_documents,
                    generate,
                    answer_relevance,
                    hallucination_checker)
from ..llm import llm_model
from ..graph_viewer import view_mermaid_graph

load_dotenv()

file_path = Path(__file__)

def route_request(state: AgentState) -> Literal["vector_store", "llm_knowledge", "web_search"]:
    """
    Determines the initial node for the  workflow based on the user's intent.

    Args:
        state(AgentState): The current state of the graph, containing the route destination.
    Returns:
        str: A routing decision key. Returns:
            'vector_store' to proceed to the 'retriever_node',
            'llm_knowledge' to proceed to the 'llm_knowledge_node', 
            'web_search' to proceed to the 'web_search_node'. 
    """

    knowledge_source = state["knowledge_source"]

    if knowledge_source == "vector_store":
        print("---Route Question to RAG---")
        return "vector_store"

    elif knowledge_source == "llm_knowledge":
        print("---Route Question to LLM--")
        return "llm_knowledge"
        
    # else:
    #     print("---Route Question to the Internet---")
    #     return "web_search"

    elif knowledge_source == "web_search":
        print("---Route Question to the Internet")
        return  "web_search"

    else:
        print("---Route to the question suggester node---")
        return "undetermined"


def check_doc_retrieval_relevance(state: AgentState) -> str:
    """
    Determines whether the retrieved documents are relevant to the question.

    Args:
        state(AgentState): The current graph state containing 'is_relevant'
        and 'should_route_to_llm_or_web'
    
    Returns:
        str: A routing decision key. Returns:
            'route_to_llm_knowledge_or_web' to proceed to 'llm_knowledge' or 'web_search node',
            'relevant' to proceed to 'generate node', 
            'irrelevant' to trigger a retry/refinement by going back to 'question_rephraser_node'.
    """
    is_relevant = state["is_relevant"]
    should_rewrite = state["should_rewrite"]
    should_rewrite_count = state["should_rewrite_count"]
    should_route_to_llm_or_web_count = state["should_route_to_llm_or_web_count"]

    if should_rewrite and should_rewrite_count < 2:
        print("---Route back to Question Rephraser--")
        return "irrelevant"
    
    return "relevant" if is_relevant == "yes" else "route_to_llm_knowledge_or_web_search"

def should_route_to_llm_knowledge_or_web_search(state: AgentState)  -> str:
    """
    Args:
        state(AgentState): 
    Retr
    """
    
    should_route_to_llm_or_web = state["should_route_to_llm_or_web"]
    should_route_to_llm_or_web_count = state["should_route_to_llm_or_web_count"]
    # routing_target = state["routing_target"]
        

    if should_route_to_llm_or_web:
        # should_route_to_llm_or_web_count += 1
        if should_route_to_llm_or_web_count < 2:
            return "llm_knowledge"
        elif 2 <= should_route_to_llm_or_web_count < 3:
            return "web_search"
        else:
            return "no_of_resources_exhausted"
    else:
        raise RuntimeError("Internal Error, Invalid route target!")

def decide_to_finish(state: AgentState) -> "str":
    """
    Determines whether the generated answer is grounded in documents and relevant to the question.
    """
    hallucination_status = state["hallucination_status"]
    answer_grader_status = state["generated_answer_grader_status"]
    
    if hallucination_status == "yes" and answer_grader_status == "yes":
        print("---DECISION: ANSWER IS GROUNDED AND RELEVANT---")
        return "passed"
    else:
        print("---DECISION: ANSWER IS HALLUCINATION OR IRRELEVANT---")
        return "failed"


agent = StateGraph(AgentState)
agent.add_node("question_rephraser_node", question_rephraser.question_transformer)
agent.add_node("question_router_node", question_router.route_question)
# agent.add_node("query_clarification_node", )
agent.add_node("retriever_node", retrieve.retrieve_from_vectorstore)
agent.add_node("llm_knowledge_node", llm_knowledge.generate_using_llm_knowledge)
agent.add_node("web_search_node", web_search.fetch_from_web_search)
agent.add_node("pass_as_is_node1", lambda state: state)
agent.add_node("pass_as_is_node2", lambda state: state)
agent.add_node("document_grader_node", grade_documents.documents_grader)
agent.add_node("generate_node", generate.generate_response)
agent.add_node("hallucination_checker_node", hallucination_checker.check_hallucination)
agent.add_node("generated_answer_checker_node", answer_relevance.check_answer_relevance)

agent.set_entry_point("question_rephraser_node")
agent.add_edge("question_rephraser_node", "question_router_node")
agent.add_conditional_edges(
    "question_router_node",
    route_request,
    {
        "vector_store": "retriever_node",
        "llm_knowledge": "llm_knowledge_node",
        "web_search": "web_search_node",
        "undetermined": "query_clarification_node" ##################Not constructed a node for it yet#########################
    }
)
agent.add_edge("retriever_node", "document_grader_node")
agent.add_edge("llm_knowledge_node", "document_grader_node")
agent.add_edge("web_search_node", "document_grader_node")
agent.add_conditional_edges(
    "document_grader_node",
    check_doc_retrieval_relevance, 
    {   
        # "relevant": "pass_as_is_node", Uncomment and commment the immediate 2 if not better
        "irrelevant": "question_rephraser_node",
        "relevant": "generate_node",
        "route_to_llm_knowledge_or_web_search": "pass_as_is_node1"
    }  
)
# agent.add_conditional_edges(  Uncomment this if not better
#     "pass_as_is_node",
#     should_generate,
#     {
#         "generate": "generate_node",
#         "route_llm_knowledge_or_web_search": "pass_as_is_node"
#     }
# )
agent.add_conditional_edges(
    "pass_as_is_node1",
    should_route_to_llm_knowledge_or_web_search,
        {
            "llm_knowledge": "llm_knowledge_node",
            "web_search": "web_search_node",
            "no_of_resources_exhausted": END
        }
)
agent.add_edge("generate_node", "hallucination_checker_node")
agent.add_edge("generate_node", "generated_answer_checker_node")
agent.add_edge("hallucination_checker_node", "pass_as_is_node2")
agent.add_edge("generated_answer_checker_node", "pass_as_is_node2")
agent.add_conditional_edges(
    "pass_as_is_node2",
    decide_to_finish,
    {
        "passed": END,
        "failed": "web_search_node" # work on the no of times in web search node...
    }
)


app = agent.compile()
# mermaid_text = app.get_graph().draw_mermaid()
# view_mermaid_graph(mermaid_text, file_path.parent/"graph_tst.md")

# app.invoke({"question": "what is agent ai?"})
# print(check_hallucination({"documents": "dr martins has range of laboratory equipments like pippetes, microscope, volumentary flask",
# "generated_answer": "dr martins has pippetes, microscope, volumentary flask in his laboratory"}))

# print(check_answer_relevance({"question": "what does dr martins do?",
# "generated_answer": "dr martins is a pedeatrician"}))

# varsapp
history = []
while True:
    user_input = str(input("enter question:\n"))
    if user_input.lower() in ["exit", "quit"]:
        break

    
    history.append(HumanMessage(content=user_input))
    print("*" * 300)
    print("history before being invoked is", history)
    print("#" * 300)
    result = app.invoke({"question": history})
    history = result["question"]

    print("history contains: ", history)
    print("all history contains", result)
# result_length = len(result["messages"])
# if  history and result_length > 0:
#     if "CONVERSATION_END" == str(result["messages"][-1].content):
#         break
# user_input = input("\nWhat is your question: ")
# if user_input.lower() in ["exit", "quit"]:
#     break
# # history.append(user_input)
# history.append(HumanMessage(content=user_input))
# # print("new history looks like this: ", history)
# result = app.invoke({"messages": history})
# print("result outcome be like: ", result)
# history = result["messages"]

# print("\n=== ANSWER ===")
# # print(result["messages"][-1].content)
# print(history[-1].content)