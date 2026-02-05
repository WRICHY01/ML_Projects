from typing import TypedDict, Sequence, Annotated, List, Literal, Optional

from langchain_core.messages import BaseMessage, SystemMessage, AIMessage, HumanMessage, ToolMessage
from langgraph.graph.message import add_messages

class AgentState(TypedDict):
    """
    AgentState class Represents the state of the agent's graph schema,
    it acts as the central data structure that flows through every node in our graph workflow.

    Attributes:
        questions: user's input query
        generation: LLM's generation/response
        web search: boolean flag on whether to use the internet
        documents: list of documements(local, website)
    """
    # data_source
    question: Annotated[Sequence[BaseMessage], add_messages]
    generated_answer: str
    llm_knowledge: str
    web_search: bool
    documents: List[str]
    allow_external_sources: bool
    should_rewrite: bool
    should_rewrite_count: int
    web_search_count: int
    should_route_to_llm_or_web_count: int
    should_route_to_llm_or_web: bool
    knowledge_source: Optional[Literal["vector_store", "llm_knowledge", "web_search"]]
    routing_target: Optional[Literal["llm_knowledge", "web_search"]]
    is_relevant: str
    is_grounded: str
    hallucination_status: str
    generated_answer_grader_status: str
