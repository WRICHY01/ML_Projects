from typing import TypedDict, Sequence, Annotated, List, Literal, Optional, Any

from pydantic import BaseModel, Field
from langchain_core.messages import AnyMessage, BaseMessage, SystemMessage, AIMessage, HumanMessage, ToolMessage
from langchain_core.documents import Document
from langgraph.graph.message import add_messages


class SubQuery(BaseModel):
    """
    Represents a single atomic sub-question derived from the original user query.
    """
    reasoning: str = Field(
        description="Explanation of how this question was processed"
    )
    rephrased_question: str
    question_status: Literal["VALID", "AMBIGUOUS", "INVALID"]
    # should_rewrite: bool
    resolved_using_history: bool = Field(
        description="Whether conversational history was used to resolve this question"
    )

class StructuredQuery(BaseModel):
    """
    Canonical structured representation of the original user query.
    Includes decomposition, reformulation, and validation metadata.
    """
    sub_queries: list[SubQuery]

class RewriteQuery(BaseModel):
    question: str
    should_rewrite: bool

class SubQueryRouter(BaseModel):
    """
    Route a user query to the most relevant data source["vector_store", "llm_knowledge", "web_search"].
    """
    # reasoning: str = Field(
    #     description="Explanation of why this data source was selected"
    # )
    extracted_question: str = Field(
        description="question extracted from the rephrased_question in the StructuredQuery object"
    )
    data_source: Literal["vector_store", "llm_knowledge", "web_search"] = Field(
            description=(
                "The routing destination: "
                "'vector_store' for AI Agents, adversarial attacks, LLM security, and specialized domain documents;"
                "'llm_knowledge' for general explanation, reasoning, coding, and creative tasks;"
                "'web_search' for current information, recent events, and real-time data requiring verification;"
            )
        )

class QueryRouter(BaseModel):
    """List of routing decisions for multiple queries."""
    routes: list[SubQueryRouter]

def merge_documents(existing_docs: List, current_docs: List):
    merged_docs = existing_docs + current_docs
    return merged_docs

class IneligibleQuery(BaseModel):
    reasoning: str
    extracted_questions: str
    question_status: Literal["INVALID", "AMBIGUOUS"]
    retrieved_documents: None = None
    generated_answer: Optional[str]
    summarized_generated_answer: Optional[str]
    hallucination_status: Optional[Literal["yes", "no"]]
    generated_answer_grader_status: Optional[Literal["yes", "no"]]


class QueryContext(BaseModel):
    extracted_questions: str ########################The plural 's'
    retrieved_documents: list[Document]
    data_source: str = Field(default="llm_knowledge")
    vector_store_attempt_count: int = 0
    llm_knowledge_attempt_count: int = 0
    web_search_attempt_count: int = 0
    

class QueryContexts(BaseModel):
    query_contexts: list[QueryContext]


class RetrievalEvaluation(BaseModel):
    query_context: QueryContext = None #
    query_status: str = "VALID"
    is_relavant: Literal["yes", "no"] = "no" #
    routed_to_web_search: Optional[bool] = False
    should_rewrite: bool = True #
    generated_answer: Optional[str] = None #
    summarized_generated_answer: Annotated[list[str], merge_documents] = [] #
    hallucination_status: Optional[Literal["yes", "no"]] = None #
    generated_answer_grader_status: Optional[Literal["yes", "no"]] = None #


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
    questions: Annotated[Sequence[BaseMessage], add_messages]
    current_quesion: list[AnyMessage]
    structured_query: StructuredQuery
    query_router: QueryRouter
    ineligible_query_evaluations: list[IneligibleQuery]
    all_query_contexts: Annotated[QueryContexts, merge_documents]
    rewrite_queries: list[RewriteQuery]
    retrieval_evaluations: list[RetrievalEvaluation]
    is_initial_queries: bool
    final_generated_response: str
    llm_knowledge: str
    web_search: bool
    documents: List[Any]
    extracted_questions: Annotated[List[str], merge_documents]
    retrieved_documents: Annotated[List[str], merge_documents]
    is_initial_queries: bool
    allow_external_sources: bool
    should_rewrite: bool
    should_rewrite_count: int
    web_search_count: int
    should_route_to_llm_or_web_count: int
    should_route_to_web: bool
    knowledge_source: Optional[Literal["vector_store", "llm_knowledge", "web_search"]]
    routing_target: Optional[Literal["llm_knowledge", "web_search"]]
    is_relevant: str
    is_grounded: str
    vector_store_attempt_count: int
    llm_knowledge_attempt_count: int
    web_search_atttempt_count: int
    all_retrieved_query_evaluations = list[RetrievalEvaluation|IneligibleQuery]
    hallucination_status: str
    generated_answer_grader_status: str
    formatted_final_response: list[str]
