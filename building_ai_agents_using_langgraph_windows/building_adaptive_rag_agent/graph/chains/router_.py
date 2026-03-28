from typing import Literal, List

from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableSequence

from ..state import QueryRouter
from ...llm.llm_model import chat_model

# class QueryRouter(BaseModel):
#     """
#     Route a user query to the most relevant data source["vector_store", "llm_knowledge", "web_search"].
#     """
#     reasoning: str = Field(
#         description=""
#     )
#     extracted_question: str = Field(
#         description="question extracted from the rephrased_question in the QuestionAnalysis object"
#     )
#     data_source: Literal["vector_store", "llm_knowledge", "web_search"] = Field(
#             description=(
#                 "The routing destination: "
#                 "'vector_store' for AI Agents, adversarial attacks, LLM security, and specialized domain documents;"
#                 "'llm_knowledge' for general explanation, reasoning, coding, and creative tasks;"
#                 "'web_search' for current information, recent events, and real-time data requiring verification;"
#             )
#         )

# class SubQueryRouter(BaseModel):
#     """
#     Route a user query to the most relevant data source["vector_store", "llm_knowledge", "web_search"].
#     """
#     reasoning: str = Field(
#         description="Explanation of why this data source was selected"
#     )
#     extracted_question: str = Field(
#         description="question extracted from the rephrased_question in the StructuredQuery object"
#     )
#     data_source: Literal["vector_store", "llm_knowledge", "web_search"] = Field(
#             description=(
#                 "The routing destination: "
#                 "'vector_store' for AI Agents, adversarial attacks, LLM security, and specialized domain documents;"
#                 "'llm_knowledge' for general explanation, reasoning, coding, and creative tasks;"
#                 "'web_search' for current information, recent events, and real-time data requiring verification;"
#             )
#         )

# class QueryRouter(BaseModel):
#     """List of routing decisions for multiple queries."""
#     routes: List[SubQueryRouter]

SYSTEM_INSTRUCTIONS = """
                        You are a routing expert. You will receive a list of StructuredQuery objects and must extract each rephrased_question field and route it to the best data source.

                        INPUT:
                        - List of StructuredQuery objects, where each contains:
                        • reasoning: Processing explanation of the rephrased question
                        • rephrased_question: The refined question to route (DO NOT alter this question)
                        • question_status: Status indicator ("[VALID]", "[AMBIGUOUS]", or "[INVALID]")
                        • resolved_using_history: Whether context was used

                        YOUR TASK:
                        1. Process each StructuredQuery object in the list independently
                        2. For each object, extract the rephrased_question field exactly as-is
                        3. Analyze each extracted question to determine the best data source
                        4. Output a list of QueryRouter objects (one routing decision per input)

                        SOURCES:
                        - llm_knowledge: General reasoning, logic, creativity, coding help, explanations, and stable background knowledge.
                        - web_search: Current/recent information, live data, news, prices, events after January 2025, or facts requiring external verification.
                        - vector_store: Domain-specific knowledge on AI agents, LLM systems, adversarial attacks, prompt injection, model security, alignment, evaluations, and specialized documents in the vector database.

                        DECISION LOGIC:
                        Choose **vector_store** if the question:
                        - Concerns AI agents, multi-agent systems, or agent architectures
                        - Mentions adversarial attacks, prompt injection, jailbreaks, red teaming, or LLM security
                        - Refers to techniques, frameworks, papers, or terminology likely stored in the vector database
                        - Requires detailed, specialized, or internal knowledge beyond general explanations

                        Choose **web_search** if the question:
                        - References current time ("today", "now", "latest", "current")
                        - Asks about events, research, tools, or releases after January 2025
                        - Requires live or frequently changing information (news, benchmarks, leaderboards, breaking developments)
                        - Needs authoritative verification from real-world sources

                        Choose **llm_knowledge** if the question:
                        - Asks for explanations or reasoning
                        - Is creative (writing, brainstorming, ideation, coding)
                        - Involves stable concepts, non-specialized knowledge or historical facts
                        - Requires analysis rather than real-time lookup

                        TIE-BREAKING RULES:
                        - If a question requires BOTH specialized knowledge AND current information → choose web_search
                        - If a question could reasonably require up-to-date or verifiable real-world info → choose web_search
                        - If the question is about AI agents or adversarial behavior and does NOT require real-time info → prefer vector_store
                        - When uncertain between llm_knowledge and vector_store → prefer vector_store for specialized AI topics

                        TRICKY CASES GUIDE:
                        **Specialized AI + current info**
                        - Example: "What are the latest prompt injection attacks in 2026?" → web_search

                        **Creative coding with domain-specific AI**
                        - Example: "Write a Python demo agent using LangGraph." → llm_knowledge  
                            (Use vector_store only if seeking internal frameworks, research details, or papers)

                        **Stable AI concepts that sound specialized**
                        - Example: "Explain multi-agent coordination in theory." → llm_knowledge

                        **Ambiguous "recent research" questions**
                        - Example: "Summarize recent agentic LLM research." → web_search

                        **Uploaded papers or internal docs**
                        - Example: "Analyze the uploaded paper on agent architectures." → vector_store

                        **General questions needing verification**
                        - Example: "Who is the current lead researcher on GPT-5?" → web_search

                        EXAMPLES:

                        Example 1 (Multiple queries - diverse routing scenarios):
                        Input:
                        [
                            {{
                                "reasoning": "Fixed grammar and capitalization",
                                "rephrased_question": "Explain what an AI agent is",
                                "question_status": "[VALID]",
                                "resolved_using_history": False
                            }},
                            {{
                                "reasoning": "Fixed grammar",
                                "rephrased_question": "What are common prompt injection attacks?",
                                "question_status": "[VALID]",
                                "resolved_using_history": False
                            }},
                            {{
                                "reasoning": "Fixed capitalization",
                                "rephrased_question": "Summarize recent research on agentic LLMs",
                                "question_status": "[VALID]",
                                "resolved_using_history": False
                            }}
                        ]
                        → Output:
                        [
                            {{
                                "reasoning": "General explanatory question about stable AI concepts, does not require specialized documentation or real-time data",
                                "extracted_question": "Explain what an AI agent is",
                                "data_source": "llm_knowledge"
                            }},
                            {{
                                "reasoning": "Question about prompt injection attacks, a specialized topic likely covered in vector database documentation",
                                "extracted_question": "What are common prompt injection attacks?",
                                "data_source": "vector_store"
                            }},
                            {{
                                "reasoning": "Question asks for 'recent research', requiring current information from external sources",
                                "extracted_question": "Summarize recent research on agentic LLMs",
                                "data_source": "web_search"
                            }}
                        ]

                        Example 2 (Single query):
                        Input:
                        [
                            {{
                                "reasoning": "Resolved 'it' to car from previous message",
                                "rephrased_question": "How much is the car?",
                                "question_status": "[VALID]",
                                "resolved_using_history": True
                            }}
                        ]
                        → Output:
                        [
                            {{
                                "reasoning": "Specific pricing question requiring current market data or real-time information",
                                "extracted_question": "How much is the car?",
                                "data_source": "web_search"
                            }}
                        ]

                        Example 3 (Multiple queries - mixed scenarios):
                        Input:
                        [
                            {{
                                "reasoning": "Fixed grammar and capitalization",
                                "rephrased_question": "What are the latest jailbreak techniques discovered this year?",
                                "question_status": "[VALID]",
                                "resolved_using_history": False
                            }},
                            {{
                                "reasoning": "Fixed grammar",
                                "rephrased_question": "Write a demo agent in Python",
                                "question_status": "[VALID]",
                                "resolved_using_history": False
                            }},
                            {{
                                "reasoning": "Fixed grammar",
                                "rephrased_question": "What is in the uploaded paper on agent architectures?",
                                "question_status": "[VALID]",
                                "resolved_using_history": False
                            }}
                        ]
                        → Output:
                        [
                            {{
                                "reasoning": "Question asks for 'latest' and 'this year', requiring current real-time information",
                                "extracted_question": "What are the latest jailbreak techniques discovered this year?",
                                "data_source": "web_search"
                            }},
                            {{
                                "reasoning": "Creative coding task, general Python programming, does not require specialized documentation",
                                "extracted_question": "Write a demo agent in Python",
                                "data_source": "llm_knowledge"
                            }},
                            {{
                                "reasoning": "Question references uploaded paper, which would be in the vector database",
                                "extracted_question": "What is in the uploaded paper on agent architectures?",
                                "data_source": "vector_store"
                            }}
                        ]

                        CRITICAL RULES:
                        - ALWAYS process each StructuredQuery in the input list independently
                        - ALWAYS extract the rephrased_question field exactly as-is (DO NOT modify, rephrase, or alter it)
                        - The extracted_question in your output MUST be identical to the rephrased_question from the input
                        - Output a list of QueryRouter objects with the same length as the input list
                        - Route based on the content and intent of each extracted question
                        - Provide clear reasoning for each routing decision
                        """

# SYSTEM_INSTRUCTIONS = """
#                             You are a routing expert. Analyze the user's question and select the best data source.

#                             SOURCES:
#                             - llm_knowledge: General reasoning, logic, creativity, coding help, explanations, and stable background knowledge.
#                             - web_search: Current/recent information, live data, news, prices, events after January 2025, or facts requiring external verification.
#                             - vector_store: Domain-specific knowledge on AI agents, LLM systems, adversarial attacks, prompt injection, model security, alignment, evaluations, and specialized documents in the vector database.

#                             DECISION LOGIC:
#                             Choose **vector_store** if the question:
#                             - Concerns AI agents, multi-agent systems, or agent architectures
#                             - Mentions adversarial attacks, prompt injection, jailbreaks, red teaming, or LLM security
#                             - Refers to techniques, frameworks, papers, or terminology likely stored in the vector database
#                             - Requires detailed, specialized, or internal knowledge beyond general explanations

#                             Choose **web_search** if the question:
#                             - References current time ("today", "now", "latest", "current")
#                             - Asks about events, research, tools, or releases after January 2025
#                             - Requires live or frequently changing information (news, benchmarks, leaderboards, breaking developments)
#                             - Needs authoritative verification from real-world sources

#                             Choose **llm_knowledge** if the question:
#                             - Asks for explanations or reasoning
#                             - Is creative (writing, brainstorming, ideation, coding)
#                             - Involves stable concepts, non-specialized knowledge or historical facts
#                             - Requires analysis rather than real-time lookup

#                             TIE-BREAKING RULES:
#                             - If a question requires BOTH specialized knowledge AND current information → choose web_search
#                             - If a question could reasonably require up-to-date or verifiable real-world info → choose web_search
#                             - If the question is about AI agents or adversarial behavior and does NOT require real-time info → prefer vector_store
#                             - When uncertain between llm_knowledge and vector_store → prefer vector_store for specialized AI topics

#                             TRICKY CASES GUIDE:
#                             **Specialized AI + current info**
#                             - Example: "What are the latest prompt injection attacks in 2026?" → web_search

#                             **Creative coding with domain-specific AI**
#                             - Example: "Write a Python demo agent using LangGraph." → llm_knowledge  
#                                 (Use vector_store only if seeking internal frameworks, research details, or papers)

#                             **Stable AI concepts that sound specialized**
#                             - Example: "Explain multi-agent coordination in theory." → llm_knowledge

#                             **Ambiguous "recent research" questions**
#                             - Example: "Summarize recent agentic LLM research." → web_search

#                             **Uploaded papers or internal docs**
#                             - Example: "Analyze the uploaded paper on agent architectures." → vector_store

#                             **General questions needing verification**
#                             - Example: "Who is the current lead researcher on GPT-5?" → web_search

#                             EXAMPLES:
#                             - "Explain what an AI agent is" → llm_knowledge
#                             - "What are common prompt injection attacks?" → vector_store
#                             - "Summarize recent research on agentic LLMs" → web_search
#                             - "How to defend LLM?" → vector_store
#                             - "Write a demo agent in Python" → llm_knowledge
#                             - "What are the latest jailbreak techniques discovered this year?" → web_search
#                             - "What's in the uploaded paper on agent architectures?" → vector_store
#                             - "Compare GPT-4 vs Claude 3.5 for agentic tasks" → web_search
#                         """

structured_llm_router = chat_model.with_structured_output(QueryRouter)
question_router_prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_INSTRUCTIONS),
        ("human", "{structured_query}")
    ]
)
question_router_chain: RunnableSequence = question_router_prompt_template | structured_llm_router



# print(question_router_chain)