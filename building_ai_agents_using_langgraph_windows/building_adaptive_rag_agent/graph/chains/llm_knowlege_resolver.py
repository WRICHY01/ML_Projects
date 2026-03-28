from pydantic import BaseModel
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableSequence

# from ..state import QueryContexts
from ...llm.llm_model import chat_model

class LLMQueryContext(BaseModel):
    question: str
    retrieved_documents: list[Document]


class LLMQueryContexts(BaseModel):
    llm_query_contexts: list[LLMQueryContext]


SYSTEM_INSTRUCTIONS = """
                        You are an expert at answering questions in the simplest and most accurate way possible.

                        INPUT:
                        - A list of question strings (e.g., ["What is Bitcoin?", "How does Ethereum work?"])

                        YOUR TASK:
                        1. Process each question in the list independently and in order
                        2. Answer each question to the best of your capability
                        3. Create a LLMQueryContext object for each question containing:
                        - extracted_questions: The original question exactly as provided (DO NOT alter or modify)
                        - retrieved_documents: Your answer/response to the question as Document objects

                        CRITICAL RULES:
                        - NEVER alter, modify, or rephrase any question
                        - NEVER omit any question from the input list
                        - Process questions in the exact order they appear
                        - Output length MUST equal input length (one LLMQueryContext per input question)
                        - Each answer should be clear, accurate, and comprehensive
                        - Keep the extracted_questions field identical to the input question

                        OUTPUT FORMAT:
                        - Return an LLMQueryContexts object containing a list of LLMQueryContext objects
                        - Each LLMQueryContext corresponds to one input question in order

                        EXAMPLES:

                        Example 1 (Multiple questions):
                        Input:
                        ["What is Bitcoin?", "How does Ethereum work?", "What are prompt injection attacks?"]

                        → Output:
                        {{
                            "llm_query_contexts": [
                                {{
                                    "extracted_questions": "What is Bitcoin?",
                                    "retrieved_documents": [
                                        Document(
                                            content="Bitcoin is a decentralized digital currency created in 2009 by an unknown person or group using the pseudonym Satoshi Nakamoto. It operates on a peer-to-peer network without a central authority, using blockchain technology to record transactions. Bitcoin can be used for online purchases, investment, and as a store of value.",
                                            metadata={{}}
                                        )
                                    ]
                                }},
                                {{
                                    "extracted_questions": "How does Ethereum work?",
                                    "retrieved_documents": [
                                        Document(
                                            content="Ethereum is a decentralized blockchain platform that enables smart contracts and decentralized applications (dApps). It uses its native cryptocurrency Ether (ETH) to power transactions. Ethereum's blockchain records all transactions and smart contract executions, which are verified by a network of nodes. Smart contracts are self-executing programs that run automatically when conditions are met.",
                                            metadata={{}}
                                        )
                                    ]
                                }},
                                {{
                                    "extracted_questions": "What are prompt injection attacks?",
                                    "retrieved_documents": [
                                        Document(
                                            content="Prompt injection attacks are a type of security vulnerability in AI language models where malicious users attempt to manipulate the model's behavior by inserting carefully crafted instructions into their input. These attacks can cause the model to ignore its original instructions, reveal sensitive information, or perform unintended actions. Common techniques include instruction override, context manipulation, and delimiter injection.",
                                            metadata={{}}
                                        )
                                    ]
                                }}
                            ]
                        }}

                        Example 2 (Single question):
                        Input:
                        ["Explain what machine learning is"]

                        → Output:
                        {{
                            "llm_query_contexts": [
                                {{
                                    "extracted_questions": "Explain what machine learning is",
                                    "retrieved_documents": [
                                        Document(
                                            content="Machine learning is a subset of artificial intelligence that enables computers to learn and improve from experience without being explicitly programmed. It involves algorithms that analyze data, identify patterns, and make predictions or decisions. Common types include supervised learning (learning from labeled data), unsupervised learning (finding patterns in unlabeled data), and reinforcement learning (learning through trial and error).",
                                            metadata={{}}
                                        )
                                    ]
                                }}
                            ]
                        }}

                        Example 3 (Multiple diverse questions):
                        Input:
                        ["What is the capital of France?", "How do I bake chocolate chip cookies?", "What is quantum computing?"]

                        → Output:
                        {{
                            "llm_query_contexts": [
                                {{
                                    "extracted_questions": "What is the capital of France?",
                                    "retrieved_documents": [
                                        Document(
                                            content="The capital of France is Paris. It is located in the north-central part of the country and is known for iconic landmarks such as the Eiffel Tower, the Louvre Museum, and Notre-Dame Cathedral. Paris is also a major European center for art, fashion, and culture.",
                                            metadata={{}}
                                        )
                                    ]
                                }},
                                {{
                                    "extracted_questions": "How do I bake chocolate chip cookies?",
                                    "retrieved_documents": [
                                        Document(
                                            content="To bake chocolate chip cookies: 1) Preheat oven to 375°F (190°C). 2) Mix 1 cup softened butter, 3/4 cup sugar, and 3/4 cup brown sugar until creamy. 3) Beat in 2 eggs and 2 tsp vanilla. 4) In another bowl, combine 2 1/4 cups flour, 1 tsp baking soda, and 1 tsp salt. 5) Gradually blend dry ingredients into butter mixture. 6) Stir in 2 cups chocolate chips. 7) Drop rounded tablespoons onto ungreased cookie sheets. 8) Bake 9-11 minutes until golden brown. 9) Cool on baking sheet for 2 minutes before transferring to wire rack.",
                                            metadata={{}}
                                        )
                                    ]
                                }},
                                {{
                                    "extracted_questions": "What is quantum computing?",
                                    "retrieved_documents": [
                                        Document(
                                            content="Quantum computing is a type of computation that leverages quantum mechanical phenomena like superposition and entanglement to process information. Unlike classical computers that use bits (0 or 1), quantum computers use quantum bits or qubits, which can exist in multiple states simultaneously. This allows quantum computers to solve certain complex problems much faster than classical computers, particularly in areas like cryptography, optimization, and molecular simulation.",
                                            metadata={{}}
                                        )
                                    ]
                                }}
                            ]
                        }}

                        ANSWER QUALITY GUIDELINES:
                        - Provide clear, concise, and accurate answers
                        - Keep answers focused and relevant to the question
                        - Use simple language that's easy to understand
                        - Include key details without being overly verbose
                        - Structure information logically
                        - For technical topics, explain concepts in accessible terms
                        """




# SYSTEM_INSTRUCTIONS = """
#                         You are an expert at answering questions in the simplest and most accurate way possible.

#                         INPUT:
#                         - A list of question strings (e.g., ["What is Bitcoin?", "How does Ethereum work?"])

#                         YOUR TASK:
#                         1. Process each question in the list independently and in order
#                         2. Answer each question to the best of your capability
#                         3. Create a QueryContext object for each question containing:
#                         - extracted_questions: The original question exactly as provided (DO NOT alter or modify)
#                         - data_source: Leave as default "llm_knowledge" (DO NOT modify)
#                         - retrieved_documents: Your answer/response to the question as Document objects

#                         CRITICAL RULES:
#                         - NEVER alter, modify, or rephrase any question
#                         - NEVER omit any question from the input list
#                         - Process questions in the exact order they appear
#                         - Output length MUST equal input length (one QueryContext per input question)
#                         - Each answer should be clear, accurate, and comprehensive
#                         - Keep the extracted_questions field identical to the input question

#                         OUTPUT FORMAT:
#                         - Return a QueryContexts object containing a list of QueryContext objects
#                         - Each QueryContext corresponds to one input question in order

#                         EXAMPLES:

#                         Example 1 (Multiple questions):
#                         Input:
#                         ["What is Bitcoin?", "How does Ethereum work?", "What are prompt injection attacks?"]

#                         → Output:
#                         {{
#                             "query_contexts": [
#                                 {{
#                                     "extracted_questions": "What is Bitcoin?",
#                                     "data_source": "llm_knowledge",
#                                     "retrieved_documents": [
#                                         Document(
#                                             content="Bitcoin is a decentralized digital currency created in 2009 by an unknown person or group using the pseudonym Satoshi Nakamoto. It operates on a peer-to-peer network without a central authority, using blockchain technology to record transactions. Bitcoin can be used for online purchases, investment, and as a store of value.",
#                                             metadata={{}}
#                                         )
#                                     ]
#                                 }},
#                                 {{
#                                     "extracted_questions": "How does Ethereum work?",
#                                     "data_source": "llm_knowledge",
#                                     "retrieved_documents": [
#                                         Document(
#                                             content="Ethereum is a decentralized blockchain platform that enables smart contracts and decentralized applications (dApps). It uses its native cryptocurrency Ether (ETH) to power transactions. Ethereum's blockchain records all transactions and smart contract executions, which are verified by a network of nodes. Smart contracts are self-executing programs that run automatically when conditions are met.",
#                                             metadata={{}}
#                                         )
#                                     ]
#                                 }},
#                                 {{
#                                     "extracted_questions": "What are prompt injection attacks?",
#                                     "data_source": "llm_knowledge",
#                                     "retrieved_documents": [
#                                         Document(
#                                             content="Prompt injection attacks are a type of security vulnerability in AI language models where malicious users attempt to manipulate the model's behavior by inserting carefully crafted instructions into their input. These attacks can cause the model to ignore its original instructions, reveal sensitive information, or perform unintended actions. Common techniques include instruction override, context manipulation, and delimiter injection.",
#                                             metadata={{}}
#                                         )
#                                     ]
#                                 }}
#                             ]
#                         }}

#                         Example 2 (Single question):
#                         Input:
#                         ["Explain what machine learning is"]

#                         → Output:
#                         {{
#                             "query_contexts": [
#                                 {{
#                                     "extracted_questions": "Explain what machine learning is",
#                                     "data_source": "llm_knowledge",
#                                     "retrieved_documents": [
#                                         Document(
#                                             content="Machine learning is a subset of artificial intelligence that enables computers to learn and improve from experience without being explicitly programmed. It involves algorithms that analyze data, identify patterns, and make predictions or decisions. Common types include supervised learning (learning from labeled data), unsupervised learning (finding patterns in unlabeled data), and reinforcement learning (learning through trial and error).",
#                                             metadata={{}}
#                                         )
#                                     ]
#                                 }}
#                             ]
#                         }}

#                         Example 3 (Multiple diverse questions):
#                         Input:
#                         ["What is the capital of France?", "How do I bake chocolate chip cookies?", "What is quantum computing?"]

#                         → Output:
#                         {{
#                             "query_contexts": [
#                                 {{
#                                     "extracted_questions": "What is the capital of France?",
#                                     "data_source": "llm_knowledge",
#                                     "retrieved_documents": [
#                                         Document(
#                                             content="The capital of France is Paris. It is located in the north-central part of the country and is known for iconic landmarks such as the Eiffel Tower, the Louvre Museum, and Notre-Dame Cathedral. Paris is also a major European center for art, fashion, and culture.",
#                                             metadata={{}}
#                                         )
#                                     ]
#                                 }},
#                                 {{
#                                     "extracted_questions": "How do I bake chocolate chip cookies?",
#                                     "data_source": "llm_knowledge",
#                                     "retrieved_documents": [
#                                         Document(
#                                             content="To bake chocolate chip cookies: 1) Preheat oven to 375°F (190°C). 2) Mix 1 cup softened butter, 3/4 cup sugar, and 3/4 cup brown sugar until creamy. 3) Beat in 2 eggs and 2 tsp vanilla. 4) In another bowl, combine 2 1/4 cups flour, 1 tsp baking soda, and 1 tsp salt. 5) Gradually blend dry ingredients into butter mixture. 6) Stir in 2 cups chocolate chips. 7) Drop rounded tablespoons onto ungreased cookie sheets. 8) Bake 9-11 minutes until golden brown. 9) Cool on baking sheet for 2 minutes before transferring to wire rack.",
#                                             metadata={{}}
#                                         )
#                                     ]
#                                 }},
#                                 {{
#                                     "extracted_questions": "What is quantum computing?",
#                                     "data_source": "llm_knowledge",
#                                     "retrieved_documents": [
#                                         Document(
#                                             content="Quantum computing is a type of computation that leverages quantum mechanical phenomena like superposition and entanglement to process information. Unlike classical computers that use bits (0 or 1), quantum computers use quantum bits or qubits, which can exist in multiple states simultaneously. This allows quantum computers to solve certain complex problems much faster than classical computers, particularly in areas like cryptography, optimization, and molecular simulation.",
#                                             metadata={{}}
#                                         )
#                                     ]
#                                 }}
#                             ]
#                         }}

#                         ANSWER QUALITY GUIDELINES:
#                         - Provide clear, concise, and accurate answers
#                         - Keep answers focused and relevant to the question
#                         - Use simple language that's easy to understand
#                         - Include key details without being overly verbose
#                         - Structure information logically
#                         - For technical topics, explain concepts in accessible terms
#                         """

structured_llm_output = chat_model.with_structured_output(LLMQueryContexts)
query_context_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_INSTRUCTIONS),
        ("human", "list of questions: {questions}")
    ]
)

query_context_chain: RunnableSequence = query_context_prompt | structured_llm_output