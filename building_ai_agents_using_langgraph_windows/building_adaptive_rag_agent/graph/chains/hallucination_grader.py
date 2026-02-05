from typing import Literal

from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableSequence

from ...llm import llm_model

print("Entering the hallucination source code")
class GradeHallucinations(BaseModel):
    """
    Binary score for hallucination check on generated answers.
    """
    reason: str = Field(
        description="Step-by-step reasoning for whether the answer is grounded in the facts."
        )
    binary_score: Literal["yes", "no"] = Field(
        description="'yes' if the answer is grounded in the facts, 'no' if it is not."
        )
    


structured_llm_hallucination_grader = llm_model.chat_model.with_structured_output(GradeHallucinations)

system = """
        configuration:
        - source: {knowledge_source}
        You are a expert grader assessing whether an LLM generation is grounded in/supported by a set of retrieved facts

        Scoring Criteria:
        - 'yes': if 80-90% or more of the answer is supported by the provided facts and coherent.
        - 'no': Less than 80% is supported, or the answer contains unsupported claims or contradictions.

        Source Types:
        - vector_store: Internal knowledge base
        - llm_knowledge: LLM's training knowledge
        - web_search: Internet search results

        Special Instructions for web_search and vector_store:
        Ignore navigational elements, 'Call-to-action phrases (e.g., 'click here',
        'check out these coins'), and web-formatting artifacts when verifying facts.
        
        Provide your reasoning first, then give a binary score of 'yes' or 'no'.
        """
hallucination_grader_prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "knowledge_source: {knowledge_source}\n\nLLM Generation:\n{generated_answer}\n\nGround Truth:\n{documents}")
    ]
)

hallucination_chain: RunnableSequence = hallucination_grader_prompt_template | structured_llm_hallucination_grader

print("Done with processing hallucination source code")

# print(hallucination_grader)