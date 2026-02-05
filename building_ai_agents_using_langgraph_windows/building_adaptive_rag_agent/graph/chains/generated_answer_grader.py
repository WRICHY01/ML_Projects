from typing import Literal

from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableSequence

from ...llm import llm_model


class GeneratedAnswerGrader(BaseModel):
    """
    Binary score for assessment on generated response relevance
    """
    reasoning: str = Field(
        description="A concise justification of why the answer does or doesn't address the question"
    )
    binary_score: Literal["yes", "no"] = Field(
        description="'yes' if the answer addresses the question, 'no' if it does not."
    )
    

structured_gen_answer_grader = llm_model.chat_model.with_structured_output(GeneratedAnswerGrader)

system = """
         You are a expert grader, uncompromising about answer quality and relevance. 

         Your goal is to determine if the 'LLM Generation' adequately addresses the "User's question".
         Minor factual inaccuracies are handled separately - focus on whether the answer genuinely helps the user.

         Grading Criteria:
         - Usefulness: Does the answer meaningfully respond to what was asked?
         - Completeness: Does it address all parts of the user's question?
         - Relevance: Is the answer direct and free of unneccesary information?
         
         Score 'no' if:
         - The anser says "I don't know" or the context doesn't say"
         - The answer is mostly hallucinated or predominantly incorrect
         - The answer misses the core intent of the question

         Score 'yes' only if the answer provides a specificand relevant response that addresses the core intent of the question.

         Provide your reasoning first to justify your decision,then provide the binary score.
         """

generated_answer_template = ChatPromptTemplate(
    [
        ("system", system),
        ("human", "User question\n{question}\n\nLLM Generation\n{generated_answer}")
    ]
)

generated_answer_chain: RunnableSequence = generated_answer_template | structured_gen_answer_grader
