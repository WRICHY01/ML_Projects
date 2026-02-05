from typing import TypedDict, Dict, Any

from pydantic import BaseModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableSequence

from ..state import AgentState

from ...llm import llm_model

# class QuestionSpewer(BaseModel):

SYSTEM_INSTRUCTIONS = """
                      Generate related question suggeestions to vague questions asked by the user
                      or tell the user to type in  a valid question if the question is gibberish or invalid

                      respond with a  polite response and a few related question to a vague response
                     """


chat_model = llm_model.chat_model

question_spewer_template = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_INSTRUCTIONS),
        ("human", "faulty_question:\n{question}")
    ]
)

question_spewer_chain: RunnableSequence = question_spewer_template | chat_model
    