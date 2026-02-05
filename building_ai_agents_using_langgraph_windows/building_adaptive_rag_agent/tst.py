from typing import Literal

from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableSequence

from llm import llm_model

flag = False

class QuestionRouter(BaseModel):
    if flag:
        data_source: Literal["llm_knowledge", "web_search"] = Field(
            description="Given a user question, llm_knowledge for general tasks or web_search for real-time facts."
        )
    else:
        data_source: Literal["vector_store", "llm_knowledge", "web_search"] = Field(
        description="Given a user question, choose vector_store for internal docs, llm_knowledge for general tasks or web_search for real-time facts."
    )

if flag:
    system = """
    You are an expert at routing a user question, your goal is to send a question to the correct source:
    "llm_knowledge": use this ONLY for general logic, creative writing, coding_help, or explaining concepts that dont require external search  or specific internal documents.
    "web_search": use this ONLY when the user asks for factual, real-time, or current event information(e.g., "what is the price of Bitcoin?" or "who won the game last night?")
    """

else:
    system = """
    You are an expert at routing a user question, your goal is to send a question to the correct source:
    "vectorstore": use this ONLY for questions about agents, prompt engineering, and adversarial attacks.
    "llm_knowledge": use this ONLY for general logic, creative writing, coding_help, or explaining concepts that dont require external search  or specific internal documents.
    "web_search": use this ONLY when the user asks for factual, real-time, or current event information(e.g., "what is the price of Bitcoin?" or "who won the game last night?")
    """

question_router_template = ChatPromptTemplate(
    [
        ("system", system),
        ("human", "{question}")
    ]
)

llm = llm_model.llm_model.with_structured_output(QuestionRouter)

question_router: RunnableSequence = question_router_template | llm

print(flag)

# if __name__ == "__main__":

#     print(question_router)