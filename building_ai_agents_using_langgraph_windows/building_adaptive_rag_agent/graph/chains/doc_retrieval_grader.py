from typing import Literal

from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableSequence

from ..state import QueryContext
from ...llm import llm_model


class GradeDocument(BaseModel):
    """
    Binary score for relevance check on retreived documents.
    """
    query_context: QueryContext
    binary_score: Literal["yes", "no"] = Field(
        description="Relevance score: 'yes' if the document is relavant to the user question, 'no' if it is not."
    )


class GradeDocuments(BaseModel):
    graded_documents: list[GradeDocument]



system = """
         You are a grader assessing relevance of a retreived document to a user question.
         Strictly follow these rules:
         grade 'yes', if the document contains keywords or semantic meaning related to the user question.
         grade 'no', if the document is irrelevant or does not contain information that helps answer the question.
         Do not be overly strict; the goal is to filter out completely irrelevant noise.
        """

structured_llm_doc_grader = llm_model.chat_model.with_structured_output(GradeDocuments)
doc_grader_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "User query:\n{question}\n\nRetreived_Document:\n{documents}")
    ]
)

doc_grader: RunnableSequence = doc_grader_prompt | structured_llm_doc_grader


# print(doc_grader.invoke(
#     {
#         "question": ["what's the most grown crop in africa"],
#         "documents": ["One of the most grown crop in africa is cocoa, and cassava"]
#     }
# ))