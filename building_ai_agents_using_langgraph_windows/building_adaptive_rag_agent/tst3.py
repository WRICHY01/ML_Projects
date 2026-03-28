from typing import Literal, Optional

from pydantic import BaseModel, Field
from langchain_core.documents import Document
from graph.state import QueryRouter, SubQueryRouter, QueryContext


# what is llm poisoning and an mcp server and how do they relate

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

class RetrievalEvaluation(SubQuery):
    query_context: QueryContext
    is_relavant: Literal["yes", "no"]
    should_rewrite: bool
    generated_answer: Optional[str]
    hallucination_status: Optional[Literal["yes", "no"]]
    generated_answer_grader_status: Optional[Literal["yes", "no"]]

# tst_obj = RetrievalEvaluation(
#     reasoning="This is a test reasoning",
#     extracted_questions="This is a test extracted question",
#     # data_source="vector_store",
#     retrieved_documents=[Document(page_content="this is a test document")],
#     is_relavant="no",
#     should_rewrite=False,
# )


# print("<<<<<<<<<Lets hope the tst obj works and to confirm it contains: ", tst_obj, ">>>>>>>>")

structured_queries = StructuredQuery(sub_queries = [SubQuery(reasoning='Fixed capitalization.', rephrased_question='What is the current price of Bitcoin?', question_status='VALID', resolved_using_history=False),
                                                    SubQuery(reasoning='Input is gibberish with no discernible meaning.', rephrased_question='lahooeljoiajposjaljoi', question_status='INVALID', resolved_using_history=False),
                                                    SubQuery(reasoning='Input is gibberish with no discernible meaning.', rephrased_question='apojapojeklnsjufhljaohf', question_status='INVALID', resolved_using_history=False),
                                                    SubQuery(reasoning='Input is gibberish with no discernible meaning.', rephrased_question='oiueulkjljwoiuowj;laohoifjpsjoif', question_status='INVALID', resolved_using_history=False)]
)
valid_questions = [sq for sq in structured_queries.sub_queries if sq.question_status.lower() == "valid"]
invalid_questions = [sq for sq in structured_queries.sub_queries if sq.question_status.lower() == "invalid"]
print(structured_queries.sub_queries)

# for structured_query in structured_queries.sub_queries:
#     print("structured_queries looks thus: ", structured_queries.sub_queries, "structured_query in the for loop looks thus: ", structured_query)
#     print(f"is the structured_query.question_status which is {structured_query.question_status} -> {structured_query.question_status.lower()} equal to 'invalid' {structured_query.question_status.lower() == "invalid"}")
#     # if structured_query.question_status.lower() in ["invalid","[invalid]"]:
#     if structured_query.question_status.lower() == "invalid" or structured_query.question_status.lower() == "[invalid]":
#         print("\n\n>>>>>>>>deciding if its in the valid or invalid category<<<<<<<<<<<<<<<<\n\n")
#         invalid_questions.append(structured_query)
#         print(structured_queries.sub_queries)
#         structured_queries.sub_queries.remove(structured_query)

        # print("\n\nThis is the outcome of how the structured query looks like: ", structured_queries.sub_queries, " and how the invalid questions looks like: ", invalid_questions, "\n\n")

print(valid_questions)
structured_queries.sub_queries = valid_questions
print("\n\n", structured_queries.sub_queries)

class SubQueryRouter(BaseModel):
    id: int 
    extracted_question: str
    data_source: str


a = [SubQueryRouter(id=13, extracted_question='What is LLM poisoning?', data_source='vector_store'), 
    SubQueryRouter(id=2, extracted_question='What is an MCP server?', data_source='llm_knowledge'), 
    SubQueryRouter(id=3, extracted_question='How do LLM poisoning and an MCP server relate?', data_source='vector_store')]
# b = [SubQueryRouter(extracted_question='What is an AI agent?', data_source='llm_knowledge'), 
#      SubQueryRouter(extracted_question='What is LLM poisoning?', data_source='vector_store'), 
#      SubQueryRouter(extracted_question='How are AI agents and LLM poisoning related to adversarial attacks?', data_source='vector_store')]


ai = []


# # if not ai:
# if False:
#     print("ai content is not empty!")

# else:
#     print("ai content is empty!")

print("ai content is not empty!" if not ai else "ai content is empty!")

class TestSubQuery(BaseModel):
    reasoning: Optional[str] = None
    rephrased_question: Optional[str] = None
    question_status: Optional[Literal["VALID", "INVALID", "AMBIGUOUS"]] = None
    resolved_using_history: Optional[bool] = None
demo = SubQuery(
    reasoning="This is a demo reasoning field",
    rephrased_question="This is a demo rephrased_question field",
    question_status="AMBIGUOUS",
    resolved_using_history=False
)

print(f"demo object looks thus {demo} before altering.")
demo.resolved_using_history = True
print(f"demo object now looks thus {demo} after altering.")

demo2 = TestSubQuery(
    reasoning="hmmm..."
)

print(f"demo2 object looks thus {demo2} currently")
demo2.resolved_using_history = False
print(demo2)
demo2.resolved_using_history = None
print([demo2])

class Amalgam(BaseModel):
    testquerycontext: Optional[TestSubQuery]
    vs_attempt_count: int = 0
    llm_kb_attempt_count: int = 0
    ws_attempt_count: int = 0



tst_amalgam = Amalgam(
    testquerycontext=demo2
)

print(f"The content of tst_amalgam is thus: {tst_amalgam.testquerycontext}")

import uuid

print("The contents of a is thus", a)
new_a = sorted(a, key=lambda x: x.id)
print("The content of a after being sorted", new_a)

