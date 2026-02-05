from langchain_classic import hub
from langchain_core.output_parsers import StrOutputParser

from ...llm import llm_model

llm = llm_model.chat_model
prompt = hub.pull("rlm/rag-prompt")
output_parser = StrOutputParser()

answer_generation_chain = prompt | llm | output_parser


# print(answer_generation_chain)