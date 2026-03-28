from pydantic import BaseModel, Field

from langchain_classic import hub
from langchain_core.output_parsers import StrOutputParser

from ...llm import llm_model

class GeneratedAnswer(BaseModel):
    generated_answer: str = Field(
        description=""
        )
    summarized_generated_answer: str = Field(
        description=""
        )
    
#Prompt goes here
SYSTEM_INSTRUCTIONS = """
                        
                      """

llm = llm_model.chat_model
prompt = hub.pull("rlm/rag-prompt")
output_parser = StrOutputParser()

answer_generation_chain = prompt | llm | output_parser


# print(answer_generation_chain)