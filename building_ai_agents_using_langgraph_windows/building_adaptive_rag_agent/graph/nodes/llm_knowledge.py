from typing import Dict, Any

from langchain_core.messages import SystemMessage
from langchain_core.documents import Document

from ..state import AgentState
from ...llm import llm_model

def generate_using_llm_knowledge(state: AgentState) -> Dict[str, Any]:
    """

    """
    print("===in the llm_knowledge_node---")
    # create system prompt telling it to rephrase a question only if needed
    system = f"""
              you are an expert an answering question in the simplest way possible, 
              answer the question to the best of your capability
              """
    question = state["question"]
    all_messages = [SystemMessage(content=system)] + list(question) #+ [question]
    print(f"the content of all_messages is: {all_messages}")
    agent = llm_model.chat_model
    response = agent.invoke(all_messages)
    if not response:
        raise ValueError("LLM response is empty")
    formatted_response = [Document(page_content=response.content)]
    # print(f"response in the llm knowledge section is thus: {response}")
    # print(f"formatted_response in the llm knowledge section is thus: {formatted_response}")

    return {"question": question, "documents": formatted_response}

# print(generate_using_llm_knowledge({"question": "in a concise sentence, how do we solve global warming?"}, True))