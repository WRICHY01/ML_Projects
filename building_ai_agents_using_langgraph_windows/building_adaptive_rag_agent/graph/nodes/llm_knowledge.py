from typing import Dict, Any

from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.documents import Document

from ..chains.llm_knowlege_resolver import query_context_chain
from ..state import AgentState, QueryContext
from ...llm import llm_model

def generate_using_llm_knowledge(state: AgentState) -> Dict[str, Any]:
    """

    """
    # Add doc string too 
    print("---in the llm_knowledge_node---")
    query_router = state["query_router"]
    retrieved_documents = state.get("retrieved_documents", [])
    # print("Retrieved_documents at the very beginning of llm_knowledge node is: ?????????????????????????????", retrieved_documents)
    llm_kb_query_contexts = []

    extracted_questions = [qr_obj.extracted_question for qr_obj in query_router.routes if qr_obj.data_source == "llm_knowledge"]
    
    print(f"The content of extracted_questions is thus: {extracted_questions}")

    if extracted_questions:
        # agent = llm_model.chat_model
        # # response = agent.invoke(all_messages)
        llm_query_contexts = query_context_chain.invoke(extracted_questions) # takes in a list of questions and answers all of them in the order at which they were asked
        # formatted_responses.append(Document(page_content=response.content))
        print(f"The content of llm_kb_query_contexts in the llm_knowledge node is thus: {llm_kb_query_contexts}")
        for llm_query_context in llm_query_contexts:
            # Adding the output of the llm output (llm_query_contexts object) to the QueryContext object for the llm_knowledge node so its consistent with that of the vectorstore and websearch QueryContext too.
            llm_kb_query_context = QueryContext(
                extracted_questions=llm_query_context.extracted_questions,
                retrieved_documents=llm_query_context.retrieved_documents
            )
            llm_kb_query_contexts.append(llm_kb_query_context)
    print(f"The content of llm_kb_query_contexts is: {llm_kb_query_contexts}")
    # if not response: #Still delibrating on this condition if its necessary, it needs serious think through
    #     raise ValueError("LLM response is empty")
        # retrieved_documents.append(all_formatted_responses)
    # all_formatted_responses = [formatted_responses]
    # print(f"retrieved documents in the llm knowledge section is: *********************** {all_formatted_responses}")
    # print(f"formatted_response in the llm knowledge section is thus: {formatted_response}")

    # print("acquired_results from llm_knowledge base is thus<<<<<<<<<<<<<<<<<<<<<<<<<<<<: {formatted_response}>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")

    return {"extracted_questions": extracted_questions, 
            # "retrieved_documents": all_formatted_responses,
            "all_query_contexts": llm_kb_query_contexts}

# print(generate_using_llm_knowledge({"question": "in a concise sentence, how do we solve global warming?"}, True))