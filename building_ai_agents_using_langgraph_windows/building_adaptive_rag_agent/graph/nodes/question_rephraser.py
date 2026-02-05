from typing import Dict, Any

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

from ..chains.question_remodifier import question_rewriter_chain
# from  ...llm.llm_model import llm_model
# from .grade_documents import documents_grader
from ..state import AgentState



def question_transformer(state: AgentState) -> Dict[str, Any]:
    """

    """
    print("processing question...")
    question = state["question"]
    print(f"question in question_rephraser node is: {question}")
    should_rewrite = state.get("should_rewrite")

    # if should_rewrite is None:
    #     print(f"initially in the  question rephraser module, should_rewrite is {should_rewrite}")
    # should_rewrite_count = state.get("should_rewrite_count", 0)

    # if should_rewrite_count > 0:
    #     should_rewrite = True
    
    
    # should_rewrite_count += 1
    # needs_preprocessing = state.get("needs_preprocessing", None)
    # print("getting rephrased_question value")
    rephrased_question = question_rewriter_chain.invoke({"question": question,
                                                #    "needs_preprocessing": needs_preprocessing,
                                                        "should_rewrite": should_rewrite})
    # should_rewrite_count += 1
    print(")0"*300)
    print("rephrased_question looks thus", rephrased_question)
    # print("the last instance in the rephrased_question list is: ", rephrased_question[-1])
    print("()0"*300)
    
    return {"question": rephrased_question,
            # "needs_preprocessing": needs_preprocessing,
            "should_rewrite": should_rewrite}
            # "should_rewrite_count": should_rewrite_count}



# print(question_transformer({"question": "How many countries are there in a continent?", "rewrite_flag": True}))