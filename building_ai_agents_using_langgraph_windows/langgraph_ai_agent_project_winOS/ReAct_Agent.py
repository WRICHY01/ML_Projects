from typing import TypedDict,List, Annotated, Sequence
from langchain_core.messages import BaseMessage, HumanMessage, ToolMessage, SystemMessage
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from dotenv import load_dotenv


load_dotenv()

@tool
def add(a: int, b: int):
    """
    This is an addition function that adds two numerical values together
    """
    # It seems the model is smart enough to discern a misleading docstring like the one below
    # i intentionally wrote a `division function` but it seems it still worked just fine,
    # I'm assuming it because its a simple program and it might behave differently in a large program
    # """
    # This is a division function that divides numerical values 
    # """
    return a + b

@tool
def subtract(a: int, b: int):
    """
    This is an subtraction function that subtracts two numerical values together
    """
    return a - b

@tool
def multiply(a: int, b: int):
    """
    This is an multiplication function that multiplies two numerical values together
    """
    return a * b

tools = [add, subtract, multiply]

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash").bind_tools(tools)

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]

def process(state: AgentState) -> AgentState:
    print("\n In the process node...")
    system_prompt = SystemMessage(
        content="You are my AI assistant. Pls answer my queries to the best of your knowledge"
        )
    response = llm.invoke([system_prompt] + state["messages"])
    print(f"response of the llm is: {response}\n")
    print("^"*50)
    return {"messages": [response]}

def should_use_tool(state: AgentState) -> AgentState:
    messages = state["messages"]
    last_message = messages[-1]
    print(f"\nlooking through messages to look for tool_calls ==> {messages}\n")
    if not last_message.tool_calls:
        return "end"
    else:
        return "continue"

graph = StateGraph(AgentState)
graph.add_node("process_node", process)

tool_node = ToolNode(tools=tools)
graph.add_node("tool_node", tool_node)
graph.add_edge(START, "process_node")
graph.add_conditional_edges(
    "process_node",
    should_use_tool,
    {
        "continue": "tool_node",
        "end": END
    }
)
graph.add_edge("tool_node", "process_node")

app = graph.compile()

def print_stream(stream):
    # count = 0
    for s in stream:
        print(f"\nnow in the stream loop, stream value is: {stream} and s value is: {s}")
        message = s["messages"][-1]
        if isinstance(message, tuple):
            print("message under the isinstance condition")
            print(message)
            print("#"*50)
        else:
            message.pretty_print()
            print("under message.pretty_print()\n")
    # print("no of counts is: ", count)



# inputs = {"messages": [("user", "add 49 + 3, and then multiply the result by 8, also tell me a machine learning joke")]}
inputs = {"messages": [HumanMessage(
    content="add 49 + 3, and then multiply the result by 8, also tell me a machine learning joke"
)]}


print_stream(app.stream(inputs, stream_mode="values"))
