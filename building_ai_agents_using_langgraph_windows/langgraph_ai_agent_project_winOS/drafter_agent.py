from typing import Annotated, Sequence, TypedDict
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from dotenv import load_dotenv

load_dotenv()


document_content = ""

@tool
def update_doc(content: str) -> str:
    """
    This function updates documents with the provided content

    Args:
        content: The document to be updated.
    """
    print(f"\n currently in the update doc tool\n\n")
    global document_content
    document_content = content
    return f"Document has been successfully updated! The current content is:\n {document_content}"

@tool
def save_doc(filename: str) -> str:
    """
    This function saves the current document as a text file and finishes the process aferwards.

    Args:
        filename: Name for the text file.
    """
    print(f"\n currently in save doc tool\n\n")
    global document_content
    
    if not filename.endswith(".txt"):
        filename = f"{filename}.txt"

    try:
        with open(filename, 'w') as file:
            file.write(document_content)
        print(f"Document has been saved to: {filename}")
        return f"Document has been successfully saved to {filename}"

    except Exception as e:
        raise ValueError(f"Error saving document: {str(e)}")

    

tools = [update_doc, save_doc]

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash").bind_tools(tools)

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]

def process(state: AgentState) -> AgentState:
    system_prompt = SystemMessage(
        content=f"""
        You're a Drafter, a helpful writing assistant. You're a going to help the user update and modify documents.

        - If the user wants to update or modify content, use the 'update_doc' tool with the complete updated content.
        - If the user wants to save and finish, you need to use the 'save_doc' tool.
        - Make sure to always show the current document state after modification.

        The current document is: {document_content}
        """
    )
    
    if not state["messages"]:
        print(f"\nNow at the start of the message \\# clean slate\n\n")
        user_input = "Hello!, I am ready to create a document, lets get started!"
        # user_input = "I'm ready to help you update a document, What would you like to create"
        user_message = HumanMessage(content=user_input)
    else:
        user_input = input("\nWhat would you like to do with the document: ")
        print(f"\n USER: {user_input}")
        user_message = HumanMessage(content=user_input)

    all_messages = [system_prompt] + list(state["messages"]) + [user_message]
    print(f"\nThese are the contents inside all messages: {all_messages}\n")
    response = llm.invoke(all_messages)

    print(f"AI: {response.content}")
    if hasattr(response, "tool_calls") and response.tool_calls:
        print("response.tool_CALLS YIELDS", response.tool_calls)
        print(f"USING TOOLS: {[tc["name"] for tc in response.tool_calls]}")

    print(f"need to understand what user_message, response gives as output, user_message: {user_message}, response: {response}, [user_message, response] {[user_message, response]}")
    return {"messages": list(state["messages"]) + [user_message, response]}


def should_continue(state: AgentState) -> AgentState:
    """
    Determines if we should continue or end the conversation
    """
    print("\ncurrently in the should continue func\n\n")
    messages = state["messages"]
    if not messages:
        return "continue"

    for message in reversed(messages):
        # print("\n output of messages in reverse: ", message, "\n\n")
        # if "updated" in message.content.lower():
        #     print(f"\nfound the desired keyword 'updated' in {message.content} \n\n")
        if (isinstance(message, ToolMessage) and
        "saved" in message.content.lower() and 
        "document" in message.content.lower()):
            print(f"message content in should continue function is: {message}")
            print(f"ToolMessage content in should_continue function is: {ToolMessage}")
            print(f"\nokay before exiting this is what i look like in lowercase: {message.content.lower()}\n\n")
            return "exit"

    # print("\nlets see the contents of message as its being updated: ", messages,"\n")
    print("it seems i am gonna continue, then")
    return "continue"

# def print_messages(messages):
#     if not messages:
#         print("I'm not seeing any message here, i am exiting rn...")
#         return
#     print("\ncurrently in the print_messages func and its output is thus :", messages, "\n\n")
#     # this loop selects the last 3 values only instead of the full texts, making the output display clean instead of being to bulky
#     for message in messages[-3:]:
#         print("\nmessage in for loop", message, "\n\n")
#         # print("last message: ", messages[-1:])
#         # print("second to the last  message: ", messages[-2:])
#         # show the output of the ToolMessage if the condition is satisfied in a structured format.
#         if isinstance(message, ToolMessage):
#             print(f"tOOL RESULT: {message.content}")

graph = StateGraph(AgentState)
graph.add_node("process_node", process)
graph.add_node("tool_node", ToolNode(tools))
graph.set_entry_point("process_node")
# graph.add_edge(START, "process_node")
graph.add_edge("process_node", "tool_node")
graph.add_conditional_edges(
    "tool_node",
    should_continue,
    {
        "continue": "process_node",
        "exit": END
    }
)
graph.add_edge("process_node", "tool_node")

app = graph.compile()

def print_stream():

    input = {"messages": []}
    for s in app.stream(input, stream_mode="values"):
        # if not s["messages"]:
        #     return
        if "messages" in s:
            messages = s["messages"]
        

        # if isinstance(message, tuple):
        # print("\nin the print stream funtion, this is the output of stream s: ", s)
            print("\n\nin the print stream function, this is the output of stream messages: ", messages, "\n\n")
        

            for message in messages:
                if isinstance(message, ToolMessage):
                    print("message is: ", message)
                    print("ToolMessage content is: ", ToolMessage)
                    print(f"Tool Result: {message.content}")

            # print(messages)
            # print_messages(messages)
        # else:
        #     print(message)

print_stream()