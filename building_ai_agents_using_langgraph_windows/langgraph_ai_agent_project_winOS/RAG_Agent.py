import os

from typing import List, TypedDict, Annotated, Sequence
import pickle

from langchain_core.messages import BaseMessage, SystemMessage, AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode 
from dotenv import load_dotenv


load_dotenv()

script_path = os.path.dirname(os.path.abspath(__file__))
pdf_path = os.path.join(script_path, "datasets", "MIS_ENG405.pdf")

if not os.path.exists(pdf_path):
    raise FileNotFoundError(f"PDF file not found: {pdf_path}")


embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001"
)

pdf_loader =  PyPDFLoader(file_path=pdf_path)

try:
    pages = pdf_loader.load()
    print(f"Successfully loaded the pdf, this particular pdf has {len(pages)} pages.")
except Exception as e:
    print(f"Error Loading pdf: {e}")
    raise

#chunking process
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

pages_split = text_splitter.split_documents(pages)

# persist_directory = os.path.join(script_path, "db")
# collection_name = "technology"

# if not os.path.exists(persist_directory):
#     os.makedirs(persist_directory)
faiss_path = os.path.join(script_path,"faiss_index")
index_faiss_path = os.path.join(faiss_path, "index.faiss")
index_pkl_path = os.path.join(faiss_path, "index.pkl")

try:
    print("Creating a vector database...")

    if os.path.exists(faiss_path) and \
    os.path.isfile(index_faiss_path) and \
    os.path.isfile(index_pkl_path):
        print("vector db already exist, skipping...")
        
        vector_db = FAISS.load_local(
            folder_path=faiss_path,
            embeddings=embeddings,
            index_name="index",
            allow_dangerous_deserialization=True
        )
    # vector_db = Chroma.from_documents(
    #     documents=documents,
    #     embedding=embeddings,
    #     persist_directory=persist_directory,
    #     collection_name=collection_name
    # )

    else:
        vector_db = FAISS.from_documents(
            documents=pages_split,
            embedding=embeddings
        )

        vector_db.save_local(folder_path="./faiss_index",
                            index_name="index")
        with open("faiss_docs.pkl", "wb") as f:
            pickle.dump(pages_split, f)
        print(f"Successfully created a vector database")

except Exception as e:
    print(f"Error setting up FAISS: {str(e)}")
    raise e

retriever = vector_db.as_retriever(
    search_type="similarity",
    search_kwargs={'k':5} #'k' is the amount of chunks to return; in this case 5
)

@tool
def retriever_tool(query: str) -> str:
    """
    This tool searches and returns the desired information from the given document
    """
    print("Currently in the retriever_tool function")
    docs = retriever.invoke(query)

    if not docs:
        return f"I found no relevant information to that query in the given document."

    results = []
    for i, doc in enumerate(docs):
        results.append(f"Document {i+1}:\n{doc},\n{doc.page_content}")

    return "\n\n".join(results)


tools = [retriever_tool]
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash").bind_tools(tools)

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]

def agent(state: AgentState) -> AgentState:
    """
    This is the main function that calls both the tools and the retrieval process
    """
    is_first_interaction =  len(state["messages"]) == 1

    if is_first_interaction:
        history_instruction = """
        Since this is the very start of the conversation and you've not yet provided any answers or assistance,
        if the user's input sounds like a greeting, thank you or a farewell (e.g., "Hello", "Thanks for the help", or "I'm done"), 
        You must point out politely that you've not answered any questions yet. For example respond with something like:
        'Hello! I haven't actually answered any questions for you yet. What specific question do you want to know about the document?'

        DO NOT use the 'CONVERSATION_END' signal in this first turn.
        """
    else:
        history_instruction = """
        If you detect that the user's input indicate the end of the conversation (e.g., "that'd be all", "I'm done", "Thank you").
        your next response must be to acknowledge their thanks and ask a confirmation question like:
        "Would that be all", or "Do you have any other questions about the document?".
        
        If the user confirms they are finished in the turn *after* your confirmation question (e.g., "Yes, that's all" or "No other questions"),
        then and only then should you respond with a final farewell message that contains the exact, case-sensitive phrase 'CONVERSATION_END' anywhere in the content.
        """
    system_prompt = f"""
    You're an intelligent AI assistant who answers questions pertaining the given pdf document loaded into your knowledge base.
    Use the retriever_tool available to answer questions on the pdf document data.You can make multiple calls if needed.
    Its imperative you use the information content in the pdf document to answer the questions when asked and also when asking 
    or when being asked a follow up question.
    Please always cite the specific parts of the documents you use in your answers.

    ***

    {history_instruction}
    You shouldn't call any tools when responding with "CONVERSATION_END".
    """
    if not state["messages"]:
        print("There was no message specified...")
        return
    all_messages = [SystemMessage(content=system_prompt)] + list(state["messages"]) #+ [user_message]
    response =llm.invoke(all_messages)
    print(f"AI: {response.content}")
    if hasattr(response, "tool_calls") and response.tool_calls:
        print(f"USING TOOLS: {[tc['name'] for tc in response.tool_calls]}")
        print("The contents of response.tool_calls is/are: ", response.tool_calls)
    
    print("*" * 300)
    return {"messages": list(state["messages"]) + [response]}

tools_dict = {tool.name: tool for tool in tools}

def take_action(state: AgentState) -> AgentState:
    """
    This is the function that retrieves from the vector db using the retriever tool
    """
    tool_calls = state["messages"][-1].tool_calls
    results = []

    for t in tool_calls:
        print(f"Calling tools: {t['name']} with query: {t["args"].get('query', "No query provided")}")
        
        if not t["name"] in tools_dict:
            print(f"\nTool: {t["name"]} does not exist.")
            result = "Incorrect Tool Name, Please retry and select from list of available tools."

        else:
            result = tools_dict[t["name"]].invoke(t["args"].get("query", ""))
            
            results.append(ToolMessage(tool_call_id=t["id"], name=t["name"], content=str(result)))

    print("Tools Execution Complete. Back to the model!")
    return {"messages": results}

def should_continue(state: AgentState) -> AgentState:
    last_message = state["messages"][-1]
    content_is_end_signal = "CONVERSATION_END" in last_message.content.upper() if last_message.content else False

    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        print("True block, because the last_message has tool_calls attrb", last_message)
        return True

    if content_is_end_signal:
        print("False block, because the last_message doesnt have tool tool_calls attrb", last_message)
        return False

    return False

graph = StateGraph(AgentState)
graph.add_node("agent_node", agent)
graph.add_node("take_action_node", take_action)
graph.add_node("tool_node", ToolNode(tools))
graph.add_edge(START, "agent_node")
graph.add_conditional_edges(
    "agent_node",
    should_continue,
    {
        True: "take_action_node",
        False: END 
    }
)
graph.add_edge("take_action_node", "agent_node")

app = graph.compile()

def agent_starter():
    history = []
    result = {"messages": ""}
    # result_length = len(result["messages"]))

    while True:
        result_length = len(result["messages"])
        if  history and result_length > 0:
            if "CONVERSATION_END" == str(result["messages"][-1].content):
                break
        user_input = input("\nWhat is your question: ")
        if user_input.lower() in ["exit", "quit"]:
            break
        # history.append(user_input)
        history.append(HumanMessage(content=user_input))
        # print("new history looks like this: ", history)
        result = app.invoke({"messages": history})
        print("result outcome be like: ", result)
        history = result["messages"]

        print("\n=== ANSWER ===")
        # print(result["messages"][-1].content)
        print(history[-1].content)

if __name__ == "main":
    agent_starter()