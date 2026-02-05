from typing import TypedDict, List, Annotated, Union
from langchain_core.messages import HumanMessage, AIMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, START, END
from dotenv import load_dotenv, dotenv_values

load_dotenv()

class AgentState(TypedDict):
  message: Annotated[List[Union[HumanMessage, AIMessage]], add_messages]

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

def process(state: AgentState) -> AgentState:
  """
  A simple node that process and responds to human message using gemini
  """
  # The workhorse invoke method its purpose here is to take the message recieved from the
  # overall workflow and send it to the llm api(in this case gemini) in order to get text response.
  response = llm.invoke(state["message"]) 
  state["message"].append(AIMessage(response.content))
  print(f"AI: {response.content}")

  return state


graph = StateGraph(AgentState)
graph.add_node("process_node", process)
graph.set_entry_point("process_node")
graph.set_finish_point("process_node")
agent = graph.compile()

human_history = []
user_input = input("Enter Message: ")
while user_input != "quit":
  human_history.append(HumanMessage(content=user_input))\
  # The `invoke` method here starts the overall compiled program(`agent`)
  h_response = agent.invoke({"message": human_history})
  # human_history.append(h_response)
  human_history = h_response["message"]
  # print(h_response["message"])
  hh_length = len(human_history)
  if hh_length >= 10:
    human_history = human_history[5:]
  user_input = input("Enter Message: ")


with open("conversation_history.txt", 'a') as file:
  file.write("\nYour Coversation History.\n")
  for message in human_history:
    if isinstance(message, HumanMessage):
      file.write(f"You: {message.content}\n")
    elif isinstance(message, AIMessage):
      file.write(f"AI: {message.content}\n")
  file.write("End of Conversation")
  
print("conversation successfully saved to conversation_history.txt")