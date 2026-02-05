from typing import TypedDict, Annotated, Sequence

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, START, END
from IPython.display import display, Image
import nest_asyncio
from langchain_core.runnables.graph import MermaidDrawMethod

from llm import llm_model



class AgentState(TypedDict):
    message: Annotated[Sequence[AnyMessage], add_messages]


def node1(state: AgentState) -> AgentState:

    message = state[message]
    llm = llm_model.chat_model

    system = "write a small poem on the given user's topic"
    prompt = ChatPromptTemplate(
        [
            ("system", system),
            ("human", "{message}")
        ]
    )

    chain = prompt | llm

    return chain.invoke({"message": message})

def should_run(state: AgentState) -> bool:
    message = state["message"]
    if not message:
        return False
    return True



agent = StateGraph(AgentState)
agent.add_node("NODE1", node1)
agent.add_node("passthrough_node", lambda state: state)

agent.add_edge(START, "NODE1")
agent.add_conditional_edges(
    "NODE1",
    should_run,
    {
        True: "passthrough_node",
        False: "NODE1"
    }
)
# agent.add_edge ("NODE1", "passthrough_node")
agent.add_edge("passthrough_node", END)

app = agent.compile()

nest_asyncio.apply()

# display(Image(app.get_graph().draw_mermaid_png(draw_method=MermaidDrawMethod.PYPPETEER)))
mermaid_text = app.get_graph().draw_mermaid()
# print(mermaid_text)
config_header = """---
config:
  theme: base
  themeVariables:
    fontSize: '18px'
    primaryColor: '#0e0d0dff'
    primaryTextColor: '#0bbd3eff'
    primaryBorderColor: '#0a0a0aff'
    lineColor: '#ffffff'
    tertiaryColor: '#050505ff'
  flowchart:
    curve: linear
---
"""
config_section = mermaid_text.split("---", 2)
new_mermaid_text = config_header + config_section[2]

full_content = f"""
```mermaid
{new_mermaid_text}
```
"""

with open("graph_view.md", "w", encoding="utf-8") as f:
    f.write(full_content)

print("mermaid image saved succesfully")


from graph_viewer import view_mermaid_graph

view_mermaid_graph(mermaid_text)