import os
from langchain import hub
from langchain.agents import AgentExecutor, create_react_agent
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_groq import ChatGroq


os.environ["TAVILY_API_KEY"] = "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"
os.environ["LANGCHAIN_API_KEY"] = "sv2_pt_0b5e556536224e12b0b96559693a74aa_64d388f217"


prompt = hub.pull("hwchase17/react")


llm = ChatGroq(api_key=os.environ.get("GROQ_API_KEY"), model_name="mixtral-8x7b-32768")


tools = [TavilySearchResults(max_results=2)]


agent = create_react_agent(llm, tools, prompt)


agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

user_input = input("Enter : ")


agent_executor.invoke({"input": user_input})







