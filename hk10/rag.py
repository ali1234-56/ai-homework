import os
from langchain import hub
from langchain.agents import AgentExecutor, create_react_agent
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_groq import ChatGroq

# 導入 key
os.environ["TAVILY_API_KEY"] = "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"
os.environ["LANGCHAIN_API_KEY"] = "sv2_pt_0b5e556536224e12b0b96559693a74aa_64d388f217"


prompt = hub.pull("hwchase17/react")


llm = ChatGroq(api_key=os.environ.get("GROQ_API_KEY"))

# 定義工具列表，這裡包含了 TavilySearchResults
tools = [TavilySearchResults(max_results=2)] # 返回最多兩個結果

# 創建新的 agent，結合了 ChatGroq 模型、工具列表和模型提示
agent = create_react_agent(llm, tools, prompt)

# 由 LLM 和 prompt 所結合的 chain，用於執行代理的操作，根據這些輸入，Agent 會返回下一個要採取的動作或給使用者的最終回應
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

user_input = input("Enter_question: ")


agent_executor.invoke({"input": user_input})







