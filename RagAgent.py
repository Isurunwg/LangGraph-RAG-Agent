import os
import uuid
from typing import TypedDict, Sequence, Annotated, List, Dict, Any
from langchain_core.messages import BaseMessage, ToolMessage, SystemMessage, HumanMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END 
from langgraph.graph.message import add_messages
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from dotenv import load_dotenv

load_dotenv()

# In serverless environments like Vercel, only /tmp is writable
PERSIST_DIR = os.getenv("PERSIST_DIR", "/tmp/chroma_db" if os.getenv("VERCEL") else "./chroma_db")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]

def get_embeddings():
    return OpenAIEmbeddings(
        model="text-embedding-3-small",
        api_key=os.getenv("OPENAI_API_KEY"),
    )

def get_llm():
    return ChatOpenAI(
        model="gpt-4o-mini",
        api_key=os.getenv("OPENAI_API_KEY"),
        temperature=0
    )

def process_and_store_pdf(file_path: str, session_id: str) -> int:
    """Loads PDF, splits text, and embeds into ChromaDB with a session-specific collection."""
    loader = PyPDFLoader(file_path)
    pages = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", ".", " "],
    )
    pages_split = text_splitter.split_documents(pages)

    embeddings = get_embeddings()
    collection_name = f"session_{session_id}"

    Chroma.from_documents(
        documents=pages_split,
        collection_name=collection_name,
        embedding=embeddings,
        persist_directory=PERSIST_DIR
    )

    return len(pages_split)

def build_rag_agent(session_id: str):
    """Builds a LangGraph agent bound to a specific session's vector store."""
    embeddings = get_embeddings()
    collection_name = f"session_{session_id}"

    vectordb = Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=PERSIST_DIR
    )

    retriever = vectordb.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 5}
    )

    @tool
    def retriever_tool(query: str) -> str:
        """Searches and returns relevant information from the uploaded PDF document."""
        docs = retriever.invoke(query)
        if not docs:
            return "No relevant information found."

        results = []
        for i, doc in enumerate(docs):
            page_num = doc.metadata.get("page", "Unknown")
            results.append(f"[Page {page_num}]\n{doc.page_content}")

        return "\n\n".join(results)

    tools = [retriever_tool]
    tools_dict = {t.name: t for t in tools}
    model = get_llm().bind_tools(tools)

    def should_continue(state: AgentState) -> str:
        result = state["messages"][-1]
        return hasattr(result, "tool_calls") and len(result.tool_calls) > 0

    system_prompt = """
    You are an intelligent AI assistant who answers questions based on the PDF document loaded into your knowledge base.
    Use the retriever tool available to answer questions about the document. You can make multiple calls if needed.
    Always cite page numbers or specific sections from the retrieved document snippets in your final response.
    """

    def call_LLM(state: AgentState) -> AgentState:
        messages = [SystemMessage(content=system_prompt)] + list(state["messages"])
        response = model.invoke(messages)
        return {"messages": [response]}

    def take_action(state: AgentState) -> AgentState:
        tool_calls = state["messages"][-1].tool_calls
        tool_results = []

        for t in tool_calls:
            if t['name'] in tools_dict:
                tool_output = tools_dict[t['name']].invoke(t['args'].get("query", ""))
            else:
                tool_output = "Incorrect tool. Choose from available tools."
            tool_results.append(ToolMessage(content=str(tool_output), name=t['name'], tool_call_id=t['id']))

        return {"messages": tool_results}

    graph = StateGraph(AgentState)
    graph.add_node("LLM", call_LLM)
    graph.add_node("retriever_agent", take_action)

    graph.set_entry_point("LLM")
    graph.add_edge("retriever_agent", "LLM")
    graph.add_conditional_edges(
        "LLM",
        should_continue,
        {True: "retriever_agent", False: END}
    )

    return graph.compile()

def run_query(session_id: str, query: str) -> str:
    """Runs a question through the session's compiled RAG graph."""
    app = build_rag_agent(session_id)
    messages = [HumanMessage(content=query)]
    result = app.invoke({"messages": messages})
    return result['messages'][-1].content