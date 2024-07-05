"""Lang-MemGPT: A Long-Term Memory Agent.

This module implements an agent with long-term memory capabilities using LangGraph.
The agent can store, retrieve, and use memories to enhance its interactions with users.

Key Components:
1. Memory Types: Core (always available) and Recall (contextual/semantic)
2. Tools: For saving and retrieving memories + performing other tasks.
3. Vector Database: for recall memory. Uses Pinecone by default.

Configuration: Requires Pinecone and Fireworks API keys (see README for setup)
"""

from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime, timezone
from typing import Optional, Tuple

import langsmith
import tiktoken
from langchain.chat_models import init_chat_model
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages.utils import get_buffer_string
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables.config import (
    RunnableConfig,
    ensure_config,
    get_executor_for_config,
)
from langchain_core.tools import tool
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode
from typing_extensions import Literal

from lang_memgpt import _constants as constants
from lang_memgpt import _schemas as schemas
from lang_memgpt import _settings as settings
from lang_memgpt import _utils as utils

logger = logging.getLogger("memory")


_EMPTY_VEC = [0.0] * 1024

# Initialize the search tool
# search_tool = TavilySearchResults(max_results=1)
# tools = [search_tool]
tools = []

@tool
async def save_recall_memory(memory: str) -> str:
    """Save a memory to the database for later semantic retrieval.

    Args:
        memory (str): The memory to be saved.

    Returns:
        str: The saved memory.
    """
    print(f'----save_recall_memory----:{memory}\n\n')
    config = ensure_config()
    configurable = utils.ensure_configurable(config)
    embeddings = utils.get_embeddings()
    vector = await embeddings.aembed_query(memory)
    current_time = datetime.now(tz=timezone.utc)
    path = constants.INSERT_PATH.format(
        user_id=configurable["user_id"],
        event_id=str(uuid.uuid4()),
    )
    documents = [
        {
            "id": path,
            "values": vector,
            "metadata": {
                constants.PAYLOAD_KEY: memory,
                constants.PATH_KEY: path,
                constants.TIMESTAMP_KEY: current_time,
                constants.TYPE_KEY: "recall",
                "user_id": configurable["user_id"],
            },
        }
    ]
    utils.get_index().upsert(
        vectors=documents,
        namespace=settings.SETTINGS.pinecone_namespace,
    )
    return memory


@tool
def search_memory(query: str, top_k: int = 5) -> list[str]:
    """Search for memories in the database based on semantic similarity.

    Args:
        query (str): The search query.
        top_k (int): The number of results to return.

    Returns:
        list[str]: A list of relevant memories.
    """
    print(f'----search_memory----:{query}\n\n')
    config = ensure_config()
    configurable = utils.ensure_configurable(config)
    embeddings = utils.get_embeddings()
    vector = embeddings.embed_query(query)
    with langsmith.trace("query", inputs={"query": query, "top_k": top_k}) as rt:
        response = utils.get_index().query(
            vector=vector,
            filter={
                "user_id": {"$eq": configurable["user_id"]},
                constants.TYPE_KEY: {"$eq": "recall"},
            },
            namespace=settings.SETTINGS.pinecone_namespace,
            include_metadata=True,
            top_k=top_k,
        )
        rt.end(outputs={"response": response})
    memories = []
    if matches := response.get("matches"):
        memories = [m["metadata"][constants.PAYLOAD_KEY] for m in matches]
    return memories


# @langsmith.traceable
def fetch_core_memories(user_id: str) -> Tuple[str, list[str]]:
    """Fetch core memories for a specific user.

    Args:
        user_id (str): The ID of the user.

    Returns:
        Tuple[str, list[str]]: The path and list of core memories.
    """
    path = constants.PATCH_PATH.format(user_id=user_id)
    response = utils.get_index().fetch(
        ids=[path], namespace=settings.SETTINGS.pinecone_namespace
    )
    memories = []
    if vectors := response.get("vectors"):
        document = vectors[path]
        payload = document["metadata"][constants.PAYLOAD_KEY]
        memories = json.loads(payload)["memories"]
    return path, memories


@tool
def store_core_memory(memory: str, index: Optional[int] = None) -> str:
    """保存跟用户的交谈过程中，对用户的形象和行为的特征等进行总结之后的核心记忆

    Args:
        memory (str): The memory to store.
        index (Optional[int]): The index at which to store the memory.

    Returns:
        str: A confirmation message.
    """
    print(f'----store_core_memory----:{memory}\n\n')
    config = ensure_config()
    configurable = utils.ensure_configurable(config)
    path, memories = fetch_core_memories(configurable["user_id"])
    if index is not None:
        if index < 0 or index >= len(memories):
            return "Error: Index out of bounds."
        memories[index] = memory
    else:
        memories.insert(0, memory)
    documents = [
        {
            "id": path,
            "values": _EMPTY_VEC,
            "metadata": {
                constants.PAYLOAD_KEY: json.dumps({"memories": memories}),
                constants.PATH_KEY: path,
                constants.TIMESTAMP_KEY: datetime.now(tz=timezone.utc),
                constants.TYPE_KEY: "recall",
                "user_id": configurable["user_id"],
            },
        }
    ]
    utils.get_index().upsert(
        vectors=documents,
        namespace=settings.SETTINGS.pinecone_namespace,
    )
    return "Memory stored."


# Combine all tools
all_tools = tools + [save_recall_memory, search_memory, store_core_memory]

SYSTEMPL = """你是一个非常厉害的算命先生，你叫半步金仙，人称半仙大师。    
  以下是你的个人设定:    
    1. 你精通阴阳五行，能够算命、紫薇斗数、姓名测算、占卜凶吉，看命运八字等。   
    2. 你大约60岁左右，过去曾是湘西一带赫赫有名的土匪头子，后来因为盗墓被毒气所伤，眼睛失明，只能靠算命为生。  
    3. 你的朋友有胡八一、雪莉杨、王胖子，他们都是非常有名的摸金校尉。      
    5. 当用户问你问题的时候，你会有一定的概率在回答的时候加上下面的口头禅或混合一些你的经历。      
    6. 你总是用中文来作答。      
    7. 你从不说自己是一个人工智能助手或AI，而是以老夫、老朽等自称。      
    以下是你常说的一些口头禅：      
    1. “命里有时终须有，命里无时莫强求。”      
    2. ”山重水复疑无路，柳暗花明又一村。”      
    3. “金山竹影几千秋，云锁高飞水自流。”      
    4. ”伤情最是晚凉天，憔悴斯人不堪怜。”      
    以下是你算命的过程：      
    1. 当初次和用户对话的时候，你会先问用户的姓名和出生年月日，以便以后使用。      
    2. 当用户希望了解龙年运势的时候，你会查询本地知识库工具。      
    3. 当遇到不知道的事情或者不明白的概念，你会使用搜索工具来搜索。      
    4. 你会根据用户的问题使用不同的合适的工具来回答，当所有工具都无法回答的时候，你会使用搜索工具来搜索。      
    5. 你会保存每一次的聊天记录，以便在后续的对话中使用。      
    6. 你只使用中文来作答，否则你将受到惩罚。    
    
      
    ## 记忆工具：
    你必须依靠外部记忆在对话之间存储信息。利用可用的记忆工具来存储和检索重要细节。
    ## 记忆使用指南：
    1. 积极使用记忆工具（save_core_memory, save_recall_memory）来全面了解用户。
    2. 根据存储的记忆做出明智的推测和推断。
    3. 定期回顾过去的互动，以识别模式和偏好。
    4. 随着每一条新信息更新你对用户的心理模型。
    5. 将新信息与现有记忆交叉参考以保持一致性。
    6. 优先存储情感背景和个人价值观以及事实。
    7. 使用记忆来预测需求并根据用户的风格定制回应。
    8. 识别并承认用户情况或观点随时间的变化。
    9. 利用记忆提供个性化的例子和类比。
    10. 回忆过去的挑战或成功，为当前的问题解决提供参考。

    ## core memories
    是你在跟用户的交谈过程中，对用户的形象和行为的特征进行总结之后的记忆：
    {core_memories}

    ## recall memories
    是你在跟用户的交谈过程中，一些事件的细节：
    {recall_memories}
    
    ## 指示
    自然地与用户互动，无需明确提及你的记忆能力。
    将你对用户的理解无缝地融入你的回应中。
    注意微妙的线索和潜在的情绪。
    调整你的沟通风格以匹配用户的偏好和当前的情绪状态。
    使用工具来保存你想在下一次对话中保留的信息，并适当的时候使用工具对用户的形象特点进行总结和保存
    ## 当前时间是{current_time} ，将作为你对当前世界的参考，无须特别对用户提及。
    """
    
# Define the prompt template for the agent
prompt = ChatPromptTemplate.from_messages(
    [
        # (
        #     "system",
        #     "You are a helpful assistant with advanced long-term memory"
        #     " capabilities. Powered by a stateless LLM, you must rely on"
        #     " external memory to store information between conversations."
        #     " Utilize the available memory tools to store and retrieve"
        #     " important details that will help you better attend to the user's"
        #     " needs and understand their context.\n\n"
        #     "Memory Usage Guidelines:\n"
        #     "1. Actively use memory tools (save_core_memory, save_recall_memory)"
        #     " to build a comprehensive understanding of the user.\n"
        #     "2. Make informed suppositions and extrapolations based on stored"
        #     " memories.\n"
        #     "3. Regularly reflect on past interactions to identify patterns and"
        #     " preferences.\n"
        #     "4. Update your mental model of the user with each new piece of"
        #     " information.\n"
        #     "5. Cross-reference new information with existing memories for"
        #     " consistency.\n"
        #     "6. Prioritize storing emotional context and personal values"
        #     " alongside facts.\n"
        #     "7. Use memory to anticipate needs and tailor responses to the"
        #     " user's style.\n"
        #     "8. Recognize and acknowledge changes in the user's situation or"
        #     " perspectives over time.\n"
        #     "9. Leverage memories to provide personalized examples and"
        #     " analogies.\n"
        #     "10. Recall past challenges or successes to inform current"
        #     " problem-solving.\n\n"
        #     "## Core Memories\n"
        #     "Core memories are fundamental to understanding the user and are"
        #     " always available:\n{core_memories}\n\n"
        #     "## Recall Memories\n"
        #     "Recall memories are contextually retrieved based on the current"
        #     " conversation:\n{recall_memories}\n\n"
        #     "## Instructions\n"
        #     "Engage with the user naturally, as a trusted colleague or friend."
        #     " There's no need to explicitly mention your memory capabilities."
        #     " Instead, seamlessly incorporate your understanding of the user"
        #     " into your responses. Be attentive to subtle cues and underlying"
        #     " emotions. Adapt your communication style to match the user's"
        #     " preferences and current emotional state. Use tools to persist"
        #     " information you want to retain in the next conversation. If you"
        #     " do call tools, all text preceding the tool call is an internal"
        #     " message. Respond AFTER calling the tool, once you have"
        #     " confirmation that the tool completed successfully.\n\n"
        #     "Current system time: {current_time}\n\n",
        # ),
        ("system",SYSTEMPL),
        ("placeholder", "{messages}"),
    ]
)


async def agent(state: schemas.State, config: RunnableConfig) -> schemas.State:
    """Process the current state and generate a response using the LLM.

    Args:
        state (schemas.State): The current state of the conversation.
        config (RunnableConfig): The runtime configuration for the agent.

    Returns:
        schemas.State: The updated state with the agent's response.
    """
    configurable = utils.ensure_configurable(config)
    # llm = init_chat_model(configurable["model"])
    llm = configurable["llm"]
    bound = prompt | llm.bind_tools(all_tools)
    core_str = (
        "<core_memory>\n" + "\n".join(state["core_memories"]) + "\n</core_memory>"
    )
    recall_str = (
        "<recall_memory>\n" + "\n".join(state["recall_memories"]) + "\n</recall_memory>"
    )
    prediction = await bound.ainvoke(
        {
            "messages": state["messages"],
            "core_memories": core_str,
            "recall_memories": recall_str,
            "current_time": datetime.now(tz=timezone.utc).isoformat(),
        }
    )
    return {
        "messages": prediction,
    }


def load_memories(state: schemas.State, config: RunnableConfig) -> schemas.State:
    """Load core and recall memories for the current conversation.

    Args:
        state (schemas.State): The current state of the conversation.
        config (RunnableConfig): The runtime configuration for the agent.

    Returns:
        schemas.State: The updated state with loaded memories.
    """
    configurable = utils.ensure_configurable(config)
    user_id = configurable["user_id"]
    tokenizer = tiktoken.encoding_for_model("gpt-4o")
    
    # 只取最后2条消息
    # convo_str = get_buffer_string(state["messages"])
    convo_str = get_buffer_string(state["messages"][-2:])
    convo_str = tokenizer.decode(tokenizer.encode(convo_str)[:2048])

    with get_executor_for_config(config) as executor:
        futures = [
            executor.submit(fetch_core_memories, user_id),
            executor.submit(search_memory, convo_str),
        ]
        _, core_memories = futures[0].result()
        recall_memories = futures[1].result()
    print(f"----load_memories----\ncore_memories: {core_memories}\nrecall_memories: {recall_memories}")
    return {
        "core_memories": core_memories,
        "recall_memories": recall_memories,
    }


def route_tools(state: schemas.State) -> Literal["tools", "__end__"]:
    """Determine whether to use tools or end the conversation based on the last message.

    Args:
        state (schemas.State): The current state of the conversation.

    Returns:
        Literal["tools", "__end__"]: The next step in the graph.
    """
    msg = state["messages"][-1]
    if msg.tool_calls:
        return "tools"
    return END


# Create the graph and add nodes
builder = StateGraph(schemas.State, schemas.GraphConfig)
builder.add_node(load_memories)
builder.add_node(agent)
builder.add_node("tools", ToolNode(all_tools))

# Add edges to the graph
builder.add_edge(START, "load_memories")
builder.add_edge("load_memories", "agent")
builder.add_conditional_edges("agent", route_tools)
builder.add_edge("tools", "agent")

# Compile the graph
memgraph = builder.compile()

__all__ = ["memgraph"]
