import os
import sqlite3
import datetime
import asyncio
from typing import List, Optional, Any, TypedDict, Annotated
import json

# Import Zvec (Note: Requires Python 3.10-3.12)
try:
    import zvec
except ImportError:
    print("WARNING: zvec could not be imported. Please ensure you are running Python 3.10-3.12.")
    print("Mocking zvec for demonstration purposes...")
    class Mockzvec:
        def CollectionSchema(self, fields=None, vector_fields=None): return None
        def FieldSchema(self, name, dtype, dim=None, **kwargs): return None
        def VectorSchema(self, name, dtype, dim=None, **kwargs): return None
        class DataType:
            VARCHAR = "VARCHAR"
            FLOAT_VECTOR = "FLOAT_VECTOR"
            VECTOR_FP32 = "VECTOR_FP32"
        class MetricType:
            L2 = "L2"
    def mock_create_and_open(path, schema): return MockCollection()
    class MockCollection:
        def insert(self, docs): return [str(i) for i in range(len(docs))]
        def query(self, vectors, topk): return [MockDoc(id=str(i)) for i in range(topk)]
    class MockDoc:
        def __init__(self, **kwargs):
            self.id = kwargs.get('id', '1')
    class MockVectorQuery:
        def __init__(self, **kwargs): pass
    zvec = Mockzvec()
    zvec.Doc = MockDoc
    zvec.DataType = Mockzvec.DataType
    zvec.MetricType = Mockzvec.MetricType
    zvec.create_and_open = mock_create_and_open
    zvec.VectorQuery = MockVectorQuery

# Import LangChain Components
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, AIMessage, ToolMessage
from langchain_core.tools import tool
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages


# --- L3: Bitemporal Knowledge Engine (Storage Layer) --- #

class MimirStorage:
    def __init__(self, data_dir: str = "./mimir_data"):
        self.data_dir = data_dir
        os.makedirs(self.data_dir, exist_ok=True)
        
        # 1. Initialize SQLite
        self.db_path = os.path.join(self.data_dir, "mimir.db")
        self._init_sqlite()
        
        # 2. Initialize Zvec
        self.zvec_path = os.path.join(self.data_dir, "zvec_data")
        os.makedirs(self.zvec_path, exist_ok=True)
        self.collection_name = "mimir_chunks"
        self._init_zvec()
        
        # Embeddings Model (using a fast, local model)
        # Note: In production you might want to cache this or load it globally.
        self.embeddings = HuggingFaceEmbeddings(
            model_name="BAAI/bge-small-en-v1.5",
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True}
        )
        self.embedding_dim = 384 # BGE small dim

    def _init_sqlite(self):
        """Initializes the bitemporal SQLite graph."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS bitemporal_graph (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source_entity TEXT,
                relationship TEXT,
                target_entity TEXT,
                scope_tag TEXT,
                valid_from DATETIME DEFAULT CURRENT_TIMESTAMP,
                valid_to DATETIME,
                zvec_reference_id TEXT
            )
        ''')
        # Create indexes for fast lookup
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_source_rel ON bitemporal_graph (source_entity, relationship)")
        conn.commit()
        conn.close()

    def _init_zvec(self):
        """Initializes the zvec collection for storing raw text and embeddings."""
        # Create schema based on standard zvec architectural patterns:
        try:
            schema = zvec.CollectionSchema(
                name=self.collection_name,
                fields=[
                    zvec.FieldSchema(name="id", data_type=zvec.DataType.STRING),
                    zvec.FieldSchema(name="text", data_type=zvec.DataType.STRING)
                ],
                vectors=[
                    # Note: Zvec v0.2.0 VectorSchema init uses `data_type` and `dimension`
                    zvec.VectorSchema(name="embedding", data_type=zvec.DataType.VECTOR_FP32, dimension=384)
                ]
            )
            # This creates or opens relying on the zvec internal behavior based on earlier introspection
            self.collection = zvec.create_and_open(path=os.path.join(self.zvec_path, self.collection_name), schema=schema)
        except Exception as e:
            print(f"Error initializing real zvec schema: {e}. Attempting to open an existing collection...")
            try:
                self.collection = zvec.open(path=os.path.join(self.zvec_path, self.collection_name))
            except Exception as e2:
                print(f"Error opening existing collection: {e2}. If using Mock, this is expected.")
                try:
                    self.collection = zvec.create_and_open(path=os.path.join(self.zvec_path, self.collection_name), schema=None)
                except Exception:
                    pass

    def get_connection(self):
        return sqlite3.connect(self.db_path)

# Initialize a global storage instance to be used by the tools
storage = MimirStorage()


# --- L1 & L2: OS Context Manager & Semantic Router (Tools) --- #

@tool
def archive_memory(content: str, source: str, relation: str, target: str, scope: str) -> str:
    """
    Archives a new fact into memory. 
    It embeds the content in Zvec and updates the SQLite bitemporal graph.
    If the source-relation already exists, it invalidates the old record.
    """
    print(f"[Tool: archive_memory] Archiving: {source} -> {relation} -> {target} ({scope})")
    
    # 1. Embed content
    try:
        embedding = storage.embeddings.embed_query(content)
    except Exception as e:
         print(f"Warning: Failed to generate real embedding due to {e}. Using mock embedding.")
         embedding = [0.0] * storage.embedding_dim
         
    # Generate a unique ID for Zvec
    zvec_id = f"chunk_{datetime.datetime.now().strftime('%Y%m%d%H%M%S%f')}"
    
    # 2. Store in Zvec
    try:
        # zvec 0.2 Doc syntax: pass vectors and scalar fields separately
        doc = zvec.Doc(id=zvec_id, vectors={"embedding": embedding}, fields={"text": content})
        storage.collection.insert([doc])
    except Exception as e:
         print(f"Zvec implementation message: {e}.")
    
    # 3. Update SQLite
    conn = storage.get_connection()
    cursor = conn.cursor()
    
    # Invalidate previous records for the same source & relationship
    cursor.execute('''
        UPDATE bitemporal_graph 
        SET valid_to = CURRENT_TIMESTAMP 
        WHERE source_entity = ? AND relationship = ? AND valid_to IS NULL
    ''', (source, relation))
    
    # Insert new record
    cursor.execute('''
        INSERT INTO bitemporal_graph (source_entity, relationship, target_entity, scope_tag, zvec_reference_id)
        VALUES (?, ?, ?, ?, ?)
    ''', (source, relation, target, scope, zvec_id))
    
    conn.commit()
    conn.close()
    
    return f"Successfully archived memory and updated temporal validities for {source}."

@tool
def search_memory(query: str, timestamp: Optional[str] = None) -> str:
    """
    Searches the memory graph based on a semantic query.
    If a timestamp is provided, it returns information valid at that time.
    Otherwise, returns current valid relationships.
    """
    print(f"[Tool: search_memory] Searching for: '{query}' at time: {timestamp or 'CURRENT'}")
    
    # 1. Embed Query
    try:
         embedding = storage.embeddings.embed_query(query)
    except Exception:
         embedding = [0.0] * storage.embedding_dim

    # 2. Search Zvec (Top 5 matches)
    try:
        from zvec import VectorQuery
        results = storage.collection.query(
            vectors=VectorQuery(field_name="embedding", vector=embedding), 
            topk=5
        )
        if not results:
            return "No semantic matches found."
            
        zvec_ids = [match.id for match in results]
    except Exception as e:
         print(f"Zvec Search failed: {e}")
         zvec_ids = ["mock_id1"] # fallback for mock
         
    if not zvec_ids:
        return "No relevant memories found in Zvec."

    # 3. Cross-reference SQLite bitemporal graph
    conn = storage.get_connection()
    cursor = conn.cursor()
    
    # Build query safely. We use "IN" clause with the placeholders
    placeholders = ','.join('?' * len(zvec_ids))
    search_params = zvec_ids.copy()
    
    query_sql = f'''
        SELECT source_entity, relationship, target_entity, valid_from, valid_to 
        FROM bitemporal_graph 
        WHERE zvec_reference_id IN ({placeholders})
    '''
    
    if timestamp:
        # Check if the requested timestamp falls between valid_from and valid_to
        query_sql += '''
           AND valid_from <= ? 
           AND (valid_to IS NULL OR valid_to >= ?)
        '''
        search_params.extend([timestamp, timestamp])
    else:
        # Currently valid (valid_to is NULL)
        query_sql += " AND valid_to IS NULL"
        
    cursor.execute(query_sql, search_params)
    rows = cursor.fetchall()
    conn.close()
    
    if not rows:
         return "Found matching text, but it was not valid at the requested time period."
         
    # Format Results
    response = "Found Context:\n"
    for row in rows:
        source, relation, target, valid_from, valid_to = row
        v_to = valid_to if valid_to else "Present"
        response += f"- [{valid_from} to {v_to}]: {source} --({relation})--> {target}\n"
        
    return response


# --- L4: Procedural Optimizer (Stub) --- #

async def optimize_trajectory(session_messages: List[BaseMessage]):
    """
    Simulates reviewing chat history to extract a learned behavioral rule.
    """
    print("\n--- [L4: Procedural Optimizer] Running Analysis ---")
    await asyncio.sleep(1) # Simulate think time
    
    # In reality, you'd send the messages to an LLM prompt that asks:
    # "What procedural rule can we learn from this interaction?"
    
    learned_rule = "Learned Rule: Always ensure 'user_name' property is checked before addressing the user."
    print(f"Optimizer Result: Synthesized new rule: {learned_rule}\n")
    return learned_rule


# --- LangGraph Execution Flow --- #

class State(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

def create_agent():
    # Using OSM API as requested for testing
    llm = ChatOpenAI(
        model="gpt-4o-mini", 
        temperature=0,
        api_key="osm_eAq4kL1l2JoNbAHCtoArK2LrWUoLkK9LiFLRsk5E",
        base_url="https://api.osmAPI.com/v1"
    )
    tools = [archive_memory, search_memory]
    llm_with_tools = llm.bind_tools(tools)
    
    system_prompt = SystemMessage(content="""
    <Core_Identity>
    You are Mimir, an autonomous memory agent. 
    You have strict capabilities to archive new facts into a bitemporal knowledge graph and search past facts.
    When a user provides a verifiable statement about themselves, use the `archive_memory` tool.
    When asked a question about the past or present state, use the `search_memory` tool.
    </Core_Identity>
    """)

    def agent_node(state: State):
        messages = [system_prompt] + state["messages"]
        response = llm_with_tools.invoke(messages)
        return {"messages": [response]}
        
    import json
    def tool_node(state: State):
        messages = state["messages"]
        last_message = messages[-1]
        
        tool_results = []
        if type(last_message) is AIMessage and last_message.tool_calls:
           for tool_call in last_message.tool_calls:
               output = None
               if tool_call["name"] == "archive_memory":
                   output = archive_memory.invoke(tool_call["args"])
               elif tool_call["name"] == "search_memory":
                   output = search_memory.invoke(tool_call["args"])
               
               if output:
                   tool_results.append(ToolMessage(content=str(output), tool_call_id=tool_call["id"]))
                   
        return {"messages": tool_results}
        
    def should_continue(state: State):
        messages = state["messages"]
        last_message = messages[-1]
        if type(last_message) is AIMessage and last_message.tool_calls:
            return "tools"
        return END

    graph_builder = StateGraph(State)
    graph_builder.add_node("agent", agent_node)
    graph_builder.add_node("tools", tool_node)
    
    graph_builder.set_entry_point("agent")
    graph_builder.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})
    graph_builder.add_edge("tools", "agent")
    
    return graph_builder.compile()


# --- Main Execution --- #

async def main():
    print("Initializing Mimir Memory System...\n")
    
    graph = create_agent()
    
    # Pre-configure user prompts
    prompts = [
        "Hi Mimir! My name is John and I live in Chennai.",
        "Can you verify where I live?",
        "I just moved to London. Can you update my profile?",
        "Where do I live currently?", 
        "Where did I live before?"
    ]
    
    from langchain_core.messages import HumanMessage
    
    session_messages = []
    
    for prompt in prompts:
        print(f"\nUser: {prompt}")
        human_msg = HumanMessage(content=prompt)
        session_messages.append(human_msg)
        
        # Execute graph
        try:
           state = graph.invoke({"messages": session_messages})
           final_reply = state["messages"][-1]
           print(f"Mimir: {final_reply.content}")
           # update full message history 
           session_messages = state["messages"]
        except Exception as e:
           print(f"[Error] Graph Execution Failed: {e}")
           if "OPENAI_API_KEY" in str(e) or "AuthenticationError" in str(e):
                print("-> Please set OPENAI_API_KEY environment variable. Terminating early.")
                break
           
    # Run optimizer stub after chat session
    await optimize_trajectory(session_messages)
    
    
    print("\n--- Inspecting SQLite State ---")
    conn = storage.get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT source_entity, relationship, target_entity, valid_from, valid_to FROM bitemporal_graph")
    for row in cursor.fetchall():
        print(f"Record: {row[0]} --({row[1]})--> {row[2]} | Valid: {row[3]} to {row[4] or 'NOW'}")
    conn.close()

if __name__ == "__main__":
    asyncio.run(main())
