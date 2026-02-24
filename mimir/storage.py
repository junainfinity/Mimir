"""
L3: Bitemporal Knowledge Engine (Storage Layer)

Manages dual storage via Zvec (dense vectors) and SQLite (bitemporal graph edges).
"""

import os
import sqlite3

# Import Zvec — falls back to a transparent mock on unsupported Python versions
try:
    import zvec
except ImportError:
    from mimir._mock_zvec import install_mock
    zvec = install_mock()


class MimirStorage:
    """
    Core storage engine for Mimir.
    
    Initializes both Zvec (vector search) and SQLite (bitemporal graph)
    in the same data directory.
    
    Args:
        data_dir: Path to the data directory. Defaults to ``./mimir_data``.
        embedding_model: HuggingFace model name for embeddings. 
            Defaults to ``BAAI/bge-small-en-v1.5``.
        embedding_dim: Dimension of the embedding vectors. Defaults to 384.
    
    Example::
    
        from mimir import MimirStorage
        storage = MimirStorage("./my_data")
        # storage.collection  -> Zvec collection
        # storage.get_connection() -> SQLite connection
    """

    def __init__(
        self,
        data_dir: str = "./mimir_data",
        embedding_model: str = "BAAI/bge-small-en-v1.5",
        embedding_dim: int = 384,
    ):
        self.data_dir = data_dir
        self.embedding_dim = embedding_dim
        os.makedirs(self.data_dir, exist_ok=True)

        # 1. Initialize SQLite
        self.db_path = os.path.join(self.data_dir, "mimir.db")
        self._init_sqlite()

        # 2. Initialize Zvec
        self.zvec_path = os.path.join(self.data_dir, "zvec_data")
        os.makedirs(self.zvec_path, exist_ok=True)
        self.collection_name = "mimir_chunks"
        self._init_zvec()

        # 3. Initialize embeddings (lazy import to keep startup fast)
        from langchain_huggingface import HuggingFaceEmbeddings

        self.embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )

    # ------------------------------------------------------------------ #
    # SQLite
    # ------------------------------------------------------------------ #

    def _init_sqlite(self):
        """Create the ``bitemporal_graph`` table if it doesn't exist."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
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
        """)
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_source_rel "
            "ON bitemporal_graph (source_entity, relationship)"
        )
        conn.commit()
        conn.close()

    def get_connection(self) -> sqlite3.Connection:
        """Return a new SQLite connection to the bitemporal graph."""
        return sqlite3.connect(self.db_path)

    # ------------------------------------------------------------------ #
    # Zvec
    # ------------------------------------------------------------------ #

    def _init_zvec(self):
        """Create or open the ``mimir_chunks`` Zvec collection."""
        collection_path = os.path.join(self.zvec_path, self.collection_name)
        try:
            schema = zvec.CollectionSchema(
                name=self.collection_name,
                fields=[
                    zvec.FieldSchema(name="id", data_type=zvec.DataType.STRING),
                    zvec.FieldSchema(name="text", data_type=zvec.DataType.STRING),
                ],
                vectors=[
                    zvec.VectorSchema(
                        name="embedding",
                        data_type=zvec.DataType.VECTOR_FP32,
                        dimension=self.embedding_dim,
                    )
                ],
            )
            self.collection = zvec.create_and_open(path=collection_path, schema=schema)
        except Exception:
            # Collection may already exist on disk
            try:
                self.collection = zvec.open(path=collection_path)
            except Exception:
                self.collection = None
