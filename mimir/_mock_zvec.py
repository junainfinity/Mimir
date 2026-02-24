"""
Transparent mock for Zvec when the real C-extension is unavailable.

This mock mirrors the exact v0.2.0 API signatures so that the graph logic,
SQLite bitemporal updates, and embedding pipeline can all be tested even
without the native binary (e.g. on Python 3.13+).
"""

import types


class _MockDataType:
    STRING = "STRING"
    FLOAT_VECTOR = "FLOAT_VECTOR"
    VECTOR_FP32 = "VECTOR_FP32"


class _MockMetricType:
    L2 = "L2"


class _MockCollection:
    def insert(self, docs):
        return [str(i) for i in range(len(docs))]

    def query(self, vectors=None, topk=10, **kwargs):
        return [_MockDoc(id=str(i)) for i in range(topk)]


class _MockDoc:
    def __init__(self, id="1", **kwargs):
        self.id = id
        self.vectors = kwargs.get("vectors", {})
        self.fields = kwargs.get("fields", {})


class _MockVectorQuery:
    def __init__(self, field_name="", vector=None, **kwargs):
        self.field_name = field_name
        self.vector = vector or []


def install_mock():
    """Install and return a mock ``zvec`` module."""
    print(
        "WARNING: zvec could not be imported. "
        "Please ensure you are running Python 3.10-3.12.\n"
        "Mocking zvec for demonstration purposes..."
    )

    mock = types.ModuleType("zvec")

    mock.DataType = _MockDataType
    mock.MetricType = _MockMetricType
    mock.Doc = _MockDoc
    mock.VectorQuery = _MockVectorQuery

    def _collection_schema(name="", fields=None, vectors=None):
        return None

    def _field_schema(name="", data_type=None, **kwargs):
        return None

    def _vector_schema(name="", data_type=None, dimension=0, **kwargs):
        return None

    def _create_and_open(path="", schema=None, **kwargs):
        return _MockCollection()

    def _open(path="", **kwargs):
        return _MockCollection()

    mock.CollectionSchema = _collection_schema
    mock.FieldSchema = _field_schema
    mock.VectorSchema = _vector_schema
    mock.create_and_open = _create_and_open
    mock.open = _open

    return mock
