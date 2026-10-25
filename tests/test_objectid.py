from bson import ObjectId
from pydantic import BaseModel

from db.objectid import PyObjectId, serialise_doc, to_objectid, to_str


def test_to_str_objectid():
    oid = ObjectId()
    assert to_str(oid) == str(oid)


def test_to_str_plain_string():
    assert to_str("hello") == "hello"


def test_to_objectid_valid():
    hex_str = "507f1f77bcf86cd799439011"
    result = to_objectid(hex_str)
    assert isinstance(result, ObjectId)
    assert str(result) == hex_str


def test_to_objectid_invalid():
    import pytest

    with pytest.raises(Exception):
        to_objectid("not-an-objectid")


def test_serialise_doc_flat():
    oid = ObjectId()
    doc = {"_id": oid, "name": "test"}
    result = serialise_doc(doc)
    assert result["_id"] == str(oid)
    assert result["name"] == "test"


def test_serialise_doc_nested():
    oid1 = ObjectId()
    oid2 = ObjectId()
    doc = {"_id": oid1, "meta": {"created_by": oid2, "label": "x"}}
    result = serialise_doc(doc)
    assert result["_id"] == str(oid1)
    assert result["meta"]["created_by"] == str(oid2)
    assert result["meta"]["label"] == "x"


def test_serialise_doc_list():
    oid1 = ObjectId()
    oid2 = ObjectId()
    docs = [{"_id": oid1}, {"_id": oid2}]
    result = serialise_doc(docs)
    assert len(result) == 2
    assert result[0]["_id"] == str(oid1)
    assert result[1]["_id"] == str(oid2)


def test_serialise_doc_non_dict():
    assert serialise_doc("plain string") == "plain string"
    assert serialise_doc(42) == 42
    assert serialise_doc(None) is None


def test_pyobjectid_annotation():
    class TestModel(BaseModel):
        id: PyObjectId

    oid = ObjectId()
    m = TestModel(id=oid)
    assert m.id == str(oid)
    assert isinstance(m.id, str)


def test_pyobjectid_from_string():
    class TestModel(BaseModel):
        id: PyObjectId

    m = TestModel(id="507f1f77bcf86cd799439011")
    assert m.id == "507f1f77bcf86cd799439011"
