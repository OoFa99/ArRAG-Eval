from contextlib import closing
from unittest.mock import Mock

import numpy as np
import pytest
from qdrant_client import QdrantClient
from qdrant_client.http.models import PointStruct, Record

from src.retrieval import qdrant_store


QUERY = "القاهرة"
CAPITAL = "القاهرة عاصمة مصر"
CITY = "القاهرة مدينة كبيرة"
DISTRACTOR = "أسوان مدينة مصرية"


class FixedEmbeddings:
    def get_embedding_dimension(self):
        return 2

    def encode(self, texts, *, normalize_embeddings):
        vectors = {
            QUERY: [1.0, 0.0],
            CAPITAL: [1.0, 0.0],
            CITY: [0.8, 0.6],
            DISTRACTOR: [0.0, 1.0],
        }
        return np.array([vectors[text] for text in texts])


@pytest.fixture
def open_store(tmp_path, monkeypatch):
    def open_qdrant(**kwargs):
        client = QdrantClient(path=str(tmp_path))
        client.scroll = Mock(wraps=client.scroll)
        return client

    monkeypatch.setattr(qdrant_store, "QdrantClient", open_qdrant)
    monkeypatch.setattr(qdrant_store, "SentenceTransformer", lambda model_name: FixedEmbeddings())
    monkeypatch.setattr(qdrant_store, "BM25_SCROLL_PAGE_SIZE", 2, raising=False)
    return qdrant_store.VectorStoreQdrant


def test_hybrid_rankings_survive_restart_and_load_all_pages(open_store):
    before = open_store()
    with closing(before.client):
        before.add_documents([CITY], "z-city")
        before.add_documents([CAPITAL], "a-capital")
        before.add_documents([DISTRACTOR], "b-aswan")
        sparse = before.bm25_query(QUERY, top_k=3)
        hybrid = before.hybrid_query(QUERY, top_k=5)

    after = open_store()
    with closing(after.client):
        assert after.hybrid_query(QUERY, top_k=5) == hybrid
        pages = after.client.scroll.call_args_list
        assert len(pages) == 2
        assert pages[0].kwargs["offset"] is None
        assert pages[1].kwargs["offset"] is not None
        assert after.corpus_ids == ["a-capital_chunk_0", "b-aswan_chunk_0", "z-city_chunk_0"]
        assert after.added_documents == {"a-capital", "b-aswan", "z-city"}
        assert after.bm25_query(QUERY, top_k=3) == sparse

    assert [result["id"] for result in hybrid] == [
        "a-capital_chunk_0", "z-city_chunk_0", "b-aswan_chunk_0"
    ]
    assert [result["score"] for result in hybrid] == pytest.approx([2 / 61, 2 / 62, 2 / 63])


def test_legacy_point_without_chunk_id_still_fuses(open_store):
    before = open_store()
    with closing(before.client) as client:
        client.upsert(
            collection_name=before.collection_name,
            points=[PointStruct(id=17, vector=[1.0, 0.0], payload={"text": CAPITAL})],
        )

    after = open_store()
    with closing(after.client):
        assert after.query(QUERY)[0]["id"] == after.bm25_query(QUERY)[0]["id"] == "17"
        results = after.hybrid_query(QUERY)

    assert results == [{"id": "17", "text": CAPITAL, "score": pytest.approx(2 / 61)}]


def test_failed_scroll_leaves_no_partial_bm25_index(open_store):
    store = open_store()
    with closing(store.client) as client:
        client.scroll.side_effect = [
            ([Record(id=17, payload={"text": CAPITAL, "source_document": "partial"})], 18),
            RuntimeError("second page failed"),
        ]

        with pytest.raises(RuntimeError, match="second page failed"):
            store._load_existing_documents()

        assert store.corpus == []
        assert store.corpus_ids == []
        assert store.added_documents == set()
        assert store.bm25 is None
