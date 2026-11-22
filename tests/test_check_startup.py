from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from qdrant_client.http.models import Distance, VectorParams

from scripts import check_startup
from src.config import BASELINE_CONFIG


def test_ollama_requires_exact_model_tag(monkeypatch):
    client = Mock()
    client.list.return_value.models = [SimpleNamespace(model="qwen3:8b")]
    monkeypatch.setattr(check_startup.ollama, "Client", Mock(return_value=client))

    service, model = check_startup.check_ollama(BASELINE_CONFIG)

    assert service.status == "OK"
    assert model.status == "FAIL"
    assert "ollama pull qwen3:4b" in model.detail


@pytest.mark.parametrize(
    ("vectors", "status"),
    [
        (VectorParams(size=384, distance=Distance.COSINE), "OK"),
        (VectorParams(size=768, distance=Distance.COSINE), "FAIL"),
        (VectorParams(size=384, distance=Distance.DOT), "FAIL"),
        ({"text": VectorParams(size=384, distance=Distance.COSINE)}, "FAIL"),
    ],
)
def test_qdrant_checks_unnamed_vector_schema(monkeypatch, vectors, status):
    client = Mock()
    client.get_collections.return_value.collections = [
        SimpleNamespace(name=BASELINE_CONFIG.qdrant_collection_name)
    ]
    client.get_collection.return_value.config.params.vectors = vectors
    monkeypatch.setattr(check_startup, "QdrantClient", Mock(return_value=client))

    service, collection = check_startup.check_qdrant(BASELINE_CONFIG)

    assert service.status == "OK"
    assert collection.status == status
    client.create_collection.assert_not_called()


def test_missing_collection_is_a_first_run_warning(monkeypatch):
    client = Mock()
    client.get_collections.return_value.collections = []
    monkeypatch.setattr(check_startup, "QdrantClient", Mock(return_value=client))

    service, collection = check_startup.check_qdrant(BASELINE_CONFIG)

    assert service.status == "OK"
    assert collection.status == "WARN"
    client.get_collection.assert_not_called()
    client.create_collection.assert_not_called()


def test_dataset_check_does_not_create_a_cache(tmp_path):
    root = tmp_path / "missing"
    assert [check.status for check in check_startup.check_dataset_cache(root)] == ["WARN", "WARN"]
    assert not root.exists()


def test_dataset_check_flags_incomplete_and_accepts_present_metadata(tmp_path):
    (tmp_path / "arcd").mkdir()
    wiki = tmp_path / "wiki_ar"
    wiki.mkdir()
    for filename in ("state.json", "dataset_info.json"):
        (wiki / filename).touch()

    arcd, wikipedia = check_startup.check_dataset_cache(tmp_path)

    assert arcd.status == "FAIL"
    assert "validation/state.json" in arcd.detail
    assert wikipedia.status == "OK"


@pytest.mark.parametrize(("status", "exit_code"), [("WARN", 0), ("FAIL", 1)])
def test_cli_reports_status_and_exit_code(monkeypatch, capsys, status, exit_code):
    monkeypatch.setattr("sys.argv", ["check_startup"])
    monkeypatch.setattr(check_startup, "load_dotenv", Mock())
    monkeypatch.setattr(check_startup, "check_ollama", lambda _: [check_startup.Check(status, "Ollama", "Detail")])
    monkeypatch.setattr(check_startup, "check_qdrant", lambda _: [])
    monkeypatch.setattr(check_startup, "check_dataset_cache", lambda _: [])

    assert check_startup.main() == exit_code
    assert capsys.readouterr().out == f"[{status}] Ollama: Detail\n"
