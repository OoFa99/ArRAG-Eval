"""Check prerequisites for the local RAG pipeline without changing them."""

import argparse
from collections.abc import Iterator
from contextlib import closing
import os
from pathlib import Path
from typing import Literal, NamedTuple

from dotenv import load_dotenv
import ollama
from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import UnexpectedResponse
from qdrant_client.http.models import Distance, VectorParams

from src.config import BASELINE_CONFIG, ExperimentConfig
from src.data.corpus_download import ARCD_CACHE_NAME, DATA_PATH, WIKIPEDIA_CACHE_NAME


REQUEST_TIMEOUT = 3
EMBEDDING_DIMENSIONS = {"intfloat/multilingual-e5-small": 384}


class Check(NamedTuple):
    status: Literal["OK", "WARN", "FAIL"]
    component: str
    detail: str


def check_ollama(config: ExperimentConfig) -> Iterator[Check]:
    try:
        with closing(ollama.Client(host=config.ollama_host, timeout=REQUEST_TIMEOUT)) as client:
            models = {entry.model for entry in client.list().models}
    except ollama.ResponseError as error:
        yield Check("FAIL", "Ollama", f"{config.ollama_host} returned HTTP {error.status_code}.")
        return
    except Exception as error:
        yield Check(
            "FAIL", "Ollama",
            f"Could not list models at {config.ollama_host} ({type(error).__name__}). "
            "Check the host or run: ollama serve",
        )
        return

    yield Check("OK", "Ollama", f"Model list available at {config.ollama_host}.")
    model = config.llm_model
    if ":" not in model.rsplit("/", 1)[-1]:
        model += ":latest"
    if model in models:
        yield Check("OK", "Ollama model", f"{model} is installed.")
    else:
        yield Check("FAIL", "Ollama model", f"{model} is missing. On the server, run: ollama pull {model}")


def check_qdrant(config: ExperimentConfig) -> Iterator[Check]:
    name = config.qdrant_collection_name
    component = "Qdrant"
    try:
        with closing(QdrantClient(
            url=config.qdrant_url,
            api_key=config.qdrant_api_key or os.getenv("QDRANT_API_KEY"),
            timeout=REQUEST_TIMEOUT,
            check_compatibility=False,
        )) as client:
            collections = client.get_collections().collections
            yield Check("OK", "Qdrant", f"Collection list available at {config.qdrant_url}.")
            component = "Qdrant collection"
            if not any(collection.name == name for collection in collections):
                yield Check("WARN", "Qdrant collection", f"{name} is missing. The experiment can create it.")
                return
            vectors = client.get_collection(name).config.params.vectors
    except UnexpectedResponse as error:
        yield Check(
            "FAIL", component, f"{config.qdrant_url} returned HTTP {error.status_code}. "
            "Check the URL and access credentials.",
        )
        return
    except Exception as error:
        yield Check(
            "FAIL", component,
            f"Could not read collections at {config.qdrant_url} ({type(error).__name__}). "
            "Check that Qdrant is running and reachable.",
        )
        return

    if not isinstance(vectors, VectorParams) or vectors.multivector_config is not None:
        yield Check("FAIL", "Qdrant collection", f"{name} needs a single unnamed dense vector.")
        return
    dimension = EMBEDDING_DIMENSIONS.get(config.embedding_model)
    if vectors.distance != Distance.COSINE or (dimension is not None and vectors.size != dimension):
        expected = f"{dimension} dimensions with Cosine distance" if dimension else "Cosine distance"
        yield Check(
            "FAIL", "Qdrant collection",
            f"{name} has {vectors.size} dimensions with {vectors.distance.value} distance. "
            f"Expected {expected}. Configure a compatible collection.",
        )
    elif dimension is None:
        yield Check(
            "WARN", "Qdrant collection",
            f"{name} uses Cosine distance. The expected dimensions for {config.embedding_model} are unknown.",
        )
    else:
        yield Check("OK", "Qdrant collection", f"{name} has {dimension} dimensions with Cosine distance.")


def check_dataset_cache(root: Path) -> Iterator[Check]:
    for label, folder, required_files in (
        (
            "ARCD cache", ARCD_CACHE_NAME,
            ("dataset_dict.json", "validation/state.json", "validation/dataset_info.json"),
        ),
        ("Wikipedia cache", WIKIPEDIA_CACHE_NAME, ("state.json", "dataset_info.json")),
    ):
        path = (root / folder).resolve()
        if not path.exists():
            yield Check("WARN", label, f"{path} is missing. The loader can download it on first use.")
        else:
            missing = [file for file in required_files if not (path / file).is_file()]
            if missing:
                yield Check("FAIL", label, f"{path} is missing cache metadata ({', '.join(missing)}).")
            else:
                yield Check("OK", label, f"Found cache metadata in {path}.")


def main() -> int:
    argparse.ArgumentParser(description=__doc__).parse_args()
    load_dotenv()
    failed = False
    for checks in (
        check_ollama(BASELINE_CONFIG),
        check_qdrant(BASELINE_CONFIG),
        check_dataset_cache(Path(DATA_PATH)),
    ):
        for status, component, detail in checks:
            print(f"[{status}] {component}: {detail}")
            failed |= status == "FAIL"
    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main())
