import os
import json
from src.core import config

_index_cache: dict[str, dict] = {}


def _paths_for(collection_name: str) -> dict:
    base = config.DISK_STORAGE_PATH
    collection_dir = os.path.join(base, collection_name)
    return {
        "vocab": os.path.join(collection_dir, "vocab.json"),
        "postings": os.path.join(collection_dir, "inv_index.json"),
        "docs": os.path.join(collection_dir, "docs.json"),
        "meta": os.path.join(collection_dir, "meta.json"),
    }


def store_index(collection_name: str, indexed_docs: dict[str, dict]) -> None:
    paths = _paths_for(collection_name)
    os.makedirs(os.path.dirname(paths["vocab"]), exist_ok=True)

    for key in paths:
        with open(paths[key], "w") as f:
            json.dump(indexed_docs[key], f)
