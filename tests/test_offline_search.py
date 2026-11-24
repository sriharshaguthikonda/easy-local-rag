import subprocess
import sys
from pathlib import Path
from textwrap import dedent


def test_offline_reader_searches_a_persisted_collection_without_sockets(tmp_path):
    # Keep Chroma's native runtime outside other tests' SDK-mocking process.
    script = dedent("""
    import socket
    import sys

    import chromadb
    from chromadb.config import Settings
    from offline_search import open_local_collection, search_documents

    def deny_socket(*args, **kwargs):
        raise AssertionError("Offline search opened a socket")

    socket.socket.connect = deny_socket
    socket.socket.connect_ex = deny_socket
    socket.getaddrinfo = deny_socket
    tmp_path = sys.argv[1]
    client = chromadb.PersistentClient(
        path=str(tmp_path),
        settings=Settings(anonymized_telemetry=False),
    )
    collection = client.create_collection("offline_regression", embedding_function=None)
    evidence = "Synthetic pseudoferrite ledger detail."
    collection.add(
        ids=["doc-alpha", "doc-beta"],
        documents=[evidence, "Unrelated document."],
        metadatas=[{"file_name": "alpha.txt"}, {"file_name": "beta.txt"}],
        embeddings=[[1, 0], [0, 1]],
    )

    selected = open_local_collection(tmp_path, "offline_regression")
    results = search_documents(selected, "pseudoferrite", limit=5)

    assert [result["id"] for result in results] == ["doc-alpha"]
    assert results[0]["document"] == evidence
    assert results[0]["metadata"]["file_name"] == "alpha.txt"
    assert search_documents(selected, "absent", limit=5) == []
    """)
    result = subprocess.run(
        [sys.executable, "-B", "-c", script, str(tmp_path)],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
