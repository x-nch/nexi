import os
import tempfile

import pytest


@pytest.fixture(autouse=True)
def _isolate_chromadb():
    tmp = tempfile.mkdtemp(prefix="chroma_test_")
    os.environ["STORAGE_PATH"] = tmp
    import agentmemory.client as _am_client
    _am_client.client = None
    yield
    _am_client.client = None
    import shutil
    shutil.rmtree(tmp, ignore_errors=True)
