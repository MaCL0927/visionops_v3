from __future__ import annotations

import json
import threading
import urllib.request
from pathlib import Path

from apps.server_api.backend.config import ServerConfig
from apps.server_api.backend.main import VisionOpsServer


def test_server_health_api(tmp_path: Path) -> None:
    config = ServerConfig(host="127.0.0.1", port=0, data_root=tmp_path / "server_data")
    server = VisionOpsServer(config)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        port = server.server_address[1]
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/api/server/health", timeout=5) as response:
            document = json.loads(response.read().decode("utf-8"))
        assert document["message_type"] == "server_health"
        assert document["status"] == "ok"
        assert Path(document["data_root"]).name == "server_data"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
