#!/usr/bin/env bash
# Start localhost.run tunnel for port 8003, write current URL to classifier/logs/tunnel_url.txt
set -e

URL_FILE="/path/to/MetaP/classifier/logs/tunnel_url.txt"
mkdir -p "$(dirname "$URL_FILE")"

ssh -o StrictHostKeyChecking=no \
    -o ServerAliveInterval=30 \
    -o ServerAliveCountMax=3 \
    -R 80:localhost:8003 nokey@localhost.run -- --output=json 2>&1 | \
python3 -u - <<'PYEOF'
import sys, json

for raw in sys.stdin:
    line = raw.rstrip()
    print(line, flush=True)
    try:
        d = json.loads(line)
        if d.get("type") == "opened" and "address" in d:
            url = "https://" + d["address"]
            with open("/path/to/MetaP/classifier/logs/tunnel_url.txt", "w") as f:
                f.write(url + "\n")
            print(f"=== Tunnel URL: {url} ===", flush=True)
            print(f"=== API Docs:   {url}/docs ===", flush=True)
    except Exception:
        pass
PYEOF
