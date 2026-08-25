# Reach the panel from another machine

The panel and the engine both bind `127.0.0.1` and neither has any
authentication. The panel starts processes and writes files; the engine accepts
any bearer token. **Do not bind either to a LAN address.**

Use an SSH tunnel:

```bash
ssh -L 8080:127.0.0.1:8080 -L 1919:127.0.0.1:1919 you@your-box
```

Then <http://127.0.0.1:8080> on your laptop reaches the panel, and anything
pointed at `http://127.0.0.1:1919` reaches the engine.

## Serving other machines on purpose

If you genuinely want other hosts to use the engine — a second workstation, a
home-lab service — put a reverse proxy in front of it that terminates TLS and
checks a token, and keep llama-server itself on loopback. The `Authorization`
header the engine receives is ignored, so anything that can reach port 1919 can
use your GPU and read every prompt sent through it.
