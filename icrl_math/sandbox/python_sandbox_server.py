"""Python sandbox HTTP server, plug-in replacement for ICRL's retrieval_server.py.

Drops into the same URL slot (e.g. http://127.0.0.1:8000/retrieve) and answers
with the SAME response shape ICRL's `_passages2string` consumes, so we do NOT
need to modify ICRL's generation.py or trainer code.

ICRL expects:
    POST /retrieve
    request:  {"queries": [str], "topk": int, "return_scores": bool}
    response: {"result": [[{"document": {"contents": "title\\nbody..."}}], ...]}

We treat each "query" as a Python source snippet, run it in a subprocess with
timeout + memory limit, and return stdout in the same envelope (title="stdout").
ICRL's _passages2string then wraps it as `Doc 1(Title: stdout) <stdout>` and
the trainer surrounds it with <information>...</information> automatically.

Loss masking on the returned tokens is already handled by ICRL's
ray_trainer._create_loss_mask, so the model is not credited for sandbox output.

Run:
    python python_sandbox_server.py --host 127.0.0.1 --port 8000 --timeout 5

Safety knobs (sufficient for training; not a substitute for proper isolation):
  --timeout SEC          per-call wall clock (default 5)
  --memory-mb MB         per-call RSS via resource.RLIMIT_AS (default 1024)
  --cpu-sec SEC          per-call CPU time (default 5)
  --max-output-bytes N   truncate stdout/stderr (default 4096)
  --max-concurrent N     server-side semaphore (default 8)

For stronger isolation, wrap this whole server in docker / firejail.
"""

import argparse
import asyncio
import resource
import sys
import textwrap
from typing import List, Optional

# uvicorn/fastapi/pydantic are imported lazily inside build_app/main so the
# sandbox runner functions remain importable in environments that don't have
# them installed (e.g. unit tests of the subprocess runner).


# Code that runs inside the subprocess BEFORE the user snippet, to set rlimits
# and disable some unsafe operations. Kept minimal — real isolation should be
# at the container level.
_PRELUDE_TEMPLATE = """
import sys, os, signal, resource, builtins
# Quiet ubuntu's apport hook so blocked-import errors stay readable.
sys.excepthook = sys.__excepthook__
# Resource limits
resource.setrlimit(resource.RLIMIT_AS, ({mem_bytes}, {mem_bytes}))
resource.setrlimit(resource.RLIMIT_CPU, ({cpu_sec}, {cpu_sec}))
resource.setrlimit(resource.RLIMIT_NOFILE, (64, 64))
# Disable Python's __import__ for obvious foot-guns (best-effort only)
_BLOCKED = {{"subprocess", "socket", "ctypes", "multiprocessing", "_socket",
            "asyncio.subprocess", "smtplib", "requests", "urllib", "http",
            "ftplib", "telnetlib", "shutil"}}
_orig_import = builtins.__import__
def _safe_import(name, *a, **k):
    root = name.split('.')[0]
    if root in _BLOCKED or name in _BLOCKED:
        raise ImportError(f"module '{{name}}' is blocked in the sandbox")
    return _orig_import(name, *a, **k)
builtins.__import__ = _safe_import
"""


def _build_runner_source(user_code: str, mem_bytes: int, cpu_sec: int) -> str:
    prelude = _PRELUDE_TEMPLATE.format(mem_bytes=mem_bytes, cpu_sec=cpu_sec)
    return prelude + "\n# === user snippet starts here ===\n" + user_code


async def _run_one(code: str, timeout: float, mem_mb: int, cpu_sec: int,
                   max_output_bytes: int) -> str:
    src = _build_runner_source(code, mem_mb * 1024 * 1024, cpu_sec)
    try:
        proc = await asyncio.create_subprocess_exec(
            sys.executable, "-I", "-c", src,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
    except Exception as e:
        return f"[sandbox: failed to launch subprocess: {e}]"

    try:
        out, err = await asyncio.wait_for(proc.communicate(), timeout=timeout)
    except asyncio.TimeoutError:
        try:
            proc.kill()
            await proc.communicate()
        except Exception:
            pass
        return f"[sandbox: timeout after {timeout:.1f}s]"

    stdout = out.decode("utf-8", errors="replace") if out else ""
    stderr = err.decode("utf-8", errors="replace") if err else ""

    body = stdout
    if proc.returncode != 0:
        body = (stdout + "\n" if stdout else "") + "[stderr] " + stderr.strip()
    elif stderr.strip():
        body = stdout + "\n[stderr] " + stderr.strip()

    body = body.strip()
    if len(body.encode("utf-8")) > max_output_bytes:
        body = body.encode("utf-8")[:max_output_bytes].decode("utf-8", errors="replace") + "\n[sandbox: output truncated]"
    if not body:
        body = "[sandbox: no output produced]"
    return body


def build_app(args):
    from fastapi import FastAPI
    from pydantic import BaseModel

    class QueryRequest(BaseModel):
        queries: List[str]
        topk: Optional[int] = 1
        return_scores: Optional[bool] = True

    app = FastAPI(title="python-sandbox", version="0.1.0")
    sem = asyncio.Semaphore(args.max_concurrent)

    async def run_with_sem(code: str) -> str:
        async with sem:
            return await _run_one(
                code,
                timeout=args.timeout,
                mem_mb=args.memory_mb,
                cpu_sec=args.cpu_sec,
                max_output_bytes=args.max_output_bytes,
            )

    @app.get("/health")
    async def health():
        return {"status": "ok", "timeout": args.timeout, "memory_mb": args.memory_mb}

    @app.post("/retrieve")
    async def retrieve(req: QueryRequest):
        """Mirror ICRL's retrieval response shape; each query is interpreted as Python source."""
        outputs = await asyncio.gather(*(run_with_sem(q) for q in req.queries))
        # contents = "title\nbody"; ICRL splits on first '\n' to extract title and text.
        resp_items = []
        for out in outputs:
            contents = "stdout\n" + out
            doc = {"document": {"contents": contents}}
            if req.return_scores:
                doc["score"] = 1.0
            resp_items.append([doc])
        return {"result": resp_items}

    return app


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--timeout", type=float, default=5.0, help="per-call wall clock (s)")
    p.add_argument("--memory-mb", type=int, default=1024, help="per-call RSS limit (MB)")
    p.add_argument("--cpu-sec", type=int, default=5, help="per-call CPU time limit (s)")
    p.add_argument("--max-output-bytes", type=int, default=4096)
    p.add_argument("--max-concurrent", type=int, default=8)
    args = p.parse_args()

    print(textwrap.dedent(f"""
    Starting python-sandbox server at http://{args.host}:{args.port}
      timeout={args.timeout}s, memory={args.memory_mb}MB, cpu={args.cpu_sec}s
      max_output={args.max_output_bytes}B, max_concurrent={args.max_concurrent}
      endpoint: POST /retrieve  (compatible with ICRL retrieval client)
    """))
    import uvicorn
    uvicorn.run(build_app(args), host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
