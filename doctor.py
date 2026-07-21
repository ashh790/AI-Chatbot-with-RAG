"""Diagnose a broken setup. Run from the Chatbot folder:

    python doctor.py

Checks dependencies, config, ports, imports, and the request path, then
tells you exactly what to fix.
"""

from __future__ import annotations

import importlib.util
import os
import socket
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

problems: list[str] = []
warnings: list[str] = []


def ok(msg: str) -> None:
    print(f"  [ OK ] {msg}")


def bad(msg: str, fix: str) -> None:
    print(f"  [FAIL] {msg}")
    print(f"         fix: {fix}")
    problems.append(msg)


def warn(msg: str, note: str = "") -> None:
    print(f"  [WARN] {msg}")
    if note:
        print(f"         {note}")
    warnings.append(msg)


def section(title: str) -> None:
    print(f"\n{title}")
    print("-" * len(title))


# ---------------------------------------------------------------- 1. python
section("1. Python")
v = sys.version_info
if v < (3, 10):
    bad(f"Python {v.major}.{v.minor} is too old", "install Python 3.10 or newer")
else:
    ok(f"Python {v.major}.{v.minor}.{v.micro}")
print(f"         interpreter: {sys.executable}")

in_venv = sys.prefix != sys.base_prefix
if in_venv:
    ok("running inside a virtual environment")
else:
    warn(
        "not running in a virtual environment",
        "if you installed deps into a venv, activate it first: venv\\Scripts\\activate",
    )

# ---------------------------------------------------------- 2. project files
section("2. Project files")
expected = {
    "app.py": HERE / "app.py",
    "chatbot/__init__.py": HERE / "chatbot" / "__init__.py",
    "chatbot/routes.py": HERE / "chatbot" / "routes.py",
    "chatbot/config.py": HERE / "chatbot" / "config.py",
    "templates/index.html": HERE / "templates" / "index.html",
    "requirements.txt": HERE / "requirements.txt",
}
for label, path in expected.items():
    if path.exists():
        ok(label)
    else:
        bad(f"missing {label}", f"expected at {path}")

stray = HERE / "config.py"
if stray.exists():
    warn(
        "a stray top-level config.py exists",
        "config now lives in chatbot/config.py -- delete the old one to avoid shadowing",
    )

# ------------------------------------------------------------ 3. dependencies
section("3. Dependencies")
for mod, pip_name, required in [
    ("flask", "flask", True),
    ("dotenv", "python-dotenv", True),
    ("openai", "openai", True),
    ("chromadb", "chromadb", False),
    ("pypdf", "pypdf", False),
]:
    if importlib.util.find_spec(mod) is not None:
        ok(f"{pip_name} installed")
    elif required:
        bad(f"{pip_name} NOT installed", f"pip install {pip_name}")
    else:
        warn(f"{pip_name} not installed", f"optional -- pip install {pip_name}")

# ------------------------------------------------------------------ 4. config
section("4. Configuration")
env_file = HERE / ".env"
if env_file.exists():
    ok(".env found")
else:
    bad(".env missing", "copy .env.example to .env and add your key")

try:
    from chatbot.config import Config

    ok("config loads")
    if Config.has_valid_api_key():
        ok(f"OPENAI_API_KEY looks valid ({Config.OPENAI_API_KEY[:7]}...)")
    else:
        warn(
            "no valid OPENAI_API_KEY -- app runs in demo mode",
            "the server still starts; replies will just echo your input",
        )
    print(f"         model:  {Config.MODEL_NAME}")
    print(f"         debug:  {Config.DEBUG}")
    print(f"         chroma: {Config.CHROMA_PATH}")
except Exception as exc:
    bad(f"config failed to load: {exc}", "check .env formatting -- no quotes needed around values")

# ------------------------------------------------------------------- 5. ports
section("5. Port 5000")


def port_in_use(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(1.0)
        return s.connect_ex((host, port)) == 0


if port_in_use("127.0.0.1", 5000):
    warn(
        "something is ALREADY listening on port 5000",
        "either your server is already running, or another app holds the port.\n"
        "         Windows: netstat -ano | findstr :5000   then  taskkill /PID <pid> /F\n"
        "         Or change the port in app.py.",
    )
else:
    ok("port 5000 is free")

# ----------------------------------------------------------------- 6. imports
section("6. Application import")
try:
    from chatbot import create_app

    ok("chatbot package imports")
    app = create_app()
    ok("create_app() succeeded")

    routes = sorted(str(r.rule) for r in app.url_map.iter_rules())
    ok(f"{len(routes)} routes registered: {', '.join(routes)}")

    tpl = Path(app.jinja_loader.searchpath[0]) / "index.html"
    if tpl.exists():
        ok(f"template resolves -> {tpl}")
    else:
        bad(f"template NOT found at {tpl}", "templates/index.html must sit next to app.py")
except Exception as exc:
    import traceback

    bad(f"import failed: {exc}", "full traceback below")
    traceback.print_exc()
    app = None

# ------------------------------------------------------------ 7. request path
section("7. Request handling (in-process, no network)")
if app is not None:
    try:
        app.config.update(TESTING=True)
        c = app.test_client()

        r = c.get("/")
        (ok if r.status_code == 200 else lambda m: bad(m, "check templates/"))(
            f"GET /            -> {r.status_code}"
        )

        r = c.get("/api/health")
        if r.status_code == 200:
            ok(f"GET /api/health  -> 200")
            h = r.get_json()
            for k in ("demo_mode", "vector_store_available", "documents_indexed"):
                print(f"         {k}: {h.get(k)}")
            if h.get("documents_indexed") == 0:
                warn("no documents indexed", "run: python ingest.py")
        else:
            bad(f"GET /api/health -> {r.status_code}", "see traceback above")

        r = c.post("/api/chat", json={"message": "ping", "session_id": "doctor"})
        if r.status_code == 200:
            ok(f"POST /api/chat   -> 200")
            print(f"         reply: {r.get_json()['response'][:60]}")
        else:
            bad(f"POST /api/chat -> {r.status_code}: {r.get_json()}", "see error above")
    except Exception as exc:
        import traceback

        bad(f"request handling crashed: {exc}", "traceback below")
        traceback.print_exc()
else:
    print("  skipped -- app did not import")

# ----------------------------------------------------------------- 8. summary
section("Summary")
if problems:
    print(f"  {len(problems)} problem(s) blocking startup:\n")
    for p in problems:
        print(f"    - {p}")
    print("\n  Fix those, then re-run: python doctor.py")
else:
    print("  No blocking problems. The app works in-process.")
    print()
    print("  If the browser still says 'server unreachable', the server isn't")
    print("  actually running or you're not viewing it through Flask:")
    print()
    print("    1. Start it:  python app.py")
    print("    2. Leave that terminal OPEN -- closing it kills the server")
    print("    3. Open http://127.0.0.1:5000 in your browser")
    print()
    print("  Do NOT double-click templates/index.html. Opening it as a file")
    print("  (file:///...) means the page has no server to talk to, and every")
    print("  fetch fails with exactly that 'server unreachable' message.")

if warnings:
    print(f"\n  {len(warnings)} warning(s) -- not fatal:")
    for w in warnings:
        print(f"    - {w}")

sys.exit(1 if problems else 0)
