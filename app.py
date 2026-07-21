"""Entrypoint. Run with `python app.py` or `flask run`."""

import sys
from pathlib import Path

# Make sure the project root is importable even when launched from elsewhere
# (e.g. `python "AI Chatbot/Chatbot/app.py"` from your home directory).
sys.path.insert(0, str(Path(__file__).resolve().parent))

from chatbot import Config, create_app  # noqa: E402

app = create_app()

if __name__ == "__main__":
    # debug is driven by FLASK_DEBUG in .env, and bound to localhost by
    # default. Never expose a debug server on 0.0.0.0 -- the Werkzeug
    # debugger console is remote code execution to anyone who can reach it.
    host = "127.0.0.1" if Config.DEBUG else "0.0.0.0"

    # Loud banner so you can tell at a glance whether the process you're
    # looking at is running THIS code or a stale one from a previous session.
    print("=" * 58)
    print("  AI Chatbot  |  build 2026-07-21  |  RAG + tools")
    print(f"  provider : {Config.provider_name()}  ({Config.MODEL_NAME})")
    print(f"  open     : http://127.0.0.1:{Config.PORT}")
    print("=" * 58, flush=True)

    # Keep the debug reloader away from directories we WRITE to at runtime.
    # Uploads land in docs/ and the index in vectorstore/ -- both inside the
    # project tree. With watchdog installed the reloader watches everything,
    # so an upload would restart the server mid-request and the browser would
    # report a bare "network error".
    reloader_excludes = [
        str(Config.DOCS_DIR / "*"),
        str(Path(Config.CHROMA_PATH).parent / "*"),
    ]

    run_kwargs = {
        "host": host,
        "port": Config.PORT,
        "debug": Config.DEBUG,
        "use_reloader": Config.USE_RELOADER,
    }
    if Config.USE_RELOADER:
        run_kwargs["exclude_patterns"] = reloader_excludes
    if Config.USE_RELOADER:
        print("  note: auto-reload is ON; uploads may interrupt if it triggers")

    try:
        app.run(**run_kwargs)
    except OSError as exc:
        if getattr(exc, "errno", None) in (48, 98, 10048):
            print()
            print(f"!! Port {Config.PORT} is already in use by another process.")
            print("!! That stale server is what your browser is talking to.")
            print("!! Kill it:   netstat -ano | findstr :%d" % Config.PORT)
            print("!!            taskkill /PID <pid> /F")
            print("!! Or run on a different port:  set PORT=5001 && python app.py")
            raise SystemExit(1)
        raise
