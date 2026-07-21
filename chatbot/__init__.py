"""Flask application factory for the RAG chatbot."""

import logging

from flask import Flask

from .config import Config


def create_app(config_object=Config) -> Flask:
    # Without this, our logger.info() calls are swallowed when debug is off,
    # so the terminal shows nothing useful while diagnosing an upload.
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)-8s %(name)s: %(message)s",
    )
    # These two are chatty at INFO and drown out our own messages.
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)

    app = Flask(__name__, template_folder="../templates")
    app.config.from_object(config_object)
    # Re-read templates from disk on every request. Without this, Jinja caches
    # the compiled template in memory and a long-running server keeps serving
    # a stale page even after you've edited the file.
    app.config["TEMPLATES_AUTO_RELOAD"] = True
    app.jinja_env.auto_reload = True

    # Deliberately set well ABOVE the per-file limit enforced in documents.py.
    #
    # When a request exceeds MAX_CONTENT_LENGTH, Werkzeug aborts and closes the
    # connection without reading the body. The browser never receives a
    # response, so XHR reports a bare "network error" and the user has no idea
    # what went wrong. Letting the body through and validating per-file means
    # we can return a real JSON message naming the file and its size.
    from .documents import MAX_UPLOAD_BYTES

    app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_BYTES * 8

    # Imported here (not at module top) so that importing `chatbot` never
    # triggers OpenAI/Chroma client construction as an import side effect.
    from .routes import chatbot_bp

    app.register_blueprint(chatbot_bp)
    return app


__all__ = ["create_app", "Config"]
