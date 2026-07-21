"""Live web access: search + page fetch.

Two capabilities, both free and key-less:
  search_web(query)  -- DuckDuckGo results via the `ddgs` library
  fetch_url(url)     -- download a page and extract readable text

SECURITY NOTE
-------------
The URL passed to fetch_url is chosen by the language model, which may in turn
be influenced by whatever a user typed or whatever text was on a previously
fetched page. That makes it untrusted input, so fetch_url deliberately refuses
to touch anything that isn't a public http(s) address -- see _assert_public_url.
Without those checks the tool is an SSRF hole: a model could be talked into
fetching http://127.0.0.1:5000/api/... or a cloud metadata endpoint.
"""

from __future__ import annotations

import ipaddress
import logging
import re
import socket
from html.parser import HTMLParser
from urllib.parse import urlparse

from .config import Config

logger = logging.getLogger(__name__)

try:
    import requests
except ImportError:  # pragma: no cover
    requests = None

try:
    from ddgs import DDGS
except ImportError:  # pragma: no cover
    try:
        from duckduckgo_search import DDGS  # older package name
    except ImportError:
        DDGS = None


USER_AGENT = "Mozilla/5.0 (compatible; AIChatbot/1.0; +local dev)"
ALLOWED_SCHEMES = {"http", "https"}
MAX_BYTES = 2_000_000        # stop reading after ~2MB
MAX_TEXT_CHARS = 12_000      # how much page text to hand the model
FETCH_TIMEOUT = 15


# ---------------------------------------------------------------------------
# URL safety
# ---------------------------------------------------------------------------

class UnsafeURL(ValueError):
    """Raised when a URL points somewhere the tool must not go."""


def _is_public_ip(host: str) -> bool:
    """Resolve a hostname and confirm every address it maps to is public."""
    try:
        infos = socket.getaddrinfo(host, None)
    except socket.gaierror as exc:
        raise UnsafeURL(f"could not resolve host '{host}': {exc}") from exc

    for info in infos:
        addr = info[4][0]
        try:
            ip = ipaddress.ip_address(addr)
        except ValueError:
            raise UnsafeURL(f"unparseable address for '{host}'")
        # Covers 127.0.0.0/8, 10/8, 172.16/12, 192.168/16, 169.254/16
        # (cloud metadata), ::1, fc00::/7, and friends.
        if (
            ip.is_private
            or ip.is_loopback
            or ip.is_link_local
            or ip.is_reserved
            or ip.is_multicast
            or ip.is_unspecified
        ):
            raise UnsafeURL(
                f"'{host}' resolves to non-public address {addr} -- refusing "
                "to fetch internal network resources"
            )
    return True


def _assert_public_url(url: str) -> str:
    parsed = urlparse(url.strip())

    if parsed.scheme.lower() not in ALLOWED_SCHEMES:
        raise UnsafeURL(
            f"scheme '{parsed.scheme or '(none)'}' is not allowed; "
            "only http and https"
        )
    if not parsed.hostname:
        raise UnsafeURL("URL has no hostname")

    _is_public_ip(parsed.hostname)
    return url.strip()


# ---------------------------------------------------------------------------
# HTML -> text
# ---------------------------------------------------------------------------

class _TextExtractor(HTMLParser):
    """Strip tags, keeping visible text. Avoids a BeautifulSoup dependency."""

    SKIP = {"script", "style", "noscript", "svg", "head", "nav", "footer", "form"}
    BREAK = {"p", "br", "div", "li", "tr", "h1", "h2", "h3", "h4", "h5", "h6"}

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._parts: list[str] = []
        self._skip_depth = 0
        self.title = ""
        self._in_title = False

    def handle_starttag(self, tag, attrs):
        if tag in self.SKIP:
            self._skip_depth += 1
        elif tag == "title":
            self._in_title = True
        elif tag in self.BREAK:
            self._parts.append("\n")

    def handle_endtag(self, tag):
        if tag in self.SKIP and self._skip_depth:
            self._skip_depth -= 1
        elif tag == "title":
            self._in_title = False

    def handle_data(self, data):
        if self._in_title and not self.title:
            self.title = data.strip()
        if self._skip_depth == 0:
            text = data.strip()
            if text:
                self._parts.append(text)

    def text(self) -> str:
        raw = " ".join(self._parts)
        raw = re.sub(r"[ \t]+", " ", raw)
        raw = re.sub(r"\n\s*\n+", "\n\n", raw)
        return raw.strip()


def html_to_text(html: str) -> tuple[str, str]:
    """Return (title, readable_text)."""
    p = _TextExtractor()
    try:
        p.feed(html)
    except Exception:  # pragma: no cover - malformed HTML
        pass
    return p.title, p.text()


# ---------------------------------------------------------------------------
# Public operations
# ---------------------------------------------------------------------------

def search_web(query: str, max_results: int = 5) -> list[dict]:
    """DuckDuckGo search. Returns [{title, url, snippet}]."""
    if DDGS is None:
        raise RuntimeError("ddgs is not installed. Run: pip install ddgs")

    max_results = max(1, min(int(max_results), 10))
    with DDGS() as ddgs:
        raw = list(ddgs.text(query, max_results=max_results))

    out = []
    for r in raw:
        out.append(
            {
                "title": (r.get("title") or "").strip(),
                "url": (r.get("href") or r.get("url") or "").strip(),
                "snippet": (r.get("body") or r.get("snippet") or "").strip(),
            }
        )
    return out


def fetch_url(url: str, max_chars: int = MAX_TEXT_CHARS) -> dict:
    """Fetch a public web page and return {url, title, text, truncated}."""
    if requests is None:
        raise RuntimeError("requests is not installed. Run: pip install requests")

    safe = _assert_public_url(url)

    resp = requests.get(
        safe,
        headers={"User-Agent": USER_AGENT, "Accept": "text/html,text/plain,*/*"},
        timeout=FETCH_TIMEOUT,
        allow_redirects=True,
        stream=True,
    )

    # Redirects are followed by requests, so re-check where we actually landed.
    _assert_public_url(resp.url)
    resp.raise_for_status()

    ctype = (resp.headers.get("Content-Type") or "").lower()
    if not any(t in ctype for t in ("text/html", "text/plain", "json", "xml", "")):
        raise ValueError(f"unsupported content type '{ctype}'")

    chunks, total = [], 0
    for chunk in resp.iter_content(chunk_size=16384, decode_unicode=False):
        if not chunk:
            continue
        chunks.append(chunk)
        total += len(chunk)
        if total >= MAX_BYTES:
            break
    resp.close()

    encoding = resp.encoding or "utf-8"
    body = b"".join(chunks).decode(encoding, errors="replace")

    if "text/html" in ctype:
        title, text = html_to_text(body)
    else:
        title, text = "", body.strip()

    truncated = len(text) > max_chars
    return {
        "url": resp.url,
        "title": title,
        "text": text[:max_chars],
        "truncated": truncated,
    }
