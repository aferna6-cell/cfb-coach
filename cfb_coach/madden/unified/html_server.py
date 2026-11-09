"""Serve the ESTABLISHED live HTML page with Unified V1 behind it.

Only ``make_handler`` / ``render_live_html`` / ``pick_port`` / ``open_url`` are reused from ``cfb_coach.live_server``;
the legacy ``LivePlayController`` (which opens ``CoachDB``) and ``run_live_server`` (session arming) are NOT used.
"""
from __future__ import annotations

import hmac
import json
import secrets
import threading
from pathlib import Path
from typing import Any

from cfb_coach.live_server import ContextThreadingHTTPServer, make_handler, pick_port, render_live_html
from cfb_coach.madden.unified import guard
from cfb_coach.madden.unified.html_controller import UnifiedLiveController
from cfb_coach.madden.unified.session import UnifiedSession
from cfb_coach.madden.unified.store import UnifiedStore
from cfb_coach.madden.unified.world_model import WorldModel


def build_session(*, confirmed_books: str, store_path: str, opponent: str, model: str | None = None, game_id: str | None = None,
                  resume: bool = False, opponent_class: str | None = None, verified_roles: str | None = None,
                  adjustment_confirmations: str | None = None) -> tuple[UnifiedSession, str]:
    from cfb_coach.madden.unified.preflight import load_confirmed_books

    raw = load_confirmed_books(confirmed_books)
    books = {s: {"inventory": b["inventory"], "fingerprint": b["fingerprint"]} for s, b in raw.items()}
    wm = WorldModel.from_dict(json.loads(Path(model).read_text())) if model else None
    art = f"{Path(model).name}:{guard.sha256_file(Path(model))[:10]}" if model else "cold_start_prior_v1"
    conf = json.loads(Path(adjustment_confirmations).read_text()) if adjustment_confirmations else {}
    store = UnifiedStore(store_path)
    if game_id is None and not resume:
        unfinished = store.unfinished_games()
        if unfinished:
            print(f"NOTE: unfinished game(s) in this store: {', '.join(unfinished)}. To continue one without losing anything: "
                  f"add --unified-game-id <id> --unified-resume")
    s = UnifiedSession(books=books, opponent_id=opponent, store=store, model=wm, artifact=art, game_id=game_id, resume=resume,
                       opponent_class=opponent_class, roles_file=verified_roles, adjustment_confirmations=conf)
    return s, art


def make_guarded_handler(ctrl: UnifiedLiveController, cfg: dict[str, Any]):
    """The established handler plus protection for a state-changing localhost API.

    * Host must be 127.0.0.1:<port> or localhost:<port> (blocks DNS rebinding);
    * a present Origin must be exactly this server's origin (blocks cross-site form/fetch posts);
    * POST needs ``Content-Type: application/json`` (a cross-site form cannot send it) and the per-process secret token that is
      embedded in the page we serve and echoed in ``X-Unified-Token``.
    """
    base = make_handler(ctrl)

    class Guarded(base):  # type: ignore[misc, valid-type]
        def _host_ok(self) -> bool:
            return (self.headers.get("Host") or "").lower() in cfg["hosts"]

        def _deny(self, why: str) -> None:
            self._json(403, {"ok": False, "error": f"forbidden: {why}"})

        def do_GET(self) -> None:  # noqa: N802
            if not self._host_ok():
                return self._deny("unexpected Host header")
            if self.path.split("?")[0] in ("/", "/index.html"):
                html = render_live_html(ctrl).replace("<script>", '<script>window.UNIFIED_TOKEN=' + json.dumps(cfg["token"]) + ";", 1)
                return self._send(200, html.encode("utf-8"), "text/html; charset=utf-8")
            return super().do_GET()

        def do_POST(self) -> None:  # noqa: N802
            if not self._host_ok():
                return self._deny("unexpected Host header")
            origin = self.headers.get("Origin")
            if origin and origin.lower() not in {f"http://{h}" for h in cfg["hosts"]}:
                return self._deny("cross-origin request")
            if not (self.headers.get("Content-Type") or "").lower().startswith("application/json"):
                return self._deny("JSON content type required")
            if not hmac.compare_digest(self.headers.get("X-Unified-Token") or "", cfg["token"]):
                return self._deny("missing or wrong session token")
            return super().do_POST()

    return Guarded


def start_server(ctrl: UnifiedLiveController, *, host: str = "127.0.0.1", port: int | None = None) -> tuple[ContextThreadingHTTPServer, threading.Thread, str]:
    port = port or pick_port(host)
    cfg: dict[str, Any] = {"hosts": set(), "token": secrets.token_urlsafe(24)}
    server = ContextThreadingHTTPServer((host, port), make_guarded_handler(ctrl, cfg))
    real = server.server_address[1]
    cfg["hosts"] = {f"127.0.0.1:{real}", f"localhost:{real}"}
    server.unified_token = cfg["token"]  # type: ignore[attr-defined]
    th = threading.Thread(target=server.serve_forever, daemon=True, name="unified-html")
    th.start()
    return server, th, f"http://{host}:{real}/"


def run(*, open_browser: bool = True, port: int | None = None, model: str | None = None, **kw: Any) -> int:
    from cfb_coach.browser_open import open_url

    session, art = build_session(model=model, **kw)
    ctrl = UnifiedLiveController(session, model_path=model)
    server, th, url = start_server(ctrl, port=port)
    print(f"UNIFIED V1 · HTML live play ON → {url}")
    print(f"  game {session.game_id} · artifact {art} · both coordinators active · next snap {session.game_id}-{session.seq:04d}")
    if session.pending is not None:
        print("  RESUMED: the last snap is still waiting for its result — it is shown in the window.")
    print("  Same window as always. Ctrl+C stops the server; the game stays open and can be resumed with --unified-resume.")
    if open_browser:
        open_url(url)
    try:
        th.join()
    except KeyboardInterrupt:
        print("\nStopping unified live server…")
    finally:
        server.shutdown()
        server.server_close()
        session.store.close()
    return 0
