"""Standalone welfare-panel HTTP route.

Drop-in: register the blueprint once in `app.py` (`app.register_blueprint(welfare_bp)`)
and the frontend can hit `GET /api/welfare_panel?lookup_key=<key>` to fetch the
slide cards. This module does NOT touch any other request path; if it fails for
any reason, no other handler in the app is affected.
"""

from __future__ import annotations

from flask import Blueprint, jsonify, request

welfare_bp = Blueprint("welfare", __name__)


@welfare_bp.get("/api/welfare_panel")
def get_welfare_panel_route():
    # In query strings, `+` decodes to a space — restore `+` so callers can
    # send the canonical token-joined form without URL-encoding manually.
    lookup_key = (request.args.get("lookup_key") or "").replace(" ", "+").strip()
    if not lookup_key:
        return jsonify({"welfare_panel": [], "lookup_key": ""}), 200

    try:
        from inference.welfare_panel import panel_for_lookup_key
    except Exception as exc:
        print(f"[welfare_panel] module unavailable: {exc}")
        return jsonify({"welfare_panel": [], "lookup_key": lookup_key}), 200

    try:
        panel = panel_for_lookup_key(lookup_key) or []
    except Exception as exc:
        print(f"[welfare_panel] call failed for {lookup_key!r}: {exc}")
        panel = []

    return jsonify({
        "welfare_panel": panel,
        "lookup_key": lookup_key,
    }), 200
