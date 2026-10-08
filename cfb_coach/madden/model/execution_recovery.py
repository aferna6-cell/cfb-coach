"""One-time auditable recovery for experimental game 141d4a16b4184ee7.

Laptop-only. Never invents outcomes for unlinked decisions. Requires explicit
user attestation and ``--apply`` before mutating. Idempotent.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from cfb_coach.madden.model import dataset as dataset_mod
from cfb_coach.madden.model.schema import ExecutedStatus, Verification

# Hard-locked to the first experimental CPU Lions game. Do not broaden.
RECOVERY_GAME_ID = "141d4a16b4184ee7"
RECOVERY_META_KEY = "ml_execution_recovery_141d4a16b4184ee7"
ATTESTATION_PHRASE = (
    "I followed the final displayed recommendation on every play in game "
    "141d4a16b4184ee7, including Texas Y-Stutter Wheel on snap 0010"
)


def _row_get(row: Any, key: str, default: Any = None) -> Any:
    if row is None:
        return default
    try:
        if hasattr(row, "keys"):
            return row[key] if key in row.keys() else default
    except Exception:  # noqa: BLE001
        pass
    try:
        return row[key]
    except Exception:  # noqa: BLE001
        return default


def _outcome_payload(row: Any) -> dict[str, Any]:
    raw = _row_get(row, "outcome_json")
    if not raw:
        return {}
    try:
        payload = json.loads(raw) if isinstance(raw, str) else dict(raw)
        return dict(payload or {})
    except (TypeError, ValueError, json.JSONDecodeError):
        return {}


def _is_verified(status: str | None, verification: str | None) -> bool:
    return (
        str(status or "").lower() == ExecutedStatus.IDENTIFIED.value
        and str(verification or "").lower() == Verification.VERIFIED.value
    )


def _counts(db: Any, *, game_id: str = RECOVERY_GAME_ID) -> dict[str, Any]:
    decisions = list(
        db.conn.execute(
            "SELECT * FROM ml_decisions WHERE game_id = ? OR session_id = ? ORDER BY id ASC",
            (game_id, game_id),
        )
    )
    outcomes_by_snap: dict[str, Any] = {}
    for o in db.conn.execute(
        "SELECT * FROM ml_outcomes WHERE game_id = ? OR snap_id LIKE ? ORDER BY id ASC",
        (game_id, f"{game_id}-%"),
    ):
        sid = str(_row_get(o, "snap_id") or "")
        if sid:
            outcomes_by_snap[sid] = o

    verified = 0
    linked = 0
    for d in decisions:
        sid = str(_row_get(d, "snap_id") or "")
        outc = outcomes_by_snap.get(sid)
        if outc is None:
            continue
        linked += 1
        if _is_verified(
            _row_get(outc, "executed_status"),
            _row_get(outc, "executed_verification"),
        ):
            verified += 1

    rows = dataset_mod.build_rows(db=db)
    game_rows = [
        r
        for r in rows
        if str(r.get("game_id") or r.get("session_id") or "") == game_id
        or str(r.get("snap_id") or "").startswith(f"{game_id}-")
    ]
    supervised = [
        r
        for r in game_rows
        if (r.get("eligibility") or dataset_mod.classify_eligibility(r))
        in dataset_mod.SUPERVISED_ELIGIBLE
    ]
    quality = dataset_mod.quality_report(rows)
    return {
        "game_id": game_id,
        "n_decisions": len(decisions),
        "n_linked_outcomes": linked,
        "n_verified_executions": verified,
        "n_game_training_rows": len(game_rows),
        "n_game_supervised_rows": len(supervised),
        "db_supervised_training_rows": quality.get("supervised_training_rows"),
        "db_verified_executions": quality.get("verified_executions"),
    }


def preview_recovery(
    db: Any,
    *,
    game_id: str = RECOVERY_GAME_ID,
    attestation: str | None = None,
) -> dict[str, Any]:
    """Dry-run plan: what would be recovered / skipped. No writes."""
    if game_id != RECOVERY_GAME_ID:
        return {
            "ok": False,
            "error": (
                f"Recovery is locked to game {RECOVERY_GAME_ID!r}; "
                f"refusing {game_id!r}"
            ),
            "would_recover": [],
            "skipped": [],
            "unlinked": [],
        }

    attest_ok = _attestation_ok(attestation)
    before = _counts(db, game_id=game_id)
    decisions = list(
        db.conn.execute(
            "SELECT * FROM ml_decisions WHERE game_id = ? OR session_id = ? ORDER BY id ASC",
            (game_id, game_id),
        )
    )
    outcomes_by_snap: dict[str, Any] = {}
    for o in db.conn.execute(
        "SELECT * FROM ml_outcomes WHERE game_id = ? OR snap_id LIKE ? ORDER BY id ASC",
        (game_id, f"{game_id}-%"),
    ):
        sid = str(_row_get(o, "snap_id") or "")
        if sid:
            outcomes_by_snap[sid] = o

    would_recover: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    unlinked: list[dict[str, Any]] = []
    already_ok: list[dict[str, Any]] = []

    for d in decisions:
        sid = str(_row_get(d, "snap_id") or "")
        final_form = _row_get(d, "final_formation")
        final_play = _row_get(d, "final_play")
        decision_id = _row_get(d, "id")
        entry = {
            "snap_id": sid,
            "decision_id": decision_id,
            "final_formation": final_form,
            "final_play": final_play,
            "agree": _row_get(d, "agree"),
        }
        if not sid:
            skipped.append({**entry, "reason": "missing snap_id"})
            continue
        if not final_form or not final_play:
            skipped.append({**entry, "reason": "missing final displayed formation/play"})
            continue
        outc = outcomes_by_snap.get(sid)
        if outc is None:
            unlinked.append(
                {
                    **entry,
                    "reason": (
                        "no linked outcome — never invent results for unlinked decisions"
                    ),
                }
            )
            continue

        status = str(_row_get(outc, "executed_status") or "unknown").lower()
        ver = str(_row_get(outc, "executed_verification") or "unknown").lower()
        exec_form = _row_get(outc, "executed_formation")
        exec_play = _row_get(outc, "executed_play")
        payload = _outcome_payload(outc)

        if _is_verified(status, ver):
            if (
                str(exec_form or "") == str(final_form)
                and str(exec_play or "") == str(final_play)
            ):
                already_ok.append(
                    {
                        **entry,
                        "reason": "already verified as final displayed call (idempotent)",
                    }
                )
                continue
            skipped.append(
                {
                    **entry,
                    "reason": (
                        "conflicting verified execution "
                        f"{exec_form!r}/{exec_play!r} ≠ final {final_form!r}/{final_play!r}"
                    ),
                    "existing_executed": f"{exec_form}/{exec_play}",
                }
            )
            continue

        # Ambiguous: identified but unverified with a different play named.
        if (
            status == ExecutedStatus.IDENTIFIED.value
            and exec_play
            and str(exec_play) != str(final_play)
        ):
            skipped.append(
                {
                    **entry,
                    "reason": (
                        "ambiguous identified-but-unverified execution "
                        f"{exec_form!r}/{exec_play!r} ≠ final {final_play!r}"
                    ),
                }
            )
            continue

        would_recover.append(
            {
                **entry,
                "outcome_id": _row_get(outc, "id"),
                "prior_executed_status": status,
                "prior_executed_verification": ver,
                "prior_executed": f"{exec_form}/{exec_play}" if exec_play else None,
                "preserved_result": payload.get("result") or _snap_result(db, sid),
                "preserved_decision_id": _row_get(outc, "decision_id") or decision_id,
                "action": "set executed = final displayed recommendation",
            }
        )

    return {
        "ok": True,
        "dry_run": True,
        "game_id": game_id,
        "attestation_accepted": attest_ok,
        "attestation_required": ATTESTATION_PHRASE,
        "before": before,
        "would_recover": would_recover,
        "already_verified_ok": already_ok,
        "skipped": skipped,
        "unlinked": unlinked,
        "n_would_recover": len(would_recover),
        "n_already_ok": len(already_ok),
        "n_skipped": len(skipped),
        "n_unlinked": len(unlinked),
        "note": (
            "Dry-run only. Pass matching --attest and --apply to mutate. "
            "Other Franchise games are untouched. Unlinked decisions stay unlinked."
        ),
    }


def _snap_result(db: Any, snap_id: str) -> str | None:
    try:
        row = db.conn.execute(
            "SELECT result FROM snaps WHERE ml_snap_id = ? ORDER BY id DESC LIMIT 1",
            (snap_id,),
        ).fetchone()
        if row is None:
            return None
        return _row_get(row, "result")
    except Exception:  # noqa: BLE001
        return None


def _attestation_ok(attestation: str | None) -> bool:
    text = " ".join((attestation or "").strip().lower().split())
    if not text:
        return False
    # Accept the canonical phrase or a clearly equivalent confirmation naming the game.
    if text == ATTESTATION_PHRASE.lower():
        return True
    need = (
        "141d4a16b4184ee7" in text
        and "followed" in text
        and "final displayed recommendation" in text
        and "every play" in text
    )
    return need


def apply_recovery(
    db: Any,
    *,
    game_id: str = RECOVERY_GAME_ID,
    attestation: str,
    backup_path: Path | str | None = None,
    apply: bool = False,
) -> dict[str, Any]:
    """Preview or apply recovery. Mutates only when ``apply`` and attestation pass."""
    plan = preview_recovery(db, game_id=game_id, attestation=attestation)
    if not plan.get("ok"):
        return plan

    if game_id != RECOVERY_GAME_ID:
        plan["ok"] = False
        plan["error"] = f"locked to {RECOVERY_GAME_ID}"
        return plan

    if not apply:
        return plan

    if not _attestation_ok(attestation):
        return {
            **plan,
            "ok": False,
            "dry_run": False,
            "applied": False,
            "error": (
                "Refusing to apply without explicit attestation matching the "
                f"required confirmation for game {RECOVERY_GAME_ID}."
            ),
            "attestation_required": ATTESTATION_PHRASE,
        }

    # Backup before mutation.
    backup_dest: Path | None = None
    if backup_path is not None:
        backup_dest = Path(backup_path)
    else:
        src = Path(getattr(db, "path", ".") or ".")
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        backup_dest = src.parent / "backups" / f"{src.stem}.pre-recovery-{RECOVERY_GAME_ID}.{stamp}{src.suffix}"
    backup_dest.parent.mkdir(parents=True, exist_ok=True)
    db.backup_to(backup_dest)

    recovered: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    for item in plan["would_recover"]:
        try:
            result = _recover_one(db, item, attestation=attestation)
            recovered.append(result)
        except Exception as exc:  # noqa: BLE001
            errors.append({"snap_id": item.get("snap_id"), "error": str(exc)})

    after = _counts(db, game_id=game_id)
    meta = {
        "recovered_at": datetime.now(timezone.utc).isoformat(),
        "game_id": game_id,
        "n_recovered": len(recovered),
        "backup": str(backup_dest),
        "attestation": attestation.strip()[:400],
        "snap_ids": [r.get("snap_id") for r in recovered],
    }
    try:
        db.set_meta(RECOVERY_META_KEY, json.dumps(meta))
    except Exception:  # noqa: BLE001
        pass

    return {
        "ok": len(errors) == 0,
        "dry_run": False,
        "applied": True,
        "game_id": game_id,
        "backup": str(backup_dest),
        "before": plan["before"],
        "after": after,
        "recovered": recovered,
        "already_verified_ok": plan["already_verified_ok"],
        "skipped": plan["skipped"],
        "unlinked": plan["unlinked"],
        "errors": errors,
        "delta_verified": after["n_verified_executions"] - plan["before"]["n_verified_executions"],
        "delta_supervised_game": (
            after["n_game_supervised_rows"] - plan["before"]["n_game_supervised_rows"]
        ),
        "note": (
            "Recovery updated existing linked outcomes only. "
            "Unlinked decisions were not invented. Other games untouched."
        ),
    }


def _recover_one(db: Any, item: Mapping[str, Any], *, attestation: str) -> dict[str, Any]:
    snap_id = str(item["snap_id"])
    final_form = str(item["final_formation"])
    final_play = str(item["final_play"])
    decision_id = item.get("preserved_decision_id") or item.get("decision_id")

    outc = db.conn.execute(
        "SELECT * FROM ml_outcomes WHERE snap_id = ? ORDER BY id DESC LIMIT 1",
        (snap_id,),
    ).fetchone()
    if outc is None:
        raise LookupError(f"outcome disappeared for {snap_id}")

    payload = _outcome_payload(outc)
    preserved_result = item.get("preserved_result") or payload.get("result")
    # Preserve existing outcome metadata; annotate recovery.
    new_payload = dict(payload)
    if preserved_result is not None:
        new_payload["result"] = preserved_result
    new_payload["recovery"] = {
        "kind": "sprint42_execution_default_recovery",
        "game_id": RECOVERY_GAME_ID,
        "source": "final_displayed_recommendation",
        "attestation": attestation.strip()[:400],
        "recovered_at": datetime.now(timezone.utc).isoformat(),
        "prior_executed_status": item.get("prior_executed_status"),
        "prior_executed_verification": item.get("prior_executed_verification"),
        "prior_executed": item.get("prior_executed"),
    }

    # Update snap row when present (preserve notes/result; only execution fields).
    snap = db.conn.execute(
        "SELECT id, notes, result FROM snaps WHERE ml_snap_id = ? ORDER BY id DESC LIMIT 1",
        (snap_id,),
    ).fetchone()
    if snap is not None:
        notes = _row_get(snap, "notes") or ""
        tag = f"[recovery {RECOVERY_GAME_ID}: executed=final displayed]"
        if tag not in notes:
            notes = (notes + " " + tag).strip() if notes else tag
        db.update_snap(
            int(_row_get(snap, "id")),
            executed_status=ExecutedStatus.IDENTIFIED.value,
            executed_formation=final_form,
            executed_play=final_play,
            executed_verification=Verification.VERIFIED.value,
            notes=notes,
            # result unchanged — update_snap only touches provided fields
        )

    db.log_ml_outcome(
        snap_id=snap_id,
        game_id=RECOVERY_GAME_ID,
        decision_id=int(decision_id) if decision_id is not None else None,
        executed_status=ExecutedStatus.IDENTIFIED.value,
        executed_formation=final_form,
        executed_play=final_play,
        executed_verification=Verification.VERIFIED.value,
        outcome=new_payload,
        replace=True,
    )
    return {
        "snap_id": snap_id,
        "decision_id": decision_id,
        "executed_formation": final_form,
        "executed_play": final_play,
        "preserved_result": preserved_result,
        "status": "recovered",
    }
