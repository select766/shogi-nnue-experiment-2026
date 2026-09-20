"""Read-only failed-chunk review; no engines and no strength statistics."""
import json
from pathlib import Path
import cshogi
from train_nnue.decision_replication_formal import check, reusable
from train_nnue.decision_replication_audit import audit_pair, load

RUN = Path(".research-loop/experiments/decision-replication-formal-floodgate2025-chunk001/attempt-001").resolve()
REPORT = Path(__file__).parent
chunk, rows, protocol = check(RUN)
libs = load(RUN / "libraries.json")
out = Path(load(RUN / "artifacts/latest.json")["directory"])
markers = sorted((RUN / "artifacts").glob("pair-*.json"))
assert [load(p)["pair_id"] for p in markers] == list(range(1,64))
entries = [reusable(RUN, i, rows[i-1], protocol, libs) for i in range(1,64)]
assert all(entries)
assert reusable(RUN, 64, rows[63], protocol, libs) is None
try:
    audit_pair(out / "pair-00064", rows[63], protocol, libs)
except AssertionError:
    pass
else:
    raise AssertionError("Expected original pair64 audit failure")
records = load(out / "pair-00064/records.json")
assert len(records) == 2 and all(r["complete"] for r in records)
zeros = []
for r in records:
    b = cshogi.Board(rows[63]["initial_sfen"])
    for token in rows[63]["history_usi"]:
        b.push_usi(token)
    for i,s in enumerate(r["trace"]["searches"]):
        if s["nodes"] == 0:
            assert s["bestmove"] == "win" and b.is_nyugyoku()
            zeros.append(dict(candidate_color=r["candidate_color"], search_index=i,
                              sfen=b.sfen(), legal_entering_king=True, info=s["info"]))
        if s["bestmove"] not in ("win", "resign"):
            move=b.move_from_usi(s["bestmove"])
            assert move and b.is_legal(move)
            b.push(move)
assert len(zeros) == 1
result = dict(status="incomplete", planned_pairs=166, accepted_pairs=63,
              accepted_games=126, stored_complete_games=128,
              accepted_searches=sum(e["audit"]["searches"] for e in entries),
              original_pair64_audit="reproduced AssertionError", zero_node_declarations=zeros,
              strength_inference="not performed", input_hashes_and_separation="passed")
(REPORT / "verification.json").write_text(json.dumps(result, ensure_ascii=False, indent=2)+"\n")
print("Verified 63 complete pairs; reproduced pair64 audit failure; legal zero-node entering-king declaration confirmed")
