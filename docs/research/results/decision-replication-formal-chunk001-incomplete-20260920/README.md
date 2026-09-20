# 正式chunk001 attempt-001: 監査失敗による未完了

2026-09-20、H-DECISION-REPLICATION。job `decision-replication-formal-floodgate2025-chunk001` は **replan**。
計算は実行されたが、予定166ペア332局の完遂条件を満たさない。棋力の支持・棄却は行わず保留。

## 実行と原因

execution.jsonはexit_code=1、reason=exited、2777.252秒。timeoutではない。
固定順1〜63の63ペア126局に受理markerがあり、64番目は両局のcomplete記録があるが未受理。
保存上128局、正式に受理した126局を区別する。65〜166は未実行。
64番目・候補先手局の233探索目で、controlが合法な入玉宣言を返した。
通信には `score mate 1 nodes 0` と `bestmove win`、blendとclear=1がある。
`decision_replication_audit.py`の`nodes > 0`が拒否し停止した。
全履歴を再生して`cshogi.is_nyugyoku()`を確認した。通常探索の欠測とは異なる。

[verification.json](verification.json)と[verify.py](verify.py)が読み取り専用reviewの証跡。
入力hash、cost/formal/reserveの分離、対象166開始履歴、1〜63の全ペアについて
合法手・終局・score・元履歴送信・毎局reset・clear・nodes・通信・library hashを再検査した。
64番目の旧監査失敗を再現し、合法入玉を独立に確認したが受理markerは作成していない。
初期設定handshakeはrunnerで検査・保存済み。全chunk最終監査は実行されていない。

32件の回帰検査（formal/cost/match_protocol/qualification）は通過。
旧検査に自由探索のzero-node入玉を含むauditor回帰がないことが露呈した。
今回は計算中の実装hashを保持し、修正と新manifest作成は次prepareへ渡す。

```bash
timeout 300 scripts/project_python.sh -c "import runpy; runpy.run_path('docs/research/results/decision-replication-formal-chunk001-incomplete-20260920/verify.py')"
timeout 300 scripts/project_python.sh -m unittest tests.test_decision_replication_formal tests.test_decision_replication_cost tests.test_match_protocol tests.test_match_protocol_qualification
```

[evidence.tar.gz](evidence.tar.gz)はjob/plan/handoff/execution/input-manifest/protocol/chunk/libraryと
全artifacts（全64ペアの記録・gzip通信、63受理marker）を保存。元attemptと絶対path対応を保持する。
再監査は元配置を読む。archive展開先へ移す場合は参照pathを明示的に対応させる必要がある。

## 再計画と判断の範囲

具体的な移行契約は[resolution.json](resolution.json)。新prepareで合法入玉だけにzero nodesを許容し、
欠測・負数・通常手zero・不正win・通信破損を拒否する回帰を追加する。
監査のみの版差と対局条件の同一性を記録して1〜63を明示移行・再監査する。
旧identityを上書きせず、新旧manifestの対応を残す。未受理64は両局を新processで再実行し、
65〜166へ固定順で進める。旧64の記録は隔離保存し、得点には加えない。
同じjobをreplanし、chunk002は提案しない。replan履歴は空、今回が上限2回中の1回目。

正式20,000局全完了後の一度だけの主判定を維持。途中勝率・Elo・区間は計算せず、
旧14,000局・cost・reserve・日次を合算しない。事前登録の標本数・閾値・対局条件を変更しない。
既知24,995旧局面の除外と棋譜単位splitを利用し、全pretraining祖先・全教師の非重複と
checkpoint再export一致は未確認。主張は固定release/seed42/履歴clearありに限定する。
全祖先分離や追加棋譜を再要求しない。human_decisionsは空、人間待ちなし。

新モデル・独立選抜結果はなく、champion `decision-aligned-weight005-epoch9`を維持。
新科学仮説はない。優先順位1〜6を再評価し維持する。
独立課題H-DAILY-ADJUDICATIONは合成fixtureで日次裁定を検査でき、次ジョブとして提案する。
H-MATCH-PROTOCOLの既存cost8計画も独立して継続できる。配備・routing・horizonは台帳の順を維持。
