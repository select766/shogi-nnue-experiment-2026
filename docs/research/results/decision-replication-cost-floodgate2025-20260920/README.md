# Floodgate2025 cost16: 費用・手順測定完了

2026-09-20、H-DECISION-REPLICATION。
ジョブ `decision-replication-cost-floodgate2025/attempt-001` は **complete**。
固定8棋譜×先後2の16局を完遂し、全1,273探索の監査を通過した。
棋力の独立再現は保留。正式20,000局は未実行で、費用標本の勝敗からモデル・条件を選ばない。

## 実行と監査

execution.jsonはreason=exited、exit_code=0、303.567秒。終了コードに加えて
[audit.py](audit.py)で全記録を検査した。[verification.json](verification.json)が結果。

- 入力31ファイルの固定hash、cost/formal/reserveの8/10,000/1,324件と棋譜・局面非重複を再確認。
- 全16局の元履歴を合法再生し、抽出SFENと一致。追加1,273着手もすべて合法。
- 全USI送信列を照合。元履歴＋追加履歴、go nodes 100000、各engineの毎局isready→usinewgameを確認。
- 1,273探索すべてにblendとclear=1が各1件、実nodesとbestmove応答が記録と一致。
- 実nodes合計118,633,938、1探索21〜100,951。nodes指定は停止上限であり、早い終局探索や超過を含む実測値を保存する。
- 終局は合法手なし14局、4回反復2局。各探索前の終局条件と最終盤面・scoreを再検査。
  このcost標本には連続王手・入玉・512手到達はなく、それらの自由探索での実現を保証しない。
  元履歴を跨ぐ反復・両側連続王手の裁定は合成回帰検査で補う。
- 設定、engine、モデル、実ロードlibraryのhash一致。qualification済みprivate binaryを再利用。

[manifest.json](manifest.json)、[protocol.json](protocol.json)、全16局・通信ログを含む
[evidence.tar.gz](evidence.tar.gz)を保存。元記録はattemptの
`artifacts/execution-20260920T201240-190171/`、生USIログは`/tmp/decision-cost-execution-20260920T201240-190171-{candidate,control}.log`。
archive内のtmp/は元/tmp/に対応する。監査は元配置を読み、engineを起動しない。

```bash
timeout 300 scripts/project_python.sh -c "import runpy; runpy.run_path('docs/research/results/decision-replication-cost-floodgate2025-20260920/audit.py')"
timeout 300 scripts/project_python.sh -m unittest tests.test_decision_replication_cost tests.test_match_protocol tests.test_match_protocol_qualification
```

review回帰28件通過（[tests.txt](tests.txt)）。prepareの41件通過とは実行対象が異なる。
監査初回はファイル直接起動でscripts importに失敗し、wrapperからrunpyを使う上記入口で解消した。
実験計算やモデルの再実行はしていない。

## 費用と次段階

[summary.json](summary.json)より平均36.918秒/ペア、最遅63.230秒/ペア。
平均から正式10,000ペアは逐次102.55時間。
事前規則の最遅×2＝126.4597秒と600秒の余裕から
`floor((21600-600)/126.4597)=166`ペア/chunk。
保守的な計画値の総対局時間は351.28時間（各chunkの起動・検査余裕は別）。
8ペアだけの外挿であり実時間の保証ではなく、並列速度向上は仮定しない。

[chunks.json](chunks.json)で固定ファイル順に61 chunk（166ペア×60＋40ペア）を確定。
次ジョブはchunk001、openings10000.jsonlの1〜166行の332局のみ。
元のモデルhash・engine・履歴・clear・nodesと
[事前登録](../../experiments/decision-replication-floodgate2025-20260920.md)の全設定を維持する。
正式runner、完全ペア再利用の監査と中断時保存をprepareで実装し、外部runで6時間以内に実行する。
各正常完了reviewは成績を選抜に使わず次の固定chunkを提案し、61 chunk完了後に一度だけ
ペア単位95%区間と主判定を計算する。timeout/失敗は未完了としてreplanし、自動再実行しない。
全20,000局に届かなければ主判定は未確定。予備局面との入替や成績による停止は禁止。

## 判断の範囲と研究台帳

旧manifest attemptは完全祖先分離を要求して未実行終了したが、現行自律方針に従い
整備済み最終版を利用した。今回jobのreplan_history/human_decisionsは空、通常不足・人間待ちはなし。
双方R3500以上、棋譜単位split、既知24,995旧局面の除外を利用する。
全pretraining祖先と全過去教師の分離、checkpointからのrelease再export一致は未確認。
主張は固定release・seed42・履歴/clearあり条件に限定し、完全独立とは呼ばない。
旧14,000局・cost・reserve・日次seriesを正式結果へ合算しない。

新モデルは作成せず、configs/research_champion.jsonの
`decision-aligned-weight005-epoch9`を維持。費用測定は候補選抜根拠にならない。
新たな科学的仮説は生じていない。優先順位1〜6は、棋力再現・手順影響・日次裁定・配備比較・
条件付きrouting・教師horizonの順を再評価して維持する。
H-MATCH-PROTOCOLの別cost8/感度試験はこの正式再現と独立で、既存の後続計画を重複登録しない。
