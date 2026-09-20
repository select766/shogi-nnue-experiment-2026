# 対局と実験の並列化

結果を変えず独立実行できる作業は共通ツール・実験固有スクリプトとも並列化する。
prepareのplanに単位、worker数、CPU/メモリ上限、同等性検証を記録する。
逐次実行には依存関係・測定条件・資源制約など具体的な理由が必要。

## 固定nodes対局

`bash scripts/eval_match.sh OUTPUT [options] --workers 4`で先後ペアを並列に処理する。
固定nodesは既定4 workers、時間制限は既定1。時間制限でworkers>1は拒否する。
各workerは専用の2エンジン（root stats使用時は補助エンジンも）を持ち、ペア内は先手→後手。
Threads=1を維持し、集計は固定game ID順。全workerが成功した場合のみ集計結果を公開する。
汎用run_matchの途中ペア再利用は未対応。明示再開を必要とする正式追試にはformal runnerを使う。

formal runnerは`--workers 4`が既定。各workerのprogress、USIログ、エンジン設定、
ライブラリ証跡を分離し、監査済みの完全ペアだけをatomic markerで保存する。
再開時は全markerを再監査する。片側だけ完成したペアは両局とも再実行する。
固定IDからworkerへの配分をschedule.jsonへ保存し、集計はpair ID順。
停止・失敗は協調キャンセルで全workerのエンジンを終了する。
Threadsやnodes、モデル、開始局面、裁定・標本数・主判定は並列数で変更しない。

このホストは8物理コア、32GiBメモリ。標準4 workersは探索約4コア、エンジン8個で
実測約7GiBのRSSが目安。CPU/RAMを確認し、最大16の実装上限を自動推奨値として使わない。
乱数seed、入力順、集計順、共有cacheの有無を確認する。逐次対照との実測一致が必要。
時間制限対局、速度/費用測定、GPU競合は資源競合により条件が変わるため逐次等の理由を記す。
日次runnerは既にworkers対応。既存seriesの設定は本変更では変更しない。

## chunk001の移行と再開準備

旧attempt-001は63ペア受理後、64ペア目の合法入玉宣言nodes=0を旧監査が拒否した。
対局条件は維持し、監査を合法なbestmove winに限り修正した。通常手zero、負nodes、
欠測、通信不一致、不正winは拒否する。

2026-09-20のユーザー依頼により、旧resolutionの監査修正に加え、結果同等性を検証した
ペア並列化も次attemptへ適用する。旧resolutionの「監査のみの版差」はこの追加方針で拡張する。
次のprepareでは、以下の専用移行ツールを新しい保存先に実行する。計算は開始しない。

```bash
scripts/project_python.sh -m train_nnue.migrate_formal_pairs \
  --target "$RUN_DIR/continuation"
```

ツールは旧commit f3e7435の実装hash、固定入力/モデル/engine/library、固定chunkと対局条件、
旧identityを確認する。63ペアを旧・新両監査で再検査し、旧ソース・旧manifest・新旧identity・
変更ファイルhashを保存する。旧成果物は上書きしない。64ペア目にはmarkerを作らず、両局を
新processで再実行する。正式166ペアと全61 chunk・20,000局を維持する。
移行先が既に存在すれば拒否する。失敗した移行先は保存し、調査後に別名で明示再試行する。

prepareはplanに並列化の手順変更を追記し、独立ペアの実行順だけを変更したと明記する。
移行後のmanifestは変更しない。移行manifestは対局入力と実装を固定済みであり、
outer run.sh/planはloopのprepared.jsonで別途固定する。

外部run.shはcontinuationに対してcheck-only→execute→audit-onlyを実行し、
result-summary.mdをouter RUN_DIRへコピーする（失敗時にもコピー）。全体6時間以内、
stdoutは/tmpへ保存。`--workers 4 --budget-seconds 20400`を指定する。
prepare中は本体を起動しない。完了後reviewで正常完了ならchunk002へ進み、
途中成績による選抜・停止や旧64ペア目との二重計上をしない。

## 管理操作の復旧

保存済みreviewのキュー反映だけ失敗した場合、`apply-review`でセッション正常終了、
コミット・報告・台帳・clean worktreeを再検証して反映できる。Codexや計算は起動しない。
paused/inactive/blocked reviewが必要。元review.jsonは保持し、適用履歴をstateに記録する。

```bash
bash scripts/research_loop.sh apply-review JOB_ID --omit-existing-job EXISTING_ID
```

除外は明示指定のみ。既存ジョブがqueuedで仮説・依存が一致することを検査する。
文面変更の意味まで機械判定できないため、実行前にtask内容の同等性を確認する。
通常enqueueでは引き続き重複IDを拒否する。reviewには既存キューを提示し再提案を禁止する。

`run --max-jobs 1`はreplan後の計算にも進み得る。phase単位で止める場合:

```bash
bash scripts/research_loop.sh resume
bash scripts/research_loop.sh run --max-phases 1
```

これは現在の1 phaseだけで終了する。停止後のstateのpaused値とworker終了は別なので、
編集前にstatusのworker_present=false、active=nullを確認する。適用済みreviewの再適用は拒否する。

## 2026-09-20実装検証

`check_match_parallelism.py`でcost8先頭4ペア（各方式8局）を逐次/4並列で実測した。
全着手、最終SFEN、裁定、得点、探索局面、bestmove、実nodesは完全一致。
逐次130.617秒、4並列65.254秒（約2.00倍）。局の長さの偏りを含む小標本であり、
全正式対局の同等性や一定のspeedupを証明するものではない。
証跡は`tmp/match-parallel-qualification/comparison.json`とその下の全USI通信・棋譜。
検証用runは正式の受理markerへ合算しない。CPUのみ・sandbox外・timeout 300で実施。
移行検証では旧/新監査が63ペアすべて一致した。旧pair64は受理せず保存している。

回帰検査77件、仮説台帳検査（34仮説・37結果文書）を通過。正式対局は本整備では再実行していない。
