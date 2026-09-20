# 対局手順qualification attempt-002 (2026-09-20)

対象: H-MATCH-PROTOCOL、H-DECISION-REPLICATION。job: `match-protocol-qualification`。
**48件480探索を完遂し手順合格。棋力影響は未測定なのでH-MATCH-PROTOCOLは保留。**
外部runは95.174秒、exit=0、timeoutなし。旧attemptの8件は合算していない。
新モデル・選抜・棋力対局はなく、research_champion.jsonは維持する。

## 条件と実測

[事前登録](../../experiments/match-protocol-qualification-20260920.md)のcandidate/control各release、
千日手・先後連続王手の3fixture × 履歴2 × clear2 × 8/12手境界 = 48件。
GenerateAllLegalMoves=trueをfixture専用に必須化し、protocol_version=2と実設定を保存した。
前回不一致だった `go nodes 128 searchmoves 5c4c` は今回は `bestmove 5c4c`。
全480探索の指定手と応答が一致し、失敗探索・欠測・違法手は0。
棋力試験の設定や成功閾値は変更していない。

| 終局 | ケース数 | 各ケース着手数 |
|---|---:|---:|
| 3回出現時点のmax_moves | 24 | 8 |
| 4回出現の千日手引分 | 8 | 12 |
| 先手連続王手による先手負け | 8 | 12 |
| 後手連続王手による後手負け | 8 | 12 |

全探索で重み合成情報1件、指定どおりclear=0/1の証跡1件を確認。
実nodesは8〜133であり、指定128に常に一致するとは主張しない。
fixtureでは同じ実プロセスを両側で共有するため、24ケース/modelにつき
各側のisready→usinewgameが48回/model記録される。
単一searchmoveで制限した実USIとrunner裁定の統合試験であり、自由探索で自発的に
反復を選ぶこと、エンジン内部の反復score、棋力差・同等性は検証していない。

## 再検査と保存

reviewはエンジンを起動せず、`scripts/review_match_protocol_qualification.py`で
48個のfixtureとrecordsの一致、セルの重複欠落なし、全着手の合法再生、
送信position・go・実bestmove、終局理由・結果・境界、blend/clear記録を再照合した。
実行時manifestの20ファイル、prepareの16入力、各modelのロードlibrary9個のhashが一致。
cshogi実行時版0.9.7。エンジンなし36回帰検査は通過した。
入力hash・版・実通信を下記へ恒久保存し、仮説台帳検査とdiff検査を通して実装と共にコミットする。
失敗のexpected/actual/position/command/info/responses/errorの保存は故障注入回帰で検証した。

[execution](execution.json)、[summary](summary.json)、[verification](verification.json)、
[records](records.json)、[manifest](manifest.json)、[prepare入力hash](input-manifest.json)、
[候補設定](engine-candidate.json)、[対照設定](engine-control.json)、
[候補library](libraries-candidate.json)、[対照library](libraries-control.json)、
[候補USI](candidate-usi.txt)、[対照USI](control-usi.txt)、[回帰結果](regression.txt)。
元artifact: `.research-loop/experiments/match-protocol-qualification/attempt-002/artifacts/execution-20260920T195941-182783/`。
run/plan/handoffは管理側snapshotに保存。progress.jsonは最終ケース開始時のcomplete=falseを残すが、
最終fixture-048、records、summaryの全件完了を確認した。

## 分離と判断

実測入力は固定合成fixtureのみ。train/test/日次監視/正式棋力対局へ混ぜない。
将来のcost8、reserve全1,324、formal10,000は整備済みの別用途集合であり、今回その入力hashを
確認したが棋力測定には使用していない。既知の棋譜分離はカタログの範囲に限る。
全pretraining祖先の分離やcheckpointから既存releaseへの再export一致は未検証。
今回は固定release自体の手順検証で、完全分離証明を次の開始条件にしない。

replan履歴1件・残数1・human_decisions空を確認。既知障害は解消したため、追加replanではなく
completeとし、正常終了後の次段階を提案する。新たな科学的代替説明は生じておらず新仮説IDは追加しない。
未完了優先順位は独立再現、手順感度、日次裁定、配備価値、root依存性、教師horizonの順を維持する。

次の新規ジョブ`match-protocol-cost8`は固定cost8の8棋譜×履歴2×clear2×先後2=64局の費用測定。
事前登録の100,000 nodes/手・512手上限、毎局reset、4回反復裁定、既存releaseを固定する。
GenerateAllLegalMovesはfixture専用であり、棋力試験へ持ち込まず既定値を明示記録する。
成績で条件・標本を選ばず、最遅局秒×10,592×1.5と160時間上限を比較する。
費用内ならfloor(18000/(8×最遅局秒×1.5))棋譜/chunk以下、管理上限6時間として
感度本体を別ジョブへ渡す。超過・欠測は現行自律方針で再計画する。今回長時間計算は起動しない。

既に待機中の`decision-replication-cost-floodgate2025`は主条件の費用16局と正式20,000局の
事前登録を担当するので重複提案しない。独立の日次裁定・教師horizonも既存キューにある。
日次裁定の2ジョブは同じ目的が重複しているため、後続prepareは先行成果を確認して重複改修を避ける。
配備価値・root依存性は独立再現後の条件付き課題として維持する。
