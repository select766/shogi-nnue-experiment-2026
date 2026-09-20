# 対局手順pilot (2026-09-20)

対象: H-MATCH-PROTOCOL。job: `match-protocol-pilot`、attempt-001。
**手順検証は合格、棋力差への影響は保留。** 予定7開始局面×履歴2×clear2×先後2の56局が
完了し、全6,034探索の履歴・合法性・重み合成・cache証跡をreviewで再検査した。
候補選抜や仮説の棋力面での支持・棄却・同等性判定はしない。新モデルは作っておらず、
`configs/research_champion.json`を維持する。

## 計画と完遂

[事前の段階計画](../../experiments/research-next-20260920.md)に従い、既存decision-aligned
weight005 checkpoint 9対task-only control checkpoint 9を固定。Threads=1、Hash=16 MiB、
MultiPV=1、book/ponderなし、100,000 nodes/手、最大512着手、CPU ONNXを用いた。
全条件で毎局isready→usinewgame、4回反復裁定を共通にした。
履歴なし・clearなしもこのresetと修正裁定を使うため、旧14,000局の完全再現ではない。

16個の24手synthetic合法開始局面をseed=20260920で生成。最初のhistory=true/clear=trueの
先後2局の時間だけから、安全係数1.5を使って7局面へ固定した。最遅31.822秒、割当時残り
2,917.741秒。時間測定2局も最初の8セルへ再利用し、成績による局数変更はない。
7局面の全8セルを完了、未完局・失敗・timeout・除外セルは0。
管理executionのexit=0だけでなく、56個のgame JSONとrecordsの一致、セルの重複欠落なし、
summary再集計一致を確認した。外部run全体1,355.373秒で1時間予算内。
progress.jsonは最後の局の開始時点のcompleted=55を残すが、game-056と最終recordsは完了している。

## 観測（棋力推定ではない）

各行7ペア・14局。秒数はresetを含む局時間合計、clearは更新callback内の実測合計。

| 履歴 | clear | 候補 勝/敗/分 | ペアscore平均 | 局時間 秒 | clear 秒 | 探索数 |
|---|---|---|---:|---:|---:|---:|
| なし | なし | 8/6/0 | 0.5714 | 315.459 | 0 | 1,493 |
| なし | あり | 9/5/0 | 0.6429 | 349.338 | 7.235 | 1,524 |
| あり | なし | 8/6/0 | 0.5714 | 316.861 | 0 | 1,493 |
| あり | あり | 9/5/0 | 0.6429 | 351.011 | 7.266 | 1,524 |

clearを変えると各履歴条件で14/14局の着手列が変わった。同じclearで履歴だけを変えた場合は
どちらも14/14一致。終局理由は全56局が合法手なし。実対局に千日手・連続王手の終局はなく、
その裁定の確認は実着手fixtureを使う回帰検査に限られる。履歴の影響なしとは結論しない。

全6,034探索で重み合成1件と指定どおりclear=0/1の記録1件を確認。両エンジンに各56回の
毎局reset、実送信positionと保存traceの一致、全goが100,000 nodes指定であることを確認した。
実nodesは21〜102,868。100,000未満の448探索は全てmate score付きであり、全探索が厳密に
同ノード数だったとは扱わない。違法手・protocol errorは0、全着手をcshogiで再生し終局を確認。
最大VmHWMは1エンジン921,436 KiB。clearありのcallback費用は局時間の約2.1%だが、
局時間差は着手列・探索数・固定実行順序にも依存し、約34秒の差をclear単独費用とはしない。

## 分離と限界

開始局面の生成履歴24手は出典であり、対局エンジンへ渡す履歴は開始局面以降だけ。
盤面・手番・持駒の重複は生成16局面内で除外。全trainとの非重複や棋譜単位の独立性、
人間棋譜の代表性は未証明。日次growth/固定accuracyを使わず、この標本を正式追試へ流用しない。
7ペアの得点差をElo改善・手順選抜に使わず、旧14,000局にも合算しない。

計画上の版記録には欠落がある。manifestのcshogi_versionは`unknown`。
review環境のdistribution版は0.9.7で同じ16開始局面を再生成できたが、これを実行時版の
完全な証明としない。次runはimportlib.metadataによる版と依存lock hashを記録する。
checkpointとreleaseは個別にhashしたが、この実験では再export一致を確認していない。
正式試験の入力固定時にこの対応も確認する。

## 実装と検査

run_matchに毎局reset、開始SFENと全着手履歴、合法手検査、4回反復・連続王手裁定、
探索traceを実装。RootStatisticsEngineも履歴を両engineへ送り現在局面を復元する。
YaneuraOuには任意のLogDynamicWeightCacheを追加し、clear後の時間を記録する。
ClearTTOnDynamicWeightsの恒久既定値は変更しない。pilot専用driverは応答deadline、
全条件完了局面だけの集計、失敗時部分記録を持つ。配備binaryは上書きしなかった。

外部runとreviewで28回帰検査が通過（match protocol、root statistics、merge、audit）。
reviewはエンジンを起動せず、`scripts/review_match_protocol_pilot.py`で保存データとUSIログを検査。
実行時manifestの18ファイルのSHA-256も一致した。仮説台帳検査も33仮説・31結果文書で通過した。
エンジン側の変更はサブモジュールcommit `524bb430`へ先に記録した。

## 判断と次課題

H-MATCH-PROTOCOLは棋力影響が未確定なので保留。手順健全性は次段階へ進む条件を満たす。
未完了の優先順位は独立再現、手順感度、日次裁定、配備価値、root依存性、教師horizonの順とする。

1. H-DECISION-REPLICATION: まず新規10,000棋譜・各1局面の出典ID、旧train/教師/選抜/対局/
   accuracyとの除外manifestを固定。IDや除外が証明できなければ保留し別出典の調達要件を記録。
   20,000局は旧候補/対照・履歴/clearありで一度だけ最終判定。2局からの費用目安は直列176.8時間、
   理想4並列44.2時間で、synthetic局面からの外挿にすぎない。実標本の費用確認後、6時間以内の
   chunkを別ジョブにする。manifest担当がその具体的chunkを提案する。
2. H-MATCH-PROTOCOL: 実機の反復/連続王手fixture、版記録、失敗時の記録を補強する。
   棋力影響の別感度試験は開始局面を単位とする履歴×clearの対応あり比較として、局数・区間・
   停止規則を事前登録。費用と検出力を評価し、56局の点推定に基づく標本増減はしない。
3. 新規H-DAILY-ADJUDICATION: daily_benchmark.gameは依然cshogi.is_draw()を直接終局判定に使う。
   2回出現で反復を返す挙動に対する裁定共通化・合成fixture検査を独立課題にする。
   日次監視値への実際の影響は未測定。変更時は新seriesを準備し旧seriesを保持、候補選抜には使わない。
4. H-DECISION-HORIZON: 次期計画の256 root教師診断を独立の費用上限付きジョブで行える。
   H-DEPLOYMENT-STRENGTHとH-CONDITIONAL-ROUTINGは独立再現後の条件付き課題として維持し、
   再現結果review時に具体的な対局/学習ジョブを作る。

## 保存証跡

恒久保存: [summary](summary.json)、[allocation](allocation.json)、[manifest](manifest.json)、
[openings](openings.json)、[review検証](verification.json)、
[候補USI options](engine0.json)、[対照USI options](engine1.json)。
全明細と実USI optionsは
`.research-loop/experiments/match-protocol-pilot/attempt-001/artifacts/execution-20260920T155154-72380/`。
`records.json`のhashをverificationへ保存した。実通信ログは同フォルダのengine0/1.json記載の/tmpパス。
plan/handoff/run.shはattemptに外側がsnapshot済み。元の実行成果物は改変していない。
