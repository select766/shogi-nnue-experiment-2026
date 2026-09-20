# 対局手順qualification attempt-001 (2026-09-20)

対象: H-MATCH-PROTOCOL。job: `match-protocol-qualification`。
**手順試験は途中失敗。replanで同じジョブの次attemptへ渡す。棋力仮説は保留。**
外部計算は実行済みだが、48ケース・480探索の成功条件を満たさない。
新モデル・棋力対局・候補選抜はなく、championは維持する。

## 計画と完遂性

[事前登録](../../experiments/match-protocol-qualification-20260920.md)はcandidate/control各release、
千日手と先後連続王手の3fixture、履歴2×clear2×8/12手境界の48ケース。
実行は26.048秒、exit=1、timeoutではない。candidateの千日手8セルのみ完了（80探索）。
9件目の先手連続王手・履歴なし・clearなし・8手境界は1手だけ保存して失敗。
39件は未着手、controlは起動前。保存traceは81探索、実通信は失敗応答を含む82探索。
部分結果を48件合格へ拡張せず、失敗セルを引分にも置換しない。

9件目の2探索目は `go nodes 128 searchmoves 5c4c` に対して `bestmove 5c7c+`。
不一致を検出して停止した。失敗探索の応答はrecordsのsearchesには入らないが、
[実通信ログ](candidate-usi.txt)に重み合成・clear=0・bestmoveを含めて残る。

## 原因と再計画

実行時はGenerateAllLegalMovesを指定せず、advertised default=false。
`YaneuraOu/source/movegen.cpp`は成れる飛車の不成を通常生成から省き、
`source/thread.cpp`はsearchmovesと生成手の積集合が空なら全生成手へ戻る。
今回の `5c4c` は成れる飛車の不成であり、この経路と観測が整合する。
自由探索の反復裁定不良や棋力差を示す結果ではない。修正後の実機確認はまだ行っていない。

次prepareではfixture専用オプション `GenerateAllLegalMoves=true` を必須・記録対象にし、
不成の単一searchmoveがそのまま返る回帰検査を追加する。棋力試験の既定値は変更しない。
不一致探索もexpected/actual/position/responseを構造化して残すよう失敗記録を補強する。
実機は次の外部runで全48ケースを最初から検証し、今回の8件を合算しない。
成功条件・合成fixture・480探索・予算は維持。cost8と感度本体は合格後へ延期する。
履歴は空で残数2のため今回が最初のreplan。人間の入力やデータ追加は必要ない。

## 検証と分離

reviewではエンジンを起動せず、9個のfixtureとrecordsの一致、完了8セルの重複欠落なし、
8/12手の終局境界、81保存着手の合法再生・position・USI応答の一致、
各保存探索のblend=1とclear証跡を確認した。実通信82応答と失敗の差分も確認。
実行時manifestの20ファイル、prepare入力16ファイル、ロード済みlibrary9個のhashを再照合。
cshogi実行時distribution版は0.9.7。版記録の改善は確認できるが実機qualification全体は未完了。
reviewのエンジンなし33回帰検査が通過。台帳検査（34仮説・34結果文書）とgit diff --checkも通過した。

データは合成fixtureのみで、train/test/日次/正式対局へ混ぜない。
将来の感度試験は固定reserve全1,324棋譜×8セル=10,592局、正式10,000棋譜・cost8と用途分離。
prepareは棋譜ID非重複と履歴再生を確認済みで、reviewでは入力hashの不変を確認した。
全祖先分離・checkpointから既存releaseへの再export一致は主張しない。
感度試験の3対比・補正区間・検出力・160時間上限は事前登録のまま保持する。
今回は棋力影響・同等性のいずれも未測定。

## 判断と次課題

H-MATCH-PROTOCOLを保留へ戻す。失敗の説明は同仮説の手順修復に収まり、新しい棋力仮説は追加しない。
未完了優先順位は独立再現、手順感度、日次裁定、配備価値、root依存性、教師horizonの順を維持。
H-DECISION-REPLICATIONはqualification合格後にcost8へ進む。
独立したH-DAILY-ADJUDICATIONは合成fixtureで日次裁定と研究裁定を比較し、共通化と新series準備を
進められる。旧seriesの改変や日次値による選抜は行わない。
同仮説のnext_jobsは作らず、修正はresolution.jsonにより同ジョブへ引き継ぐ。

## 証跡

[execution](execution.json)、[summary](summary.json)、[verification](verification.json)、
[manifest](manifest.json)、[records](records.json)、[失敗fixture](fixture-009.json)、
[engine設定](engine-candidate.json)、[実ロードlibrary](libraries-candidate.json)、[USIログ](candidate-usi.txt)。
元artifactは `.research-loop/experiments/match-protocol-qualification/attempt-001/artifacts/execution-20260920T195120-177676/`。
途中実装・事前計画・回帰検査も今回コミットする。run/plan/handoffは管理側snapshotを保持する。
