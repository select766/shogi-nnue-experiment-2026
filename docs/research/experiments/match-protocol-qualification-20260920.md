# H-MATCH-PROTOCOL qualificationと感度試験の事前登録

2026-09-20。job: match-protocol-qualification / attempt-001。
前回の56局は手順pilotであり、棋力効果推定・標本数決定に成績を使用しない。
replan_history / human_decisionsはいずれも空。全祖先分離証明を要求する旧計画は適用しない。

## 今回の外部計算

既存candidate/control release各1個について、千日手・先手連続王手・後手連続王手の
3合成fixture × 履歴2 × clear2 × 打切り8/12手の48ケースを行う。
各着手を実機の`go nodes 128 searchmoves MOVE`で制限し、返ったbestmove、合成・clear情報、
送信positionと全着手を保存する。3回出現に当たる8手ではmax_moves、4回の12手では
それぞれ引分・先手負け・後手負けを要求する。毎ケースisready→usinewgame。
ResignValue=99999、Threads=1、Hash=16 MiB、MultiPV=1、book/ponderなし。
各modelの単一実プロセスを先後で共有する（手順fixture専用）。探索は全480回。

成功は48/48の期待終局・着手数・全探索の重み合成1回・要求clear証跡1回、実行時依存版/hashの
保存が成立すること。違法手、想定外bestmove、通信失敗、入力hash相違は停止し手順不合格。
時間切れ・途中終了は未確定であり棋力仮説の棄却ではない。
単一手制限による実USIとrunner裁定の統合試験であり、エンジンが自由探索で自発的に反復を
選ぶことや内部反復scoreを検証したとは主張しない。棋力試験は今回実行しない。

総上限1,800秒。回帰300秒、private binary build 300秒、fixtureプロセス900秒以内
（内部期限840秒）。CPUのみ、GPU不要、モデル学習/再exportなし。メモリ目安2 GiB。
初期NNUE、両release、checkpoint、lock、native ONNX library、固定入力のhashをprepare時に固定し、
run時に再照合する。実行時Python/distribution全版、git/submodule、binary、loaded shared library hashも保存。
checkpointと既存releaseの再export一致は未検証であり、今回の対象は固定releaseそのもの。

## 次ジョブの棋力感度試験（本runには含めない）

仮説は履歴または動的重み更新時clearが、固定candidate対controlの得点差を変えること。
モデルはdecision_aligned_weight005_from_m0 / control_from_m0 checkpoint 9に由来する既存releaseを
固定する。正式再現の主条件は履歴あり・clearありで、感度試験成績による条件選抜はしない。

標本はmatch-plan-v1/reserve.jsonlの全1,324棋譜、ファイル順固定、各1開始局面。
cost8の8棋譜と正式再現openings10000の10,000棋譜とは独立した用途として予約する。
reserveを正式再現の悪成績の置換に使わない。本体10,592局=1,324×履歴2×clear2×先後2。
開発・選抜・test・日次データを使わない。既知旧局面との照合範囲はカタログ準拠で、
全pretraining祖先の独立性を主張しない。棋譜単位で解析する。

今回の履歴介入は抽出SFEN以降の全対局着手を送るか、毎手現在SFENのみ送るか。
原棋譜history_usiは出典検証専用とし、両条件とも抽出以前の履歴はエンジン/裁定へ入れない。
この定義はpilotと一致するが、元棋譜全履歴の有無を調べる試験ではない。
毎局のreset、4回反復裁定は共通。100,000 nodes/着手、512着手で引分、1 thread、16 MiB hash、
ponder/bookなし。8セルの順はseed=20260920で棋譜ごとにshuffleし、結果を見て変更しない。
候補が先後を各1回担当し、先後のresultを候補視点へ変換する。

棋譜i・条件hcの先後2局平均得点をP_i,hc（勝1・分0.5・負0）とする。
事前の3対比は履歴 H_i=(P_i,11+P_i,10-P_i,01-P_i,00)/2、
clear C_i=(P_i,11+P_i,01-P_i,10-P_i,00)/2、
交互作用 I_i=(P_i,11-P_i,10-P_i,01+P_i,00)/2。
添字は履歴、clearの順。交互作用は差の差の半分であり、全対比の範囲は[-1,1]。
各平均に対し棋譜間標本SD / sqrt(1324)とt分布df=1323を用いた両側98.3333%区間を計算。
Bonferroniで3対比のfamilywise alpha=0.05。棋譜を独立単位とする大標本近似であり、
同じ棋譜内の8局を独立扱いしない。分散ゼロでは有意/同等性の断定をせず未確定とする。
各条件得点・Elo換算・終局理由・着手列一致率・実nodes・所要時間は副指標で選抜には使わない。

支持: 少なくとも1対比の補正区間が0を除外。実用差の目安は得点差1ポイント。
限定的棄却: 3区間すべてが[-0.01,+0.01]内なら、この固定条件で1ポイント以上の効果を棄却。
有意だが実用閾値内なら「小効果あり、実用差は棄却」と併記する。
その他は未確定。有意差なしを同等とはしない。

標本数は予備集合全件を費用と既存入力で固定し、pilotの7ペアの分散/成績では決めない。
両側alpha=0.05/3、80% powerの正規近似MDEは
(2.39398+0.84162)*SD/sqrt(1324)。SD=0.5なら4.45ポイント、最悪上界SD=1なら8.89ポイント。
1ポイントの微小効果には不足する（SD=0.5で同近似必要数約26,173棋譜）。
検出力は仮定依存であり保証しない。入力増量を人間へ要求せず、小差は未確定で終了する。

## 次ジョブの費用と停止

1. qualification合格後、別cost8ジョブで8×8=64局を全条件同一設定で測る。
   成績を標本数・条件選抜へ使わず、所要時間・メモリ・失敗だけで実行可否を決める。
2. pilot最遅31.822秒/局からの粗い外挿は10,592局で93.63 engine hours。
   syntheticからの推定なので、正式予算はcost8の最遅局×10,592×1.5と160 engine hoursを比較。
   超過なら感度試験を着手せずreviewへ再計画する。閾値や標本を成績に応じて緩めない。
3. 各chunkは直列で5時間以内を目標、管理上限6時間、全chunk合計160時間上限。
   chunkの棋譜数はcost8後にfloor(18000/(8*最遅局秒*1.5))を上限としてmanifestに固定する。
   1棋譜も入らなければ再計画。並列効率は未測定なので予算を架空の並列化で割らない。
4. 成績による早期終了・延長・逐次有意判定なし。全1,324棋譜の8セル完了後に一度だけ解析。
   障害・期限・違法手は停止し全部分結果を保存。未完局を引分へ置換しない。欠測があれば
   主解析を行わず未確定。完了部分の記述統計のみ探索的と明記する。
5. 再開は人手によるデータ供給を要さず、管理側の明示retryで既存入力を使用する。
   本試験driver/集約器、入力SFEN/棋譜ID重複検証、全chunk manifest、厳密な再開仕様の実装は
   別prepareで行い、本書の条件を維持する。今回の外部runはこれらの長時間計算を起動しない。

## 固定入力

match-plan manifest SHA256: `7e95ae6ba79537616c7554bef94dfeec1d51399cc453aa216f8f464dda92a7af`。
reserve SHA256: `fef2d6763ecc5d0c6de41700219995446d994bb4f8f1dde1561b1d9520357cc2`。
cost8 SHA256: `489fb26bf06718c05b0f447195a9c4a02dedecca2fc2c01c3b79bf7f256e2168`。
openings10000 SHA256: `ecf70e1cec54e5274f2581c21cf66657110a25e5d91902f4e1b2c21a5bed654d`。
モデル等の実hash一覧はattemptのinput-manifest.json。合成fixtureはqualification実装の
FIXTURES定数で完全指定し、学習・棋力評価標本へ混ぜない。
