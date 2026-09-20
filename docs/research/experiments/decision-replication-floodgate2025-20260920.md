# H-DECISION-REPLICATION: Floodgate2025 cost16と正式再現の事前登録

2026-09-20 prepare。計算・学習・対局は未実行。対象仮説は検証中。
旧decision-replication-manifestは未実行終了であり、完全祖先分離の要求を撤回した現行方針を適用。
job.jsonのreplan_history/human_decisionsは空。qualification attempt-002の48件480探索合格を利用する。

## 仮説・対照と固定手順

seed42 decision-aligned weight005_from_m0 checkpoint9由来のcandidate releaseが、
同seed task-only control_from_m0 checkpoint9由来のcontrol releaseを上回るかを調べる。
モデル再選抜・再学習・再exportなし。両head.binは同一、backbone.onnxは異なる。
checkpoint/release再export一致は未確認。既存releaseに対する再現と限定する。
正確なモデル・engine・入力SHA256はattempt/input-manifest.json、設定はprotocol.jsonに固定。
qualification済みprivate binaryのhashは
5f7591ce03b55cf8016871f0ab7f11b4e2bd63be27b7d5b80c85a060ffeebc65。
新規buildせず同じbinaryを使う。ロードlibraryのhashも記録する。

全対局で100,000 nodes/着手、Threads=1、USI_Hash=16 MiB、MultiPV=1、
USI_Ponder=false、BookFile=no_book、ResignValue=99999、
ClearTTOnDynamicWeights=true、LogDynamicWeightCache=true。
GenerateAllLegalMoves=false（qualificationのtrueは単一searchmove fixture専用）。
それ以外のUSI defaultsは固定binaryのadvertised一覧に保存する。
乱数を使う局面抽出や探索optionを追加しない。対局順はファイル順、候補先手→候補後手。
同じ盤面・手番でプレイヤーだけ交換し、盤面を反転しない。

初期SFEN+history_usiを合法再生し抽出SFENと完全一致させる。各探索に元棋譜開始から
全履歴を送信し、裁定にも元履歴を含める。毎局両engineにisready→usinewgame、
各探索の動的重み更新時clearを要求。blend情報とclear=1を各探索1件確認する。
4回出現で千日手、反復区間中の連続王手側を負けとする。3回では裁定しない。
詰み/合法手なし、resign、合法な入玉宣言で決着。追加着手512手で引分
（元棋譜履歴は512に含めない）。反復・詰み判定は512手制限に優先する。
棋譜履歴付き新経路は合成回帰検査で検証し、cost16自由探索の通信をreviewで照合する。
qualificationは自発反復選択やengine内部scoreを保証しない。

## データ分離と使用hash

configs/research_data.jsonをdata-catalog.jsonにコピーする。
curated最終版manifest: 31a8d11f675285ecb8ff8ecbeacf6838f3a178412ba605836b98617d85d5f2f3。
match-plan manifest: 7e95ae6ba79537616c7554bef94dfeec1d51399cc453aa216f8f464dda92a7af。
cost8: 489fb26bf06718c05b0f447195a9c4a02dedecca2fc2c01c3b79bf7f256e2168。
formal openings10000: ecf70e1cec54e5274f2581c21cf66657110a25e5d91902f4e1b2c21a5bed654d。
reserve1324: fef2d6763ecc5d0c6de41700219995446d994bb4f8f1dde1561b1d9520357cc2。

双方原rating>=3500、1原棋譜1局面。development/selection/test/matchは棋譜単位分割。
固定JSONL間と既知24,995固有旧局面を除外済み。全pretraining祖先や全過去教師の
非重複は未確認であり、完全独立とは呼ばない。原棋譜実着手を理想教師と見なさない。
本prepareではhash、cost/formal/reserveの棋譜・局面非重複、cost全履歴再生を検査。
全データ再生済みの整備結果はdocs/research/results/floodgate2025-preparationを参照。
日次growth-v1や既存testは変更・利用せず、旧14,000局とは合算しない。

## 今回の測定・合否・費用

cost8の8開始局面×先後2=16局のみ。費用と手順の確認専用。
8棋譜の勝敗で候補・条件・局数・閾値を選ばず棋力の支持/棄却に使わない。
成功: 16局完遂、全着手合法、入力・設定一致、全探索nodes/clear/blendの証跡、
元履歴送信、毎局resetと終局裁定をreviewで再確認できること。
違法手、hash/履歴不一致、欠測、通信故障は手順失敗で即停止。仮説自体の棄却ではない。
期限・中断・一部未完は未確定。未完を敗北や引分と数えない。成績理由の早期終了なし。
CPUのみ、2 engineプロセス・1対局ずつ。GPU不要。上限21,600秒（6時間）。
内部実験期限20,400秒、外部timeout21,000秒、検査300秒+hash再照合60秒の余裕を取る。
全局の時間・探索数・実nodes・手数・終局・メモリを保存。
pairの平均時間×10,000で正式所要時間を概算し、並列speedupは仮定しない。
chunkの計画上限ペア数=floor((21,600-600)/(2×cost最遅pair秒))、最大10,000。
0なら費用再計画。実測8ペアでは所要上限は保証できず各chunkに6時間のhard limitも設ける。
全cost未完なら恣意的な完了部分だけで正式chunk数を確定しない。

## 正式20,000局の事前登録（本runでは絶対に起動しない）

openings10000の固定10,000棋譜×先後2局。cost/reserveは加えない。
同一モデルhash・engine hash・上記全設定と元棋譜履歴付き手順を維持する。
主指標: 各原棋譜iの候補score X_i=(先手score+後手score)/2、勝1/引分0.5/負0。
10,000ペアの平均pと差p-0.5。95%区間はペア標本SDを用いた
p ± 1.959963984540054 × sd(X)/sqrt(10000)（表示時のみ[0,1]へ切る）。
補助Elo=400 log10(p/(1-p))と同区間端の単調変換。境界0/1は±無限大と記載。
局を独立扱いしない。棋譜を独立単位とする近似推論で対局者など上位相関は未補正。
主判定は全20,000局完了・完全ペア・監査合格後の一度だけ、両側5%。
支持: 下端>0.5。棄却（当該固定条件の正改善のみ）: 上端<0.5。
区間が0.5を含む場合は未確定、有意差なしを同等性としない。
実用目安+5 Eloと区間の位置関係も別に記述し、主判定や標本数は変更しない。
seed43/44、標準HalfKP、同実時間優位への一般化はしない。
副指標W/D/L、先後・終局別は記述のみ。副解析で主判定を覆さない。

cost reviewが費用だけから固定順の連続ペアchunkを次ジョブとして登録する。
chunkは6時間以内。途中成績を用いた早期成功/無益中止や追加標本は行わない。
通常の中断は自動再開しない。管理側の明示retry前に成果物を確認する。
正式retryでは監査済み完全ペアのみ固定IDで再利用、途中ペアは2局とも新processで再実行し
旧部分を隔離保持。hash/設定/版不一致なら混合禁止しreviewで再計画。
入力・protocol異常は停止して修正範囲を監査し、影響全ペアを同じ固定入力で再実行する。
計算予算や運用上の打切りで20,000局に届かなければ主判定未確定、
部分結果は探索的な記述に限定。reserveによる悪い成績・長い局の差し替えは禁止。
新モデルへの変更、成功閾値の緩和、日次観測に基づく調整は別計画とする。
