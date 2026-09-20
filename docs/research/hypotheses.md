# 研究仮説レジストリ

更新日: 2026-09-20

このファイルを、未検証仮説の優先順位と過去の判定を管理する唯一の台帳とする。
実験計画や結果文書は詳しい条件と証跡を保持し、この台帳は「次に何を、なぜ検証するか」を保持する。

## 更新ルール

実験を開始するときは、対象仮説の状態を`検証中`にし、実験計画から仮説IDを参照する。
実験が終わった時点で、次を同じ変更に含める。

1. `docs/research/results/`に結果文書を追加する。
2. 対象仮説を`支持`、`棄却`、`飽和`、`保留`のいずれかに更新し、結果へのリンクを付ける。
3. 残る仮説を、情報価値、実験費用、依存関係、棋力への近さの順に再評価して、優先順位を1から振り直す。
4. 結果から生じた代替説明や制約を、新しい安定IDの仮説として追加する。既存IDの意味は変更しない。
5. 実験結果登録簿へ結果文書と対象仮説IDを追加し、
   `scripts/project_python.sh scripts/check_research_hypotheses.py`を通す。
6. 検査に通った実装、結果文書、レジストリをコミットする。コミットが作られるまでは実験完了としない。
   サブモジュールを変更した場合は、サブモジュールを先にコミットしてから親の参照更新をコミットする。

`未検証`、`検証中`、`保留`は未完了であり、優先順位を持つ。`支持`、`棄却`、`飽和`、`測定完了`は
完了した判定である。`支持`は一般的な真理ではなく、記載した条件と判定指標の範囲で証拠が得られたことを表す。
`棄却`も検証した介入と事前screening条件の範囲に限定する。不通過や有意差なしを、
広い機構仮説の否定・同等性の証明へ拡張しない。探索的結果と独立確証を区別する。

## 優先順位付き未完了仮説

以下は[2026-09-20レビュー](results/research-review-20260920/README.md)から追加した。
具体的条件と段階間の依存は[次期実験計画](experiments/research-next-20260920.md)に記す。

<!-- hypothesis id=H-DECISION-REPLICATION status=unverified priority=1 -->
### H-DECISION-REPLICATION: Decision-alignedの小効果が独立標本と明示した対局手順で再現する

- 状態: 未検証。旧decision-replication-manifest / attempt-001は未実行で手法を終了。
  [手順pilot](results/match-protocol-pilot-20260920/README.md)は56局を完遂し手順合格。
  [再開準備](results/research-loop-autonomy-20260920/README.md)で整備済みFloodgate 2025を登録し、
  全祖先の分離証明という実現困難な開始条件を撤回した。人間の追加棋譜提供は不要。
  棋力の独立再現は未着手。旧14,000局を合算しない。
- 次: 手順qualification完了後、固定cost8で履歴取扱い・reset・clear条件の費用を測定する。
  match-plan-v1の固定10,000棋譜による20,000局を6時間以内のchunkへ分割し、
  全完了後に一度だけ判定。既知の除外範囲と未確認祖先を明示し、完全独立とは呼ばない。
  その後seed 43,44への転移を調べる。費用不足は再設計し、データ増量を人間へ要求しない。

<!-- hypothesis id=H-MATCH-PROTOCOL status=held priority=2 -->
### H-MATCH-PROTOCOL: 履歴と動的重みのキャッシュ管理が棋力差の推定に影響する

- 状態: 保留。[qualification attempt-001](results/match-protocol-qualification-20260920/README.md)は
  48件中8件完了、連続王手fixtureの不成searchmove不一致で途中失敗。棋力は未測定。
  [感度試験計画](experiments/match-protocol-qualification-20260920.md)は事前登録済み。
  [pilot](results/match-protocol-pilot-20260920/README.md)の7開始局面×8セルは完遂。
  6,034探索の履歴・cache証跡を検証し、clear介入で14/14着手列が変化した。
  履歴単独の着手列は一致したが、棋力影響の方向・大きさ・同等性はいずれも未確定。
- 次: 同ジョブをreplanしfixture専用の全合法手生成・失敗記録を補強して48件を再検証。
  合格後にcost8で64局の費用を測り、reserve全1,324棋譜の対応あり感度試験をchunk化。
  正式再現の主条件は成績で選ばず履歴・clearありを維持。

<!-- hypothesis id=H-DAILY-ADJUDICATION status=unverified priority=3 -->
### H-DAILY-ADJUDICATION: 日次runnerの早期反復裁定が監視結果を変える

- 状態: 未検証。[pilot review](results/match-protocol-pilot-20260920/README.md)で研究runnerは
  4回出現へ修正した一方、daily_benchmark.gameはcshogi.is_draw()直結のままと確認。
  実際の日次勝率への影響件数・方向は未測定。
- 次: 合成fixtureで差を検査し裁定を共通化。既存日次データを候補選抜に使わず、
  手順変更は新seriesとして準備する。旧seriesを改変・遡及合算しない。

<!-- hypothesis id=H-DEPLOYMENT-STRENGTH status=unverified priority=4 -->
### H-DEPLOYMENT-STRENGTH: Expert Blendingが合成費用込みの同じ実時間で標準HalfKPを上回る

- 状態: 未検証。decision-alignedの対照はtask-only Expert Blendingであり、標準HalfKPではない。
  CPUで実行可能という速度測定だけでは、実時間での棋力優位を示さない。
- 次: 再現候補を固定し、前処理込みの同実時間を主条件、同nodesを副条件として標準NNUEと比較。

<!-- hypothesis id=H-CONDITIONAL-ROUTING status=unverified priority=5 -->
### H-CONDITIONAL-ROUTING: Decision-aligned改善にはrootごとに変わるgateが寄与している

- 状態: 未検証。現在の対照では、root依存性の改善と全体のexpert利用率の変化を分離できない。
- 次: 同expertのtrain平均gate・学習した定数logitsと比較し、root入力自体の寄与を測る。

<!-- hypothesis id=H-DECISION-HORIZON status=unverified priority=6 -->
### H-DECISION-HORIZON: 配備探索量に合わせたhard方向教師がdecision-aligned学習を改善する

- 状態: 未検証。現教師は候補10k/reference100k nodes。候補順位のhorizon転移が弱い既存結果がある。
- 次: 新規同一rootで候補100k/reference100kと候補100k/reference1mを対照化する。
  まず教師診断と費用を測り、モデル構造や候補方向は同時に変えない。

## 判定済み仮説

<!-- hypothesis id=H-FLOODGATE-2025 status=measured -->
### H-FLOODGATE-2025: 原棋譜から双方R3500以上の実手数・履歴付き評価標本を構成できる

- 状態: 測定完了。[2025年データ整備](results/floodgate2025-preparation/README.md)で
  128,843棋譜を走査し、38,351棋譜・5,710,803局面を採用、全件再生検証を通過した。
  開発・選抜・test・対局を棋譜単位で分け、費用測定8棋譜と本番10,000棋譜も固定。
- H-DECISION-REPLICATIONの次の準備は本データを使用する。上記旧計画の完全な祖先分離証明は
  開始条件から外し、既知の重複除外範囲と未確認範囲を明示する。優先順位は維持する。
  元の保留attemptは履歴として残す。新しい棋力の判定・対局はまだ行っていない。

<!-- hypothesis id=H-RESEARCH-AUDIT status=measured -->
### H-RESEARCH-AUDIT: 保存された対局明細から小効果の統計とデータ分離を再点検できる

- 状態: 測定完了。[再解析](results/research-review-20260920/README.md)で14,000局の
  ペア集計・包含関係を確認。ペア区間`[+1.36,+12.00]`、教師と重なる1開始局面を除いても同程度。
  教師train/val重複1局面、履歴欠落、キャッシュ条件の不統一を確認し、影響を上記仮説へ分離した。

<!-- hypothesis id=H-DECISION-ALIGNED-LOSS status=supported -->
### H-DECISION-ALIGNED-LOSS: Rootの意思決定に整合した目的なら棋力へ転換できる

- 状態: 支持。事前登録12,000局は5,917勝5,679敗404分、`+6.89 Elo`、95%区間
  `[+0.78,+13.00]`だった。未使用局面2,000局の感度追試後も合計14,000局で`+6.68 Elo`、
  区間`[+1.02,+12.34]`を維持した。固定条件での小効果として支持する。
- 2026-09-20限定追記: [ペア再解析](results/research-review-20260920/README.md)でも
  14,000局`+6.68 Elo [ +1.36,+12.00 ]`。旧task-only対照、seed42、10万nodes、
  履歴なし・動的重み更新時clear指定なしに対する名目区間である。追加計画は当初4,000局を見た後に
  登録され、選択・逐次判断全体を補正した確証ではない。標準HalfKPや同実時間への優位は未検証。

<!-- hypothesis id=H-ROOT-SEARCH-STATS status=rejected -->
### H-ROOT-SEARCH-STATS: 浅いroot探索統計が固定blendの予測に必要である

- 状態: 棄却。valB/valA lossと固定validation bestmoveは改善したが、独立10,000 testは
  `-0.01`ポイント、重複なし1,000局面追加splitは`-2.70`ポイント、McNemar `p=0.038878`
  だった。速度は本探索比約0.13%だが、固定bestmove改善が独立splitへ転移しなかった。
- 棄却範囲は固定1024-node MultiPVの6特徴追加。浅い探索情報一般の不必要性までは示さない。

<!-- hypothesis id=H-ROOT-SEARCH-STATS-STRENGTH status=measured -->
### H-ROOT-SEARCH-STATS-STRENGTH: 浅いroot探索統計の小さなbestmove改善は棋力へ転換する

- 状態: 測定完了。固定10万nodesの4,000局は1,921勝1,942敗137分、`-1.82 Elo`、95%区間
  `[-12.41,+8.76]`だった。追加固定bestmove splitも`-2.70`ポイントだったため、
  小さなvalidation改善の棋力転換は確認できなかった。

<!-- hypothesis id=H-SU-DIVERSITY-OBJECTIVE status=rejected -->
### H-SU-DIVERSITY-OBJECTIVE: 候補手のdisagreementを直接目的にすればutilityと多様度を両立できる

- 状態: 棄却。selectionで多様度を直接最大化したexpert別半径は、held-outで有効教師率`-2.08`pt、
  gain`-3.66cp`、候補手多様度`-0.198`となり、固定半径4より3指標すべて悪化した。

<!-- hypothesis id=H-SU-JOINT-DIRECTIONS status=rejected -->
### H-SU-JOINT-DIRECTIONS: 複数expertを同時に動かす候補方向ならutilityと多様度を両立できる

- 状態: 棄却。joint方向4本を含む9候補はselectionのutilityと多様度を改善したが、held-outでは
  有効教師率差`0`、gain差`-37.625cp`、候補手多様度差`-0.0052`となり、同時転移しなかった。

<!-- hypothesis id=H-TASK-ALIGNED-EXPERTS status=rejected -->
### H-TASK-ALIGNED-EXPERTS: Rootから予測可能な役割でexpertを専門化すればrouting価値が増える

- 状態: 棄却。32手幅phase専門化は担当expert lossとoracle gainを改善したが、独立64 rootの
  候補手多様度が`1.828`から`1.766`へ低下し、事前の同時改善条件を満たさなかった。
- 32手幅phase役割とpositive-axis候補のscreening不通過に限定する。専門化一般や棋力への寄与は否定しない。

<!-- hypothesis id=H-ROOT-FEATURES status=rejected -->
### H-ROOT-FEATURES: 現DNN出力にないroot情報が固定blend予測に必要である

- 状態: 棄却。静的7特徴は独立root lossを改善し、CPU追加時間も増やさなかったが、
  固定bestmoveは62.00%から61.95%へ低下した。浅い探索統計は`H-ROOT-SEARCH-STATS`へ分離した。
- 7特徴・1 epochの候補を採用しない判断であり、微小なbestmove点推定の負方向は一般仮説の反証ではない。

<!-- hypothesis id=H-SU-ADAPTIVE-RADIUS status=rejected -->
### H-SU-ADAPTIVE-RADIUS: Root別に候補半径を変えるとsearch utility信号を効率よく増やせる

- 状態: 棄却。適応半径はheld-outの有効教師率を`+5.21`pt改善したが、候補手多様度を
  `-0.3125`、95%区間`[-0.5000,-0.1354]`悪化させた。

<!-- hypothesis id=H-LEAF-LOSS-STRENGTH status=measured -->
### H-LEAF-LOSS-STRENGTH: 実探索leaf loss改善は小さな棋力向上を生んでいる

- 状態: 測定完了。固定10万nodesの4,000局は1,950勝1,919敗131分、`+2.69 Elo`、95%区間
  `[-7.90,+13.28]`で0を跨いだ。局数を10倍にしても正方向へ分離せず、leaf loss単独では選抜しない。
- この区間は有用な正効果も含み、効果なし・棋力同等とは判定していない。

<!-- hypothesis id=H-ROOT-REPRESENTATION status=rejected -->
### H-ROOT-REPRESENTATION: Rootだけから探索後の影響を予測する表現力が不足している

- 状態: 棄却。hidden 256はvalBで悪化し、512も未使用valAで平均group lossを改善しなかった。
  配備制約を保った特徴追加は`H-ROOT-FEATURES`へ分離した。
- 棄却範囲は入力固定・1 epochの幅256/512。表現力不足一般を否定したものではない。

<!-- hypothesis id=H-TAIL-OBJECTIVE status=supported -->
### H-TAIL-OBJECTIVE: 平均leaf lossが探索上重要な少数leafを希釈している

- 状態: 支持。上位2/8 leafのCVaR学習は独立rootでtail lossと平均lossを改善し、固定10,000局面の
  bestmove一致率をmean対照比`+0.21`ポイントとした。ただし`p=0.6325`で効果量は未確定である。
- 支持はtail/mean loss改善と当時のscreening条件に限定。loss上位leafが意思決定上重要という
  機構や棋力向上までは実証していない（[レビュー](results/research-review-20260920/README.md)）。

<!-- hypothesis id=H-SU-HORIZON status=rejected -->
### H-SU-HORIZON: Search utility教師は探索量をまたいで安定する

- 状態: 棄却。10k対1mの候補utility順位相関は平均0.087、95%区間`[-0.004,0.182]`で、
  事前閾値0.30を上端でも下回った。1mの有効教師率21.1%は残るため、候補選抜horizonを揃えて次へ進む。

<!-- hypothesis id=H-EXPERT-DIVERSITY status=rejected -->
### H-EXPERT-DIVERSITY: Expertの高相関がrouting効果を制限している

- 状態: 棄却。偏差拡大で相関を0.892へ下げても、100万nodesの有効教師率、oracle gain、候補手多様度が
  すべて悪化した。task-alignedな専門化は別仮説へ分離した。
- 棄却した介入は既存expert偏差の放射状2倍拡大であり、有用な多様性一般の否定ではない。

<!-- hypothesis id=H-SU-GEOMETRY status=rejected -->
### H-SU-GEOMETRY: Search utility候補の探索半径と形状が狭すぎる

- 状態: 棄却。同数9候補のaxis-2とcyclic contrastは、100万nodes screeningの事前3条件を
  同時に満たさなかった。root別の適応半径は別仮説へ分離した。

<!-- hypothesis id=H-RUNTIME-CPU status=measured -->
### H-RUNTIME-CPU: CPU ONNX Runtimeで実運用可能な固定blendを構成できる

- 状態: 測定完了。Python/GPU推論を除去し、やねうら王内CPU ONNX RuntimeとC++ blendを実装・計測した。

<!-- hypothesis id=H-INPUT-GAP-DIAG status=measured -->
### H-INPUT-GAP-DIAG: DNN rootとNNUE leafの手数差を層別診断できる

- 状態: 測定完了。DNNには探索root、NNUEには数十手先の末端を与える配備前提を保った診断を整備した。

<!-- hypothesis id=H-EVAL-BASELINE status=measured -->
### H-EVAL-BASELINE: 固定10,000局面でモデル差を再現可能に評価できる

- 状態: 測定完了。対応ありMcNemar検定を含む基盤を整備し、当時の候補に有意差がないことを確認した。

<!-- hypothesis id=H-GATE-SPARSITY status=rejected -->
### H-GATE-SPARSITY: Gateの疎化そのものが最善手一致を改善する

- 状態: 棄却。entmaxでlossを維持して疎化できたが、固定bestmoveは改善しなかった。

<!-- hypothesis id=H-PAIR-ROUTER status=rejected -->
### H-PAIR-ROUTER: 末端ごとのoracle蒸留が実運用routerを改善する

- 状態: 棄却。探索中に固定するroot gateと教師の粒度が一致せず、task-onlyを上回らなかった。

<!-- hypothesis id=H-ROOT-FIXED-TEACHER status=supported -->
### H-ROOT-FIXED-TEACHER: 複数末端へ共有できるroot固定blendが存在する

- 状態: 支持。末端集合Aで最適化したgateが独立な末端集合Bへ転移し、root-group lossを改善した。

<!-- hypothesis id=H-ROOT-DATA-SCALE status=saturated -->
### H-ROOT-DATA-SCALE: 棋譜proxyのroot-groupedデータ増量だけで改善が続く

- 状態: 飽和。500万root付近が最良で、700万rootまでの増量で改善しなかった。

<!-- hypothesis id=H-PROXY-GAP status=supported -->
### H-PROXY-GAP: 棋譜先局面と実探索leafの分布差がrouterを制限する

- 状態: 支持。教師生成法を揃えた対照より実探索leaf学習が実探索validationで改善した。

<!-- hypothesis id=H-BUDGET-TRANSFER status=supported -->
### H-BUDGET-TRANSFER: 低nodesで学習したrouterのleaf loss改善は探索量をまたいで残る

- 状態: 支持。250から100万nodesまで改善した。ただし棋力への転換は`H-LEAF-LOSS-STRENGTH`の4,000局でも
  0から分離せず、decision-aligned目的を別仮説とした。

<!-- hypothesis id=H-DEEP-LEAF status=rejected -->
### H-DEEP-LEAF: より深いleaf score教師がbestmoveを改善する

- 状態: 棄却。deep lossは改善したが、固定10,000局面のbestmoveを有意に悪化させた。

<!-- hypothesis id=H-ONPOLICY-QSEARCH status=saturated -->
### H-ONPOLICY-QSEARCH: On-policy qsearch leafへの再収集だけでbestmoveが改善する

- 状態: 飽和。50万rootまで増量してloss差は極小となり、bestmoveは実質同等だった。

<!-- hypothesis id=H-SU-LOCAL status=rejected -->
### H-SU-LOCAL: 局所`+1.0`候補による小規模search utility蒸留がbestmoveを改善する

- 状態: 棄却。1,600 root pilotで教師CEの改善は僅かで、固定bestmoveは改善しなかった。

## 実験結果登録簿

この表の各行は、結果文書と、それが更新した仮説IDを対応付ける。新しい結果文書がこの表にない場合、
検査スクリプトは失敗する。

<!-- result-ledger:begin -->
| 結果 | 仮説ID | 要約 |
|---|---|---|
| [Expert Blending実行速度](results/expert-blending-speed.md) | H-RUNTIME-CPU | CPU ONNX RuntimeとC++ blendの実装・速度測定 |
| [旧paired手数別loss](results/check-loss-per-gameply/README.md) | H-INPUT-GAP-DIAG | 初期の手数差診断 |
| [NNUE gate手数別loss](results/check-loss-per-gameply-nnue-backbone/README.md) | H-INPUT-GAP-DIAG | NNUE backbone条件の診断 |
| [DNN gate手数別loss](results/check-loss-per-gameply-dnn-backbone-v4/README.md) | H-INPUT-GAP-DIAG | DNN root条件の診断 |
| [固定10,000局面評価](results/accuracy-eval-10k-current/README.md) | H-EVAL-BASELINE | 評価基盤と初回比較 |
| [Gate診断](results/gate-diagnostics-lambda05-180/README.md) | H-GATE-SPARSITY, H-EXPERT-DIVERSITY | 疎性、利用均衡、oracle、expert相関を診断 |
| [Gateモデル変更](results/gate-model-change-20260823/README.md) | H-GATE-SPARSITY | entropy、balance、entmaxを比較 |
| [Router蒸留](results/router-distillation-20260823/README.md) | H-PAIR-ROUTER, H-ROOT-FIXED-TEACHER | per-pair教師を棄却しroot共有教師へ移行 |
| [Root-grouped router](results/root-grouped-router-20260823/README.md) | H-ROOT-FIXED-TEACHER | 未知末端へ転移する固定blendを確認 |
| [Root-grouped大規模追試](results/root-grouped-router-scale-20260824/README.md) | H-ROOT-DATA-SCALE | 500万から700万rootで飽和を確認 |
| [Proxy gap](results/proxy-gap-20260824/README.md) | H-PROXY-GAP | 実探索leaf分布の追加効果を確認 |
| [探索量転移と棋力](results/search-budget-and-strength-20260824/README.md) | H-BUDGET-TRANSFER, H-LEAF-LOSS-STRENGTH | loss転移を支持、棋力は不確定 |
| [Search-aware / on-policy](results/objective-onpolicy-20260825/README.md) | H-DEEP-LEAF, H-ONPOLICY-QSEARCH | deep教師を棄却、on-policy qsearchを飽和判定 |
| [Search utility pilot](results/search-utility-distillation-20260825/README.md) | H-SU-LOCAL, H-SU-HORIZON, H-SU-GEOMETRY | 局所候補条件を棄却し残る制約を分離 |
| [Search utility探索量安定性](results/search-utility-horizon-20260825/README.md) | H-SU-HORIZON, H-SU-GEOMETRY | 10kから1mへの候補順位転移を棄却し配備horizon整合へ更新 |
| [Expert多様性ボトルネック](results/expert-diversity-20260825/README.md) | H-EXPERT-DIVERSITY, H-TASK-ALIGNED-EXPERTS | 放射状多様化を棄却しtask-aligned専門化を分離 |
| [Search utility候補幾何](results/search-utility-geometry-20260826/README.md) | H-SU-GEOMETRY, H-SU-ADAPTIVE-RADIUS | 一律の広い候補を棄却しroot別半径を分離 |
| [Tail-sensitive group objective](results/tail-objective-20260826/README.md) | H-TAIL-OBJECTIVE | CVaRでtail lossを改善し固定bestmoveも小幅な正方向 |
| [Root-only adapter容量](results/root-representation-capacity-20260826/README.md) | H-ROOT-REPRESENTATION, H-ROOT-FEATURES | 単純な幅拡大を棄却し入力特徴仮説を分離 |
| [実探索leaf lossの棋力追試](results/leaf-loss-strength-20260826/README.md) | H-LEAF-LOSS-STRENGTH, H-DECISION-ALIGNED-LOSS | 4,000局で小効果を精密化しdecision-aligned目的を分離 |
| [Root decision-aligned目的](results/decision-aligned-loss-20260826/README.md) | H-DECISION-ALIGNED-LOSS | 固定bestmoveとEloは正方向だが4,000局区間が0を跨いだ |
| [Root別search utility候補半径](results/adaptive-radius-20260826/README.md) | H-SU-ADAPTIVE-RADIUS, H-SU-DIVERSITY-OBJECTIVE | 有効教師率は増えたが候補手多様度が悪化 |
| [Root予測可能なexpert専門化](results/task-aligned-experts-20260826/README.md) | H-TASK-ALIGNED-EXPERTS, H-SU-DIVERSITY-OBJECTIVE | phase専門化は成立したがsearch候補手多様度は悪化 |
| [明示root特徴ablation](results/root-features-20260826/README.md) | H-ROOT-FEATURES, H-ROOT-SEARCH-STATS | 静的7特徴はlossのみ改善し浅い探索統計を分離 |
| [候補手disagreement直接目的](results/disagreement-objective-20260827/README.md) | H-SU-DIVERSITY-OBJECTIVE, H-SU-JOINT-DIRECTIONS | expert別半径の直接多様度選定はheld-outへ転移せずjoint方向を分離 |
| [浅いroot探索統計](results/root-search-stats-20260827/README.md) | H-ROOT-SEARCH-STATS, H-ROOT-SEARCH-STATS-STRENGTH | loss・固定bestmove・追加時間の3条件を満たし棋力追試を分離 |
| [Joint signed候補方向](results/joint-candidate-directions-20260828/README.md) | H-SU-JOINT-DIRECTIONS | selection改善はheld-outへ転移せずutilityと多様度の同時改善を棄却 |
| [浅いroot探索統計の棋力追試](results/root-search-stats-strength-20260828/README.md) | H-ROOT-SEARCH-STATS-STRENGTH, H-DECISION-ALIGNED-LOSS | 4,000局で棋力転換を確認できず過去の正方向仮説を深掘り |
| [24時間 深掘り検証](results/deep-validation-24h-20260828/README.md) | H-DECISION-ALIGNED-LOSS, H-ROOT-SEARCH-STATS | decision-alignedを12,000局で支持しroot統計の独立bestmove転移を棄却 |
| [2026-09-20研究レビュー](results/research-review-20260920/README.md) | H-RESEARCH-AUDIT, H-DECISION-ALIGNED-LOSS, H-TAIL-OBJECTIVE, H-ROOT-FEATURES, H-ROOT-REPRESENTATION, H-ROOT-SEARCH-STATS, H-TASK-ALIGNED-EXPERTS, H-EXPERT-DIVERSITY, H-LEAF-LOSS-STRENGTH, H-MATCH-PROTOCOL, H-DECISION-REPLICATION, H-DEPLOYMENT-STRENGTH, H-CONDITIONAL-ROUTING, H-DECISION-HORIZON | ペア再解析で旧条件の小効果を維持し、判定範囲を限定。履歴・TT・独立性と配備価値を次期仮説へ分離 |
| [対局手順pilot](results/match-protocol-pilot-20260920/README.md) | H-MATCH-PROTOCOL, H-DECISION-REPLICATION, H-DAILY-ADJUDICATION | 7開始局面56局・6,034探索の手順合格。棋力影響は保留、独立追試manifestと日次裁定を課題化 |
| [Floodgate 2025データ整備](results/floodgate2025-preparation/README.md) | H-FLOODGATE-2025, H-DECISION-REPLICATION | 双方R3500以上38,351棋譜・実手数付き571万局面、棋譜分離と固定対局用1万局面を整備 |
| [自律研究ループの再開準備](results/research-loop-autonomy-20260920/README.md) | H-DECISION-REPLICATION, H-FLOODGATE-2025 | 整備済み入力を登録し、完全分離証明の要求を撤回。通常不足は有界再計画または手法終了へ |
| [対局手順qualification attempt-001](results/match-protocol-qualification-20260920/README.md) | H-MATCH-PROTOCOL | 8/48件で不成searchmove不一致、棋力未測定、同ジョブをreplan |
<!-- result-ledger:end -->
