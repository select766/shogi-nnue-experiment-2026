# 研究仮説レジストリ

更新日: 2026-08-25

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

## 優先順位付き未完了仮説

<!-- hypothesis id=H-SU-HORIZON status=unverified priority=1 -->
### 1. Search utility教師は探索量をまたいで安定する

- 状態: 未検証
- 仮説: 10,000 nodesで選ばれた固定blend候補の相対順位は、配備に近い100万nodesでも十分維持される。
- 根拠: pilotは候補生成10,000 nodes、独立utility 100,000 nodes、配備評価100万nodesであり、horizon差を分離していない。
- 次の実験: 100--300 rootで同じ候補群を10,000、100,000、1,000,000 nodesで評価し、候補順位相関、teacher変更率、最良候補一致を測る。
- 成功条件: 事前に固定した順位相関と最良候補一致の閾値を満たし、長い探索でも基準候補に対するutility gainが残る。

<!-- hypothesis id=H-EXPERT-DIVERSITY status=unverified priority=2 -->
### 2. Expertの高相関がrouting効果を制限している

- 状態: 未検証
- 仮説: 異なる探索局面群や目的でexpertを分化させれば、rootごとのblend変更が探索結果へ与える効果が大きくなる。
- 根拠: 現expertの同一局面出力相関は平均約0.966で、search utility pilotの教師gate変化も平均L1約0.05だった。
- 次の実験: expert別データ分割または多様性制約で小規模な分化expert群を作り、相関、oracle gain、候補bestmove多様性を現行expert群と比較する。
- 成功条件: validation lossを大きく悪化させずにexpert相関を下げ、未知rootでoracle gainと候補bestmove多様性を増やす。

<!-- hypothesis id=H-SU-GEOMETRY status=unverified priority=3 -->
### 3. Search utility候補の探索半径と形状が狭すぎる

- 状態: 未検証
- 仮説: expert logitへの単一`+1.0`摂動より、複数半径、expert対方向、疎な頂点を含む候補集合の方が、学習可能なutility差を作る。
- 根拠: pilotでは元gateと教師gateのtop-1一致が96.61%で、蒸留信号が弱かった。
- 次の実験: H-SU-HORIZONの結果に合う探索量を用い、候補数を制御したまま複数の候補幾何を比較する。
- 成功条件: 計算量当たりの独立utility gain、teacher変化量、held-out教師再現性が現行候補を上回る。

<!-- hypothesis id=H-TAIL-OBJECTIVE status=unverified priority=4 -->
### 4. 平均leaf lossが探索上重要な少数leafを希釈している

- 状態: 未検証
- 仮説: root内平均ではなく、上位分位、CVaR、最大loss、訪問回数重みなどのtail-sensitive目的が最善手・勝率へ転換しやすい。
- 根拠: root-grouped平均lossは大きく改善した一方、固定bestmoveでは改善せず、自己対局も不確定だった。
- 次の実験: 同じroot/leafと固定expertを使い、集約関数だけを変えた対応比較を行う。
- 成功条件: 未使用rootでtail指標と平均lossの両方を監視し、固定bestmoveを事前基準以上改善する候補を得る。

<!-- hypothesis id=H-ROOT-REPRESENTATION status=unverified priority=5 -->
### 5. Rootだけから探索後の影響を予測する表現力が不足している

- 状態: 未検証
- 仮説: DNNはrootで一度だけ実行する制約を保ったまま、root側の特徴またはadapter容量を増やせば、未知leaf集合に適した固定blendを予測できる。
- 根拠: 末端を見た共有teacherは転移するが、現adapterによるteacher追従やsearch utility蒸留の改善は小さい。
- 次の実験: 計算量上限を先に固定し、root-only特徴・adapter容量の小さなablationを行う。
- 成功条件: ONNX Runtime CPUで許容レイテンシ内に収まり、独立rootのgroup lossと探索指標をともに改善する。

<!-- hypothesis id=H-LEAF-LOSS-STRENGTH status=held priority=6 -->
### 6. 実探索leaf loss改善は小さな棋力向上を生んでいる

- 状態: 保留
- 仮説: 実探索leafモデルの自己対局`+13.0 Elo`は真の小効果であり、局数を増やせば0より上へ分離する。
- 根拠: 400局の95%区間は`[-20.4,+46.5]`で、正負の双方を含む。
- 次の実験: モデル変更ではなく測定精度の改善として、事前停止規則付きの追加自己対局を行う。
- 成功条件: 事前に定めた最小効果と信頼区間を満たす。満たさなければ、leaf lossを棋力の選抜指標にしない。

## 判定済み仮説

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

- 状態: 支持。250から100万nodesまで改善した。ただし棋力への転換はH-LEAF-LOSS-STRENGTHとして未確定である。

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
<!-- result-ledger:end -->
