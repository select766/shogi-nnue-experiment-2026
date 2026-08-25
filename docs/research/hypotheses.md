# 研究仮説レジストリ

更新日: 2026-08-26

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

<!-- hypothesis id=H-LEAF-LOSS-STRENGTH status=held priority=1 -->
### 1. 実探索leaf loss改善は小さな棋力向上を生んでいる

- 状態: 保留
- 仮説: 実探索leafモデルの自己対局`+13.0 Elo`は真の小効果であり、局数を増やせば0より上へ分離する。
- 根拠: 400局の95%区間は`[-20.4,+46.5]`で、正負の双方を含む。
- 次の実験: モデル変更ではなく測定精度の改善として、事前停止規則付きの追加自己対局を行う。
- 成功条件: 事前に定めた最小効果と信頼区間を満たす。満たさなければ、leaf lossを棋力の選抜指標にしない。

<!-- hypothesis id=H-TASK-ALIGNED-EXPERTS status=unverified priority=2 -->
### 2. Rootから予測可能な役割でexpertを専門化すればrouting価値が増える

- 状態: 未検証
- 仮説: 現expert差の単純拡大ではなく、探索phaseやroot特徴に対応する役割を教師ありで割り当てれば、
  各expertの品質を保ちながら候補utilityを増やせる。
- 根拠: 放射状偏差2倍は相関を0.892まで下げたが、有効教師率と候補手多様度をともに悪化させた。
- 次の実験: rootだけから再現できるclusterを事前定義し、各expertをcluster別目的へ学習する。
- 成功条件: 単体expert lossを維持し、独立rootの100万nodes utilityと候補手多様度をM0より改善する。

<!-- hypothesis id=H-ROOT-FEATURES status=unverified priority=3 -->
### 3. 現DNN出力にないroot情報が固定blend予測に必要である

- 状態: 未検証
- 仮説: adapter幅ではなく、rootの探索統計や手数・phaseなど、現backbone出力に明示されない低コスト特徴が
  未知leaf集合に適した固定blendの予測を改善する。
- 根拠: hidden幅256/512への拡大は独立rootのgroup lossを改善せず、単純な容量不足では説明できなかった。
- 次の実験: rootで一度だけ計算できる少数特徴を事前選定し、同程度のadapter容量でablationする。
- 成功条件: 独立rootのgroup lossを改善し、CPU追加時間10%以内で固定bestmoveも正方向にする。

<!-- hypothesis id=H-SU-ADAPTIVE-RADIUS status=unverified priority=4 -->
### 4. Root別に候補半径を変えるとsearch utility信号を効率よく増やせる

- 状態: 未検証
- 仮説: 全root一律の広い候補ではなく、root特徴から必要なlogit半径を予測すれば、候補数を増やさず
  informative rootとutility gainを増やせる。
- 根拠: axis-2はgain p90を35.5cpから48.4cpへ増やしたが、有効教師率と候補手多様度の増加は小さかった。
- 次の実験: 小標本で複数半径をoracle診断し、最良半径とroot特徴の予測可能性を測る。
- 成功条件: held-out rootで固定半径より有効教師率と候補手多様度を改善する。

## 判定済み仮説

<!-- hypothesis id=H-ROOT-REPRESENTATION status=rejected -->
### H-ROOT-REPRESENTATION: Rootだけから探索後の影響を予測する表現力が不足している

- 状態: 棄却。hidden 256はvalBで悪化し、512も未使用valAで平均group lossを改善しなかった。
  配備制約を保った特徴追加は`H-ROOT-FEATURES`へ分離した。

<!-- hypothesis id=H-TAIL-OBJECTIVE status=supported -->
### H-TAIL-OBJECTIVE: 平均leaf lossが探索上重要な少数leafを希釈している

- 状態: 支持。上位2/8 leafのCVaR学習は独立rootでtail lossと平均lossを改善し、固定10,000局面の
  bestmove一致率をmean対照比`+0.21`ポイントとした。ただし`p=0.6325`で効果量は未確定である。

<!-- hypothesis id=H-SU-HORIZON status=rejected -->
### H-SU-HORIZON: Search utility教師は探索量をまたいで安定する

- 状態: 棄却。10k対1mの候補utility順位相関は平均0.087、95%区間`[-0.004,0.182]`で、
  事前閾値0.30を上端でも下回った。1mの有効教師率21.1%は残るため、候補選抜horizonを揃えて次へ進む。

<!-- hypothesis id=H-EXPERT-DIVERSITY status=rejected -->
### H-EXPERT-DIVERSITY: Expertの高相関がrouting効果を制限している

- 状態: 棄却。偏差拡大で相関を0.892へ下げても、100万nodesの有効教師率、oracle gain、候補手多様度が
  すべて悪化した。task-alignedな専門化は別仮説へ分離した。

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
| [Search utility探索量安定性](results/search-utility-horizon-20260825/README.md) | H-SU-HORIZON, H-SU-GEOMETRY | 10kから1mへの候補順位転移を棄却し配備horizon整合へ更新 |
| [Expert多様性ボトルネック](results/expert-diversity-20260825/README.md) | H-EXPERT-DIVERSITY, H-TASK-ALIGNED-EXPERTS | 放射状多様化を棄却しtask-aligned専門化を分離 |
| [Search utility候補幾何](results/search-utility-geometry-20260826/README.md) | H-SU-GEOMETRY, H-SU-ADAPTIVE-RADIUS | 一律の広い候補を棄却しroot別半径を分離 |
| [Tail-sensitive group objective](results/tail-objective-20260826/README.md) | H-TAIL-OBJECTIVE | CVaRでtail lossを改善し固定bestmoveも小幅な正方向 |
| [Root-only adapter容量](results/root-representation-capacity-20260826/README.md) | H-ROOT-REPRESENTATION, H-ROOT-FEATURES | 単純な幅拡大を棄却し入力特徴仮説を分離 |
<!-- result-ledger:end -->
