# Expert Blending 調査報告と次期実験計画 (2026年8月)

## 結論

- 2026-08-23の固定test 10,000局面による追試では、ベースライン62.87%に対して現在の
  採用候補は63.32%で、差は+0.45ポイントだった。対応ありの正確McNemar検定は
  `p=0.314924`であり、正方向だが優位性は確認できない。詳細は
  `docs/research/results/accuracy-eval-10k-current/README.md`に記録した。
- checkpoint 510からのモデル変更では、1.5-entmaxがvalidation lossを維持しつつ実効expert数を
  6.13から3.92へ下げた。しかし固定10,000局面の一致率は63.04%で、HalfKPとの差+0.17ポイント
  (`p=0.714084`)、dense checkpoint 180との差-0.28ポイント (`p=0.527044`)だった。疎化は達成したが
  指し手改善へは転換されていない。詳細は
  `docs/research/results/gate-model-change-20260823/README.md`に記録した。
- 現在の採用候補 (`uniform50 + lambda=0.5`, checkpoint 180) は、100万ノード・同一1,000局面でベースライン NNUE の 64.4% に対して 63.3% だった。ただし対応ありの正確 McNemar 検定は `p=0.439` で、有意な劣化とはいえない。
- 一方、gate が密すぎるという仮説には根拠がある。checkpoint 180 の validation gate entropy は 1.814 (`log(8)=2.079` の87.2%) で、実効 expert 数 `exp(H)` は約6.1だった。代表局面の最大重みも平均0.281に留まる。
- entropy・balance・entmaxによる疎化は完了し、疎化自体は達成したが指し手一致率を改善しなかった。
- per-pair router蒸留は改善しなかったが、末端ごとのteacherは探索時の固定重み制約と一致して
  いなかった。rootごとに8末端を共有する後続実験では、A末端で決めた固定teacherが未知のB末端へ
  転移し、root-grouped task-only学習も初期checkpoint比でlossを改善した。詳細は
  `docs/research/results/root-grouped-router-20260823/README.md`に記録した。
- root-grouped task-only学習を500万rootへ拡大した追試では、500万root時点の候補が初期checkpoint
  に対し、選抜用Bで`-0.0002374`、未使用Aで`-0.0002407`のgroup平均loss差となった。両方の
  root bootstrap 95%区間は上端も0未満だった。2巡目は反転悪化し、700万rootで事前定義した
  飽和条件を満たした。次は棋譜先局面proxyではなく実探索qsearch leafをrootごとに収集する。
  詳細は`docs/research/results/root-grouped-router-scale-20260824/README.md`に記録した。
- 最善手一致率は1万局面へ拡大し、対応あり比較を行う。現観測値から1.1ポイント差を検出する概算必要数は約10,833局面なので、まず10,000、確証が必要なら12,000以上を使う。

## 現状認識の根拠

### 最善手一致率

評価条件は `data/accuracy_eval/test.jsonl` の同一1,000局面、`Threads=1`、100万ノードである。95%区間は Wilson score interval。

| モデル | checkpoint | 一致数 | 正解率 | 95%区間 | ベースライン差 |
|---|---:|---:|---:|---:|---:|
| ベースライン HalfKP | 83000 | 644/1000 | 64.4% | 61.4--67.3% | - |
| 旧 paired v4 | 160 | 647/1000 | 64.7% | 61.7--67.6% | +0.3 pt |
| uniform-50 DNN gate | 400 | 625/1000 | 62.5% | 59.5--65.4% | -1.9 pt |
| uniform-50 NNUE gate | 1150 | 639/1000 | 63.9% | 60.9--66.8% | -0.5 pt |
| uniform-50 fine-tune (`lambda=0.5`) | 180 | 633/1000 | 63.3% | 60.3--66.2% | -1.1 pt |

採用候補とベースラインの局面別比較では、両方正解555、ベースラインのみ正解89、Expert Blendingのみ正解78、両方不正解278だった。対応ありの正確 McNemar 検定は `p=0.439`。旧 paired v4 の +0.3 pt も `p=0.880` であり、1,000局面ではモデル間の小差を判定できない。

この評価の「正解」は HCPE に格納された1手との一致である。同程度の複数候補手を区別できず、勝率を直接測る指標でもない。モデル選択に使える中間指標ではあるが、最終判断には自己対局が必要である。

関連ファイル:

- 評価方法: `docs/operations/accuracy-evaluation.md`, `src/train_nnue/eval_accuracy.py`
- ベースライン: `results/accuracy_eval_halfkp_v1.json`
- 各 Expert Blending 結果: `results/accuracy_eval_expert_blending_*.json`
- 現行設定: `configs/accuracy_eval_halfkp_v1.json`,
  `configs/accuracy_eval_expert_blending_current.json`
- 当時の設定: `configs/archive/accuracy_eval_*.json`

### validation loss

uniform-50 DNN gate checkpoint 400 は、100万 validation positions において全 `game_ply` 帯でベースラインより低いlossだった。差が大きい中盤の例は次の通り。

| nnue_ply bin中心 | Expert Blending | baseline | 相対改善 |
|---:|---:|---:|---:|
| 77 | 0.044960 | 0.046814 | 4.0% |
| 97 | 0.052447 | 0.054712 | 4.1% |
| 117 | 0.053659 | 0.055827 | 3.9% |

NNUE gate checkpoint 500 も全手数帯でベースラインを下回るが、改善幅はDNN gateより小さい。したがって、validation lossの改善が探索後の指し手一致率へ転換されていないことが問題であり、単なる学習不足とは断定できない。

`lambda=0.5` run の最終 `val_loss=0.2458` は、評価値lossと勝敗lossを混合した別目的関数なので、`lambda=1.0` run の値と直接比較しない。

関連ファイル:

- DNN gate: `docs/research/results/check-loss-per-gameply-dnn-backbone-v4/README.md`
- NNUE gate: `docs/research/results/check-loss-per-gameply-nnue-backbone/README.md`
- 集計実装: `src/train_nnue/check_loss_per_gameply.py`

### gate の選ばれ方

checkpoint 180 の10,000 validation positionsに対する `argmax` 担当割合は、expert 4が49.4%、expert 1が13.0%、残りは1.0--8.5%だった。序盤、入玉、相振り飛車、穴熊などとのliftも観測されており、`argmax`の意味では専門化が生じている。

しかし実際の合成重みは密である。

- TensorBoard上のgate entropy: 1.814。最大値 `log(8)=2.079` で正規化すると87.2%。
- entropyから見た実効expert数: `exp(1.814)=6.14`。
- 代表80局面の担当expert重み: 平均0.281、範囲0.236--0.363。
- validationでの平均重み: `[0.127, 0.137, 0.132, 0.102, 0.210, 0.076, 0.097, 0.120]`。

つまり「担当expertは局面ごとに変わる」が「各局面では6個前後を広く混ぜる」状態である。次の実験では、この2種類の性質を別指標として追う必要がある。

関連ファイル:

- 可視化結果: `results/visualize_experts_8experts_lambda05_180/index.html`, `lift_analysis.json`
- 可視化仕様: `docs/operations/expert-analysis.md`
- gate実装: `src/train_nnue/expert_blending_model.py` (`DNNAdapter.forward`)
- entropy記録: `src/train_nnue/train_expert_blending.py`

## ディスク整理

2026-08-23に約1.0 TiBの再生成可能な成果物を削除した。HDD使用率は70%から33%へ低下し、空きは約839 GiBから約1.9 TiBになった。

削除したもの:

- `dataset/split_v1` (約299 GiB表示): qsearch済み旧単一局面形式。
- `dataset/split_v1_paired` (約54 GiB表示): 旧paired形式。
- `dataset/split_v1_paired_uniform_50_old` (約36 GiB表示): 旧uniform-50形式。
- `logs/*_old` 2 run。
- 評価に使った代表checkpoint以外の数値付き中間checkpoint 307個。
- `tmp/test_*_gpu_smoke`, `tmp/residual_smoke_run_cpu`。

保持したもの:

- 主学習データ `dataset/split_v1_paired_uniform_50` (約550 GiB表示)。
- 再生成元 `dataset/tanuki-.nnue-pytorch-2024-07-30.1` (約299 GiB表示)。
- 評価・履歴資料で参照する数値付きcheckpoint 10個: 83000、v2 160、paired v4 160、400、510、180、500、1150、residual 300、16-expert residual 220。
- 各runの `final.ckpt`、TensorBoard events、結果JSON、設定、コード。

整理後の `logs` は約60 GiB、checkpointは34個・58.8 GiBである。削除物はゴミ箱を経由しておらず、必要なら元データとスクリプトから再生成する。

## 関連研究

1. Shazeer et al., [Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer](https://arxiv.org/abs/1701.06538) (2017): noisy top-k gateと利用均衡の補助損失を導入した。局面ごとの疎性と全体の負荷均衡を同時に扱う基本形として直接関係する。
2. Fedus et al., [Switch Transformers](https://www.jmlr.org/papers/v23/21-0998.html) (JMLR 2022): top-1 routingと補助的load-balancing lossを採用した。低entropy化だけで起きるexpert collapseへの対策根拠になる。
3. Martins and Astudillo, [From Softmax to Sparsemax](https://proceedings.mlr.press/v48/martins16.html) (ICML 2016): softmaxと異なり厳密なゼロを出せ、backpropagation可能な確率写像を示した。
4. Peters et al., [Sparse Sequence-to-Sequence Models](https://aclanthology.org/P19-1146/) (ACL 2019): softmaxとsparsemaxの間を連続化する entmax、とくに計算しやすい 1.5-entmaxを示した。急なhard top-kより穏当な第一候補になる。
5. Hazimeh et al., [DSelect-k](https://papers.nips.cc/paper/2021/hash/f5ac21cd0ef1b88e9848571aeb53551a-Abstract.html) (NeurIPS 2021): 選択expert数を明示できる滑らかな疎gate。entmaxで不十分な場合の次候補である。
6. Zoph et al., [ST-MoE](https://arxiv.org/abs/2202.08906) (2022): routerの数値安定性とload balancingを検討した。疎gate学習時にはrouter logitsとentropyを監視すべき根拠になる。

これらは主に大規模言語・推薦モデルの結果であり、NNUE重みを一度だけ合成する本方式へ性能結果を直接外挿はできない。ただし、疎性とcollapse防止を分離する設計原理は適用できる。

## 次の実験

### 1. 評価基盤を先に固定する

実装状況 (2026-08-23): `data/accuracy_eval_10k/`へvalidation/test各10,000件を固定し、
旧3,000件とsplit間のSFEN重複がないことを確認した。`eval_accuracy`はWilson区間と層別、
`compare_accuracy`は局面対応を検査したdiscordant countsと正確McNemar検定を出力する。
最終候補用には`scripts/eval_match.sh`で固定ノード・開始局面ごとの先後ペア・Elo近似区間を
保存できる。HCPEに実手数がないため、現データの`game_ply`層別は利用不可と明示し、
実手数付きJSONLを与えた場合だけ集計する。

1. HCPE全体から、既存1,000件と重複しない固定test 10,000件を作る。ハイパーパラメータ選択には別のvalidation subsetを使い、testを反復調整に使わない。
2. 全モデルを同一局面・`Threads=1`・100万ノードで評価する。
3. 正解率とWilson区間に加え、ベースラインとのdiscordant countsと正確McNemar検定を必ず出す。
4. `game_ply`、教師評価値絶対値、終盤/入玉などで層別集計する。
5. 最終候補だけ固定ノード自己対局へ進め、勝率またはElo区間で判断する。

10,000件は現状より大幅に良いが、現在のdiscordance率16.7%で1.1 pt差を両側5%・power 80%で捉える概算は10,833件である。結果が境界的なら12,000件以上へ増やす。

### 2. gate診断を追加する

実装状況 (2026-08-23): `scripts/diagnose_gate.sh`を追加し、以下の全指標を
validation先頭10,000局面から`results/gate_diagnostics_8experts_lambda05_180.json`へ保存した。
checkpoint 180ではentropy平均1.807、実効expert数平均6.19、最大重み平均0.305で、
gateが密という従来判断を再確認した。expert単独評価値のexpert間相関は平均0.966、
gate top-1のoracle一致率は7.67%だった。周辺分布から期待される偶然一致率9.74%に対する
liftも0.79であり、top-1 routerは単独expert oracleと整合していない。blended平均loss 0.2503に対する単独expert oracleの
平均lossは0.1612で、routingを改善できる余地はある。ただしexpert間相関は高く、routerと
expert多様性の両方が制約候補である。指標定義は`docs/operations/expert-analysis.md`を参照する。
詳細結果は`docs/research/results/gate-diagnostics-lambda05-180/README.md`に保存した。

各validation epochまたは独立10,000局面で次を保存する。

- 局面別: entropy `H(p)`, `max(p)`, top-2 mass, effective experts `exp(H)`。
- 全体: `argmax`利用率、平均gate重み、利用率のCV、平均重みの一様分布からのKL。
- expert機能差: 同一局面に対する各expert単独評価値の相関と分散。
- routing価値: 最小loss expertをoracleとした場合の改善上限と、gate top-1のoracle一致率。

これにより「gateが曖昧」「experts自体が似ている」「gateが誤ったexpertを選ぶ」を区別できる。

### 3. 最初のモデル変更

実装・実験状況 (2026-08-23): 温度4条件、entropy＋balance 8条件、1.5-entmax＋balance
3条件をcheckpoint 510初期値で比較した。温度とentropy正則化には選抜基準を通る条件がなく、
entmax・balanceなしを6 epoch fine-tuneした候補だけが、独立10,000 validation positionsで
loss 0.0337878、実効expert 3.92、死expertなしとなった。この候補の固定10,000局面一致率は
63.04%で、HalfKP比+0.17ポイント (`p=0.714084`)、dense checkpoint 180比-0.28ポイント
(`p=0.527044`)だった。validation上の疎化は実探索の改善へ転換されなかったため、hard top-kへは
進まなかった。その後のrouter蒸留も改善しなかったため、次はrouter入力の情報量を見直す。

8 experts、DNN backbone、uniform-50、checkpoint 510初期値を固定し、次の順で比較する。

1. **温度softmaxの安価な診断**: `T = 1.0, 0.75, 0.5, 0.25` をvalidationで比較する。まず再学習なしで「鋭くするだけでlossが改善するか」を確認し、有望な温度のみfine-tuneする。
2. **entropy + balance**: 学習lossを次にする。

   `L = L_NNUE + lambda_sparse * mean(H(p_i)) + lambda_balance * KL(mean_i(p_i) || Uniform)`

   第2項は各局面の重みを鋭くし、第3項は全局面が同じexpertへ崩壊するのを防ぐ。これは「条件付きentropyを下げ、周辺entropyを保つ」、すなわち局面とexpert選択の相互情報量を増やす考え方に相当する。
3. **1.5-entmax + balance**: softmaxの温度調整で不十分なら、厳密なゼロを出す1.5-entmaxを比較する。
4. **hard top-k / DSelect-k**: 上記で改善傾向が出た場合だけ `k=1,2` を試す。最初からhard routingにすると同一初期値expert間の対称性とcollapseの影響が大きい。

最初のsweepは `lambda_sparse = {0, 1e-4, 1e-3, 1e-2}`、`lambda_balance = {1e-3, 1e-2}` 程度の短いrunに限定する。選抜条件は、baselineより低いvalidation lossを維持しつつ、`exp(H)`が現状6.1から2--4へ下がり、死んだexpertがないこと。正解率10,000件はこの条件を通った少数候補だけに実行する。

### 4. 判断基準

実装・実験状況 (2026-08-23): router蒸留は3段階で検証した。単一expert hard/soft教師では
oracle top-1一致が20.43%から最大28.96%へ上がった一方、validation lossは0.0338493から
0.0386505へ悪化した。勾配scaleを合わせたtaskとの混合も改善せず、実際のblend lossを局面ごとに
最適化したgate教師でも最良0.0338608だった。このper-pair結果からは候補なしとしたが、後続の
root-grouped task-only学習では固定root gateの有効性が確認された。

- **継続**: validation lossを悪化させず、gateが明確に疎になり、10,000局面の対応あり差が正方向。
- **設計変更**: oracle teacherには改善余地があるのに蒸留後のtask lossが改善しない場合。routerへ
  NNUE側の実局面特徴を直接与えるなど、入力情報を見直す。
- **expert側を変更**: expert単独出力の相関が極端に高くoracle改善も小さい場合。出力空間の多様性を直接促す方法を検討する。
- **中止**: validation loss改善が自己対局へ一貫して転換せず、信頼区間上も実用差がない場合。

16 experts化や新backboneの大規模学習は、この8-expert診断が終わるまで優先しない。
