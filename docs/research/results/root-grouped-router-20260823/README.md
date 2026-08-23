# Root-grouped router実験 (2026-08-23)

## 結論

探索ルートでDNNを一度だけ実行し、その重みを複数末端へ固定する制約を保ったデータと学習を実装した。
同じルートの末端集合Aで最適化した共有gateを、独立に抽出した末端集合Bへ適用すると、現行router比で
group平均lossが`0.0378021`から`0.0307631`へ改善した。root別差のbootstrap 95%区間も全域が負であり、
「ルートごとに未知末端へ転移する固定blendが存在する」という仮説は支持された。

共有teacherへのcross entropy蒸留は、router-to-shared-teacher top-1 matchを上げたがtask lossを
改善しなかった。一方、同じroot-groupedデータでtask lossを直接学習すると、2 epoch相当の最良モデルは
`0.0375596`となり、初期checkpointとの差は`-0.0002426`、95%区間
`[-0.0003522, -0.0001440]`だった。固定root gateの学習は有効だが、teacher分布を一様なCEで近似する
補助目的は、損失感度の大きい少数rootを扱えていない。

## 用語

完全な定義は[実験計画](../../experiments/root-grouped-router-20260823.md)に記載した。本結果では、
単独の「一致率」という語を使わない。

- **Shared-teacher argmax agreement (A/B)**: 同じrootの独立末端集合A、Bから得た共有teacherについて、
  最大重みexpert番号が等しいrootの割合。
- **Router-to-shared-teacher top-1 match**: rootだけを入力したrouterと共有teacherの最大重みexpert番号が
  等しいrootの割合。指し手一致率ではない。
- **Best-move agreement**: エンジン最善手とHCPE教師指し手の一致。本実験では未測定。

## データ

- train: 100,000 roots x 8 qsearch leaves。
- validation: 10,000 roots x 16 leavesを生成し、前半8件をA、後半8件をBとした。
- offset: rootから1--50手先を一様・復元抽出。
- 8抽出中の異なるoffset数はtrain平均7.01、validation A/B平均7.03/7.02。復元抽出による重複を
  含むが、現行の条件付き一様分布を変えないためそのまま使用した。
- trainと同一root SFEN 156件、validation内重複3件を除き9,841 unique rootsを保持。
- 正式集計: batch境界を揃えた先頭9,728 roots。
- 保存先: `tmp/root_grouped_router_pilot/`。

形式は`roots.bin`にrootを1回、`leaves.bin`にgroup-majorのK末端、`offsets.npy`に棋譜上の
手数差を保存する。学習時はroot batchのgateを`repeat_interleave(K)`し、同じ重みをK末端へ適用する。
探索時の追加DNN呼び出しは必要ない。

## 共有teacherの存在と安定性

checkpoint 510を初期値とし、1 rootにつき1つのgate logitsを8末端の平均blend lossへ10 Adam steps
最適化した。元gateへのKL係数は0.01。

| 測定 | group平均loss |
|---|---:|
| A上の現行router | 0.0372254 |
| A上のA teacher | 0.0239364 |
| B上の現行router | 0.0378021 |
| B上のB teacher（同一集合上限） | 0.0244357 |
| B上のA teacher（未知末端への転移） | **0.0307631** |

A teacherと現行routerのB上のroot別loss差:

- 平均: `-0.0070391`
- root bootstrap 95%区間: `[-0.0074797, -0.0066138]`
- 改善root割合: 73.96%
- Shared-teacher argmax agreement (A/B): 64.20%
- teacher分布のJensen-Shannon divergence平均: 0.03663

最大重みexpertが完全一致しないrootでも、分布全体としてA teacherはBへ明確に転移した。

## Router学習

8 expertsを固定し、DNN adapter 25,736 parametersだけを学習した。1 epochは99,840 roots、
root batch size 64、各root 8 leaves、adapter LR 0.003。蒸留係数は初期batchの勾配normから
0.25倍=`0.009580039`、1倍=`0.038320158`とした。

### 1 epoch screening

| 条件 | B group loss | Router-to-shared-teacher top-1 match | teacher CE |
|---|---:|---:|---:|
| task-only | **0.0376648** | 55.32% | 1.76332 |
| task + teacher 0.25x | 0.0376955 | 56.32% | 1.75687 |
| task + teacher 1.00x | 0.0377457 | 56.90% | 1.75393 |
| teacherのみ | 0.0379120 | **57.20%** | **1.75062** |

0.25倍条件はtask-onlyより64%のrootでlossを改善したが、少数の大きな悪化により平均差は
`+0.0000307`、root bootstrap 95%区間は`[+0.0000068, +0.0000566]`だった。teacher追従を
強めるほどtask lossが単調に悪化した。

### 追加学習

task-onlyと0.25倍条件を同じ学習量で延長した。

| 合計epoch相当 | task-only | task + teacher 0.25x |
|---:|---:|---:|
| 1 | 0.0376648 | 0.0376955 |
| 2 | **0.0375596** | 0.0376227 |
| 3 | 0.0384055 | 0.0380846 |

3 epoch目は両条件とも反転悪化した。最良のtask-only 2 epoch相当を再現し、
`logs/root_grouped_best_task_total2/lightning_logs/version_0/final.ckpt`へ保存した。

初期checkpoint 510と最良task-onlyのroot別対応比較:

- 平均loss差: `-0.0002426`
- root bootstrap 95%区間: `[-0.0003522, -0.0001440]`
- 改善root割合: 60.44%
- Router-to-shared-teacher top-1 match: 55.24%から54.98%へ微減

task loss改善はteacher top-1分類の改善を経由していない。soft blend全体の微調整が平均末端lossを
改善したと解釈する。

## 判断

1. ルートで一度決めた固定blendが複数の未知末端へ有効という仮説は通過。
2. shared teacherを通常のcross entropyで蒸留する方式は不採用。
3. root-grouped task-only checkpointを、別のroot集合または実探索末端で追試する価値がある。
4. 次のteacher補助損失を試すなら、rootごとの改善上限・局所曲率・大損失tailを反映した重み付けが必要。
5. 棋譜先局面は実探索末端のproxyなので、固定10,000局面の指し手評価より先に、実探索から収集した
   root-to-qsearch-leaf集合でloss改善を確認するのが最も直接的である。

機械可読結果:

- `results/root_grouped_teacher_stability.json`
- `results/root_grouped_router_short1_from510.json`
- `results/root_grouped_router_checkpoint_comparison.json`
- `results/root_grouped_task_checkpoint_comparison.json`
