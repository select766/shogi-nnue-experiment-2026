# Router蒸留実験 (2026-08-23)

## 結論

checkpoint 510の8 expertsを固定し、DNN adapterだけを3 epoch学習した。単一expert oracleの
hard/soft教師、task lossとの混合、実際のblended NNUE lossを局面ごとに最適化したgate教師の
いずれも、task-only対照のvalidation lossを改善しなかった。teacher追従を強めるほどoracle一致は
上がったが、単一expert regretとblend lossは悪化したため、候補選抜と固定test評価には進めない。

この結果は「良いrouting targetが存在しない」ことを意味しない。最適化gate teacher自身は
validation 20,480局面で平均lossを0.0347783から0.0154880へ下げた。ただし、このteacherは末端を
見た後でペアごとに別の重みを選ぶため、探索ルートで一度決めた重みを全末端へ固定する実運用と
粒度が一致しない。後続のroot-grouped実験では、複数末端で共有するteacherが未知末端へ転移することを
確認した。詳細は`../root-grouped-router-20260823/README.md`を参照する。

## 共通条件

- 初期値: `expert_blending_8experts_v4_paired_uniform50_noise0/checkpoints/510.ckpt`
- データ: `split_v1_paired_uniform_50`
- experts: 8、固定
- 学習対象: DNN adapter 25,736 parameters
- 1 epoch: 100,000 positions、3 epochs、batch size 256
- validation: `val1`先頭10,000 positions
- adapter LR: 0.003、seed 42

teacher cacheとpaired binaryの位置対応は、先頭および`start_position=256`で照合し、最大誤差0を
確認した。expert loss cacheの全要素がfiniteで、train 1,000,000件のoracle担当率は
`[10.72, 15.46, 10.11, 8.56, 13.31, 8.00, 8.03, 25.81]%`となり、教師側の単一expert collapseもない。

## 1. 単一expert oracle蒸留

各局面・各expertの単独lossをcacheし、最小loss expertのone-hotをhard教師、
`softmax(-loss / temperature)`をsoft教師とした。

| 条件 | 最終val loss | oracle top-1一致 | 期待一致 | regret | 実効experts |
|---|---:|---:|---:|---:|---:|
| task-only対照 | **0.0338493** | 20.43% | 13.73% | **0.03551** | 6.13 |
| hard、蒸留のみ | 0.0386505 | **28.96%** | 19.64% | 0.06069 | 6.73 |
| soft、T=0.005、蒸留のみ | 0.0353669 | 25.49% | 17.43% | 0.05086 | 7.76 |
| soft、T=0.020、蒸留のみ | 0.0350802 | 22.12% | 15.82% | 0.04062 | 7.89 |

分類としてのoracle一致は改善したが、間違えた局面のregretが増え、blend目的と一致しなかった。

## 2. Task lossとの混合

同じ初期batchでadapterに対するtask勾配normは0.0145290だった。soft教師の蒸留勾配に対して
0.25倍、1.00倍となる係数を使用した。

| 条件 | router係数 | 最終val loss | oracle一致 | regret |
|---|---:|---:|---:|---:|
| soft T=0.005、0.25x | 0.004507 | 0.0338731 | 20.12% | 0.03741 |
| soft T=0.005、1.00x | 0.018028 | 0.0340246 | 18.96% | 0.03833 |
| soft T=0.020、0.25x | 0.004242 | 0.0338730 | 20.06% | 0.03749 |
| soft T=0.020、1.00x | 0.016970 | 0.0340415 | 18.52% | 0.03839 |

最弱の補助損失でもtask-only対照を上回らず、強くすると一貫して悪化した。

## 3. Blended-loss最適化gate teacher

単一expert分類の目的ずれを避けるため、checkpoint 510のgateを初期値として各局面のgate logitsを
10 Adam steps (`lr=0.1`) 最適化した。目的は実際のblended NNUE lossに、元gateへのKLを0.01倍で
加えたもの。生成したteacherの統計は次の通り。

| split | positions | 初期平均loss | 最適化後平均loss | teacher entropy |
|---|---:|---:|---:|---:|
| train | 400,000 | 0.0331513 | 0.0142148 | 1.68694 |
| val1 | 20,480 | 0.0347783 | 0.0154880 | 1.68629 |

task勾配norm 0.0145290、teacher cross entropy勾配norm 0.1261112から、補助係数を
0.0288020 (0.25x) と0.1152081 (1.00x) に設定した。

| 条件 | 最良val loss | 最終val loss | oracle一致 | regret | 実効experts |
|---|---:|---:|---:|---:|---:|
| teacherのみ | 0.0339812 | 0.0340219 | 22.83% | 0.03802 | 6.21 |
| task + teacher 0.25x | **0.0338525** | **0.0338608** | 21.14% | **0.03601** | 6.20 |
| task + teacher 1.00x | 0.0338752 | 0.0339022 | 21.95% | 0.03648 | 6.20 |

最良条件もtask-only対照の最終0.0338493より0.0000116高く、実験誤差を論じる以前に採用基準を
満たさない。独立validation窓や固定10,000局面testを消費する候補はない。

## 判断

- hard/softな単一expert rankingをさらに調整する優先度は低い。既にoracle一致の上昇とtask悪化が
  同時に観測され、目的の不一致が明確である。
- per-pair blended-loss teacherは探索時に実現不能なoracleなので、root単位で複数末端を共有する
  teacherへ置き換える。

機械可読な集計は`results/router_distillation_short3_from510.json`、
`results/router_distillation_combined_short3_from510.json`、
`results/router_gate_teacher_short3_from510.json`に保存した。
