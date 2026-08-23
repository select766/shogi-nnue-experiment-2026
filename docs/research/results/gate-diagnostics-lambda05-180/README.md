# Gate診断: 8 experts lambda=0.5 checkpoint 180

## 条件

- checkpoint: `logs/expert_blending_8experts_v4_paired_uniform50_noise0_lambda05/checkpoints/180.ckpt`
- validation: `dataset/split_v1_paired_uniform_50/val1`先頭10,000 paired positions
- objective: `lambda=0.5`, label smoothing `0.001`, score scaling `361`
- DNN backbone、8 experts、weighted blend
- command: `scripts/diagnose_gate.sh`
- result: `results/gate_diagnostics_8experts_lambda05_180.json`

## Gateの疎性と利用均衡

| 指標 | 結果 |
|---|---:|
| entropy平均 | 1.8071 |
| effective experts `exp(H)`平均 | 6.1860 |
| 最大重み平均 | 0.3052 |
| top-2 mass平均 | 0.5060 |
| argmax利用率CV | 1.1439 |
| 平均重みのCV | 0.3385 |
| 平均重みの一様分布からのKL | 0.0522 |
| dead experts | なし |

argmax利用率は`[0.98, 13.02, 8.47, 6.30, 49.39, 8.25, 5.38, 8.21]%`、
平均gate重みは`[12.99, 14.13, 12.64, 10.12, 22.34, 7.35, 9.55, 10.86]%`だった。
全expertへ平均重みは流れているが、局面別には約6.2 expertsを混ぜており、gateは密である。

## Expert機能差

同一局面に対するexpert単独評価値のexpert間相関は平均0.9655、範囲0.9405--0.9816だった。
expert出力は強く相関する。一方、局面ごとのexpert評価値分散は平均73,116 (`cp-like^2`)で、
局所的な評価差は残っている。

## Routing価値

| 指標 | loss / rate |
|---|---:|
| dense blended平均loss | 0.2503 |
| 現gate top-1平均loss | 0.2602 |
| 最良固定expert平均loss (expert 4) | 0.2544 |
| 単独expert oracle平均loss | 0.1612 |
| oracleのblended比改善上限 | 0.0891 |
| gate top-1とoracleの一致率 | 7.67% |
| 周辺分布からの偶然一致率 | 9.74% |
| 一致lift | 0.79 |

oracle利用率は`[4.25, 22.69, 11.50, 14.85, 3.06, 5.84, 9.42, 28.39]%`だった。
gateはexpert 4を49.39%の局面でtop-1にするが、expert 4がoracleなのは3.06%だけである。

現在のgateをそのままhard top-1化するとdense blendよりlossが悪化する。一方、単独expert oracleの
改善上限は大きいため、expertに局所差がないのではなく、routerが適切なexpertを選べていない
可能性が高い。温度softmaxでdense重みを鋭くする診断に加え、router教師あり蒸留やranking lossを
検討する根拠になる。expert間相関も高いため、多様性促進は副次候補として残る。

oracleは各局面の教師lossを見て事後選択した上限であり、実運用性能を直接表す値ではない。
