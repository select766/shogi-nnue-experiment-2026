# 実探索leaf lossの棋力追試 (2026-08-26)

## 結論

`H-LEAF-LOSS-STRENGTH`は測定完了とする。実探索leaf lossを大きく改善した候補を固定10万nodesで
4,000局測定した結果、checkpoint 510対照に1,950勝1,919敗131分、Elo差`+2.69`、95%区間
`[-7.90,+13.28]`だった。事前規則に従い、区間が0を跨ぐため方向は未確定である。局数を10倍にしても
正方向へ分離する仮説は確認できず、実探索leaf loss単独ではモデルを選抜しない。

## 既存の予測品質

候補は探索量250から100万nodesまで独立rootのgroup lossを改善した。最大の100万nodesでは差
`-0.003843`、root bootstrap 95%区間`[-0.004316,-0.003402]`で、改善rootは71.82%だった。一方、固定
10,000局面の最善手一致率は候補62.58%、対照62.78%、差`-0.20`ポイント、正確McNemar
`p=0.652110`だった。loss改善の存在と、固定rootのbestmove改善は同じ主張ではない。

## 自己対局

candidateは`tmp/proxy_gap_actual500k_release`、controlは`tmp/proxy_gap_control510_release`とした。
両者をCPU ONNX Runtime版やねうら王へ組み込み、各engine `Threads=1`、固定100,000 nodes/着手で測った。
2,000個の互いに異なる開幕を各1回先後交換した。

| 標本 | 局数 | candidate勝 | candidate敗 | 分 | Elo差 | 95%区間 |
|---|---:|---:|---:|---:|---:|---:|
| 既存200開幕 | 400 | 200 | 185 | 15 | +13.03 | [-20.39,+46.46] |
| 追加1,800開幕 | 3,600 | 1,750 | 1,734 | 116 | +1.54 | [-9.62,+12.71] |
| 固定総標本 | 4,000 | 1,950 | 1,919 | 131 | +2.69 | [-7.90,+13.28] |

集約器で、40入力chunkの各局数と勝敗和、detail件数、総4,000局、2,000 unique SFEN、各SFENのcandidate
先手・後手各1局、SFEN×手番の重複なしを検査した。途中結果による停止やモデル変更は行っていない。

## OOM事故と再発防止

初回の高並列実行中、2026-08-26 01:40:29に`systemd-oomd`がterminal scopeを終了した。journalには
324 process、user sliceのmemory pressure 77.65%が50%を20秒超過した記録がある。12並列では
`MemAvailable`が約5.8 GiB、6並列でも約12 GiBまで低下した。1対局は約490 MiBの`head.bin`を持つ
engineを2個起動し、ONNX Runtime等も加わるため、GPUではなくCPU対局プロセス数が原因だった。

再開後は4並列へ制限し、実測で`MemAvailable`約17--18 GiB、swap使用約2 GiBを維持した。実行scriptには
並列上限4、wave開始前20 GiB検査、完了chunk再利用、二重起動lock、最終集約の完全性検査を追加した。

## 解釈

実探索leaf lossは評価値分布への適合を測るが、探索の手選択は候補手間の相対順位、margin、探索木の展開と
相互作用する。今回の区間は小さな負効果と最大約`+13 Elo`の正効果をなお含むが、leaf lossの大幅改善から
棋力改善を推定するには不十分である。次は平均leaf lossではなく、rootでの意思決定に直接対応する目的を
独立仮説`H-DECISION-ALIGNED-LOSS`として検証する。

## 成果物

- 計画: `docs/research/experiments/leaf-loss-strength-20260826.md`
- 再開可能な対局script: `scripts/run_leaf_loss_strength_match.sh`
- 集約・完全性検査: `scripts/merge_match_results.py`
- 総結果: `results/match_leaf_loss_strength_4000.json`
- chunk結果: `results/match_leaf_loss_strength_chunk_00.json`から`_35.json`
- 既存400局: `results/match_proxy_gap_nodes100000_chunk0.json`から`chunk3.json`
