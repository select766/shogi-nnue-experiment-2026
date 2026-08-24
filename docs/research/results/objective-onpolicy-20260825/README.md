# Search-aware objective / on-policy pilot 結果 (2026-08-25)

## 目的

次の2仮説を、DNNはrootで1回だけ実行しblendを探索終了まで固定する配備条件のまま検証した。

- H1: qsearch教師より深い探索教師で実探索leafを学習すると、bestmove一致へ転換する。
- H2: baseline探索木より現在router M0自身の探索木で学習すると、その探索分布で改善する。

2×2条件はC00=baseline/qsearch、C10=baseline/deep、C01=on-policy/qsearch、
C11=on-policy/deepとした。全条件はcheckpoint 510からadapterだけを100,000 root、1 epoch学習した。

## データ

trainは100,000 root × 8 leaf、validationは同じ18,000 root × 16 leafである。on-policyで
採用されたroot recordをbaseline側から40-byte完全一致で抽出したため、各policy間でDNN入力は同じである。
同一rootのbaseline/on-policy leaf集合Jaccardは平均0.04616であり、探索policyによる末端分布差は大きい。

deep教師は既存qsearch PV終端から固定`nn.bin`で10,000 nodes探索し、そのPV終端へscoreを手番補正して
付けた。raw scoreは各終端のside-to-move視点なので、異なるPV終端間の意味比較には黒視点へ正規化した。
黒視点でのqsearch/deep教師整合は次の通りだった。

| 分布 | leaf score相関 | 非mate相関 | root平均相関 | 符号一致 |
|---|---:|---:|---:|---:|
| train baseline | 0.7824 | 0.9372 | 0.8842 | 91.51% |
| train on-policy | 0.7894 | 0.9360 | 0.8866 | 91.52% |
| val baseline | 0.7880 | 0.9345 | 0.8988 | 91.69% |
| val on-policy | 0.7912 | 0.9390 | 0.8996 | 91.61% |

実装中、USIの`mate -0`を整数化すると負号を失う問題を検出した。文字列先頭の負号を保持する修正と
回帰テストを入れ、誤って生成したdeepデータは全削除・再生成した。

## 交差loss評価

18,000 rootのうち17,920 rootを使い、各checkpointを4 validation分布で評価した。以下は平均
root-group lossである。

| model | baseline/qsearch | baseline/deep | on-policy/qsearch | on-policy/deep |
|---|---:|---:|---:|---:|
| checkpoint 510 | 0.022738 | 0.148901 | 0.022278 | 0.153712 |
| M0 | **0.017631** | 0.137536 | **0.017229** | 0.141979 |
| C00 | 0.018800 | 0.139279 | 0.018390 | 0.143795 |
| C10 | 0.021329 | **0.125694** | 0.021018 | **0.129688** |
| C01 | **0.018756** | 0.139174 | **0.018350** | 0.143693 |
| C11 | 0.021502 | **0.125604** | 0.021192 | **0.129580** |

H2の直接差C01−C00はon-policy/qsearchで−0.0000395
(95% bootstrap CI −0.0000452..−0.0000337)、baseline/qsearchでも−0.0000438
(−0.0000496..−0.0000383)だった。差は有意だが分布特異的でなく効果量も小さい。

H1の直接差はC10−C00がbaseline/deepで−0.013585、baseline/qsearchで+0.002529、
C11−C01がon-policy/deepで−0.014112、on-policy/qsearchで+0.002841だった。深い教師内の改善と
qsearch指標の悪化が明瞭に両立している。

## bestmove評価

固定validation 10,000局面、Threads=1、1,000,000 nodesで4条件を評価した。ここでいう一致率は、
各HCPE局面に保存された教師bestmoveと、固定blendを使うやねうら王が返すbestmoveが同一である局面の
割合を指す。H1はC00対C10およびC01対C11、H2はC00対C01を同一局面の対応ありMcNemar検定で判定した。

| 比較 | baseline | candidate | 差 | exact McNemar p |
|---|---:|---:|---:|---:|
| H1 baseline: C00→C10 | 62.35% | 60.73% | −1.62 point | 0.000372 |
| H1 on-policy: C01→C11 | 62.31% | 61.17% | −1.14 point | 0.0126 |
| H2: C00→C01 | 62.35% | 62.31% | −0.04 point | 0.942 |

H1はdeep lossを大きく改善したがbestmove一致を有意に悪化させたため、この10k-node leaf score教師は
打ち切る。これは「より深いleaf score」がrootの指し手目的を近似するという仮説を支持しない。

H2はqsearch lossで小さい有意差があった一方、bestmoveは実質同等だった。効果の飽和を切り分けるため、
deep教師を使わないon-policy/qsearchだけを約500,000 rootへscaleし、既存baseline-policy 500,000 root
モデルM0と同じ学習量で比較する。

## H2 scale

pilot 100,000 rootに非重複source区間399,840 rootを追加し、499,840 root × 8 leafを作った。
source record消費数は499,878、skipは38だった。root recordの重複623件は元500,000 rootにも同数あり、
区間重複による増加はなかった。M0と同じcheckpoint 510から約500,000 root/epochを5 epoch学習した。

epoch 4はM0 checkpoint 4に対し、baseline/qsearch lossを0.01763063から0.01762204へ改善した。
対応差は−0.00000858 (95% bootstrap CI −0.00001427..−0.00000346)だった。on-policy/qsearchでも
0.01722908から0.01722191、差−0.00000718 (−0.00001272..−0.00000214)で同方向だった。

一方、固定validation 10,000局面のbestmove一致はM0 61.91% (6,191件)、scale epoch 4は
61.89% (6,189件)だった。M0のみ正解850、scaleのみ正解848、差−0.02 point、exact McNemar
`p=0.981`である。50万root×5 epochでもlossの極小改善はbestmoveへ転換しなかった。

したがってH2もこの規模で飽和・打ち切りとする。最終候補はないため、固定test 10,000局面と
自己対局には進めない。次に検証するならleaf score近似の精緻化でなく、root候補blendを実探索して
得るsearch utilityを直接教師にする必要がある。
