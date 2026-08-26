# Root予測可能なexpert専門化 実験計画 (2026-08-26)

対象は`H-TASK-ALIGNED-EXPERTS`である。現expert偏差を単純拡大せず、game plyだけから再現できる8個の
探索phase役割を固定し、通常のblend品質を保ちながら役割expertへ教師あり補助lossを与える。

## 役割と学習

role `k`はgame ply `[32k+1, 32(k+1)]`、role 7は225手以降も含む。これはrootで探索前に正確に得られ、
学習後のcluster再割当はしない。M0のgate argmaxは10,000局面でexpert 4が99.1%を占めるため、argmaxを
役割教師には使わない。

- initial: M0
- train: `tmp/proxy_gap_v2/train` 99,840 root x 8 leaf
- selection: valB 19,968 root、independent check: valA 19,968 root
- adapter/backbone固定、expertsのみ1 epoch、root batch 64、通常mean group lossのweight 1
- role expert補助weight: `0.02, 0.05`
- matched control: 同条件でrole weight 0

role補助lossはroot内8 leafを、そのrootのone-hot role expert単体で評価した平均KLである。通常blend lossも
同時に最適化し、単体品質を壊すだけの専門化を避ける。

## 事前判定

1. valB blended lossがmatched control比`+0.0002`以内の候補だけvalAと単体expert診断へ進める。
2. valAで8 expert平均単体lossがM0比`+0.002`以内、かつ担当phase lossがmatched controlより改善する
   候補だけsearch utilityへ進める。
3. 未使用valA先頭64 rootでcandidate/referenceとも1,000,000 nodes、基準+8 axis候補を比較する。
4. M0より有効教師率と候補手多様度がともに正方向なら`支持`、いずれかが0以下なら`棄却`する。

候補数と探索量はM0対照と同一にし、expert品質、専門化、routing utilityを別指標として保存する。
