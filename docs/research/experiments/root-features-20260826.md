# 明示root特徴ablation 実験計画 (2026-08-26)

対象は`H-ROOT-FEATURES`である。DNN backboneの盤面入力・容量・8 expertsを固定し、global average pooled
出力へ、rootで一度だけCPU計算できる少数特徴を明示的に連結する。探索終了までblendを固定する配備制約は
変えない。

## 特徴と条件

- `ply`: game plyのみ（1次元）。
- `phase`: 盤上駒率、持駒率、成駒率、合法手率、先後の駒数差、玉間Manhattan距離（6次元）。
- `combined`: 上記7次元。

全特徴はSFENとgame plyだけから決まり、教師scoreや探索後統計を含まない。adapterの最初の線形層へ連結し、
追加列を0初期化してM0 logitを厳密に保存する。追加パラメータは最大896個で、hidden幅128は固定する。

- initial: M0
- train: `tmp/proxy_gap_v2/train` 99,840 root、group size 8
- selection: valB 19,968 root、held-out: valA 19,968 root
- experts/backbone固定、mean group loss、adapter-only、1 epoch、seed 42
- matched control: `tail_objective_mean_from_m0/checkpoints/0.ckpt`

## 事前判定

1. valBでmean group lossがmatched controlより小さい特徴だけvalAへ進める。
2. valAの対応root loss差bootstrap 95%区間上端が0未満ならloss条件を通過する。
3. 通過時だけ20局面・1000 nodesでwall-clockを比較し、追加時間10%以内なら固定validation
   10,000局面・100万nodesへ進む。
4. bestmove差が正方向なら`支持`、0以下なら`棄却`。valA loss候補がなければ`棄却`する。

この実験が棄却する範囲は、明示game ply・静的phase特徴である。別途浅い探索統計を計算する方式までは
同じ判定へ含めない。
