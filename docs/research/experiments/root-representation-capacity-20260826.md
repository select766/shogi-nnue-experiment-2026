# Root-only adapter容量 実験計画 (2026-08-26)

対象は`H-ROOT-REPRESENTATION`である。DNN backbone、入力特徴、8 experts、rootで1回だけDNNを実行して
探索中blendを固定する配備条件を変えず、adapter hidden幅だけを128から256/512へ増やす。

## 初期化と学習

M0の128 hidden unitを2回または4回複製し、出力層の対応weightを複製数で割る。ReLU後のlogitは
float誤差内でM0と同一だが、複製unitは学習後に別々の値を取れる。幅変更前に同一入力でlogitが一致する
ことを回帰テストする。

- initial: M0
- train: `tmp/proxy_gap_v2/train` 99,840 root、group size 8
- validation B/A: 各19,968 root
- experts/backbone固定、mean group loss、adapter LR 0.01、momentum 0.9、1 epoch、seed 42
- matched control: hidden 128で同じ追加学習をした`tail_objective_mean_from_m0/checkpoints/0.ckpt`

## 事前判定

1. valBでmean group lossがmatched controlより改善した幅だけvalAへ進める。
2. valAのroot別mean loss差bootstrap 95%区間上端が0未満ならloss条件を通過する。
3. export後のCPU ONNX Runtime + 1000 nodes探索を同じ20局面で測り、control比wall-clock増加10%以内なら
   配備速度条件を通過する。
4. 両条件を通った最良幅を固定validation 10,000局面・100万nodesでcontrolと対応比較する。

bestmove一致率差が正方向なら`支持`、0以下なら`棄却`する。lossまたは速度条件を通る候補がなければ
`棄却`する。
