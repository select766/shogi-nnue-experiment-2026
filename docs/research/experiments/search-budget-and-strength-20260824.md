# 実探索leaf routerの棋力・探索量転移検証計画 (2026-08-24)

## 仮説

1. `nodes=1000`の実探索leafで学習したrouterのgroup loss改善は、探索後の最善手一致率または
   自己対局scoreへ正方向に転換する。
2. 同じrouterは探索量を変えて得たqsearch leafでもcheckpoint 510よりgroup lossが低い。

実行時と同じくDNNへ探索rootを1回だけ入力し、そのgateを探索中の全局面で固定する。leafごとに
DNNを再実行しない。

## 指標の定義

- **group loss差**: 同じrootとその16 leafについて計算した候補の平均評価値lossから、checkpoint
  510の平均評価値lossを引いた値。負が候補の改善である。95%区間はroot単位bootstrapで求める。
- **最善手一致**: 固定test JSONLのHCPE教師手と、100万nodes探索の`bestmove`が同じであること。
  モデル間では同一局面の正誤を対応付け、正確McNemar検定を使う。
- **自己対局score**: 候補の`(勝数 + 0.5 * 引分数) / 対局数`。各開始局面で候補とcontrolの先後を
  交換する。Elo区間はscoreの標準誤差をEloへ変換した近似区間である。

## 条件

- 候補: `logs/proxy_gap_actual500k_curve_from510/checkpoints/4.ckpt`
- control: `logs/expert_blending_8experts_v4_paired_uniform50_noise0/checkpoints/510.ckpt`
- 最善手: 固定test 10,000局面、`Threads=1`、100万nodes。HalfKPとも対応あり比較する。
- 自己対局: validation JSONLから教師評価値絶対値500以下をseed `20260824`で抽出した200開始局面、
  先後paired 400局、`Threads=1`、10万nodes。
- leaf転移: baseline `nn.bin`探索器、250/1,000/4,000/16,000 nodesは20,000 root、
  100万nodesは独立範囲5,000 root、各16 leaf。教師は固定`nn.bin`のexact qsearch。

探索量転移は各nodesのgroup loss差95%区間が0未満なら支持とする。棋力転換は最善手の対応あり差または
自己対局Elo区間が正方向を支持する場合に確認とする。一方だけ正方向で区間が0を跨ぐ場合は未確認とする。
