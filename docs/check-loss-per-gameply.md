# game_plyの差によるlossの違いの検証

`scripts/run_train_expert_blending_8experts_v4_paired_noise0.sh` を用いて学習されたモデルファイル(checkpoint 160)をやねうら王で対局させたが、ベースラインとなる通常のNNUE(expert blending不使用)より弱い。
その原因として、深い探索における評価値が正しくないのではないかと予想している。

## 検証方法

学習機が出力するvalidation lossをより細かく分析できるようにする。
validationデータの各レコードにおいて、DNN用の局面のgame_ply=xと、NNUE用の局面のgame_ply=yを抽出する。
各レコードにおけるlossを(y-x)をキーとするbinに分割し、それぞれのbinにおける平均を計算し、出力する。グラフにプロットする。

同様に、expert blending不使用のモデル `logs/halfkp_v1/checkpoints/83000.ckpt` にを用いて同じlossを計算し、同じグラフ上にプロットする。
