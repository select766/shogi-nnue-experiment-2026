将棋AIについて、以下のアイデアをもとにモデルを学習したい。

目的: 局面に応じた評価関数を動的に生成することで、同じ計算コストでより強い手を指せることを示す。

ここで構築するモデルを、Expert Blendingと呼ぶことにする。

# 仕組み

## モデル構造
DNNとNNUE形式のモデルを組み合わせる。
DNNモデル: AlphaZero方式の、局面情報をテンソルとして入力し、指し手ごとの確率を表すpolicyベクトルと局面に対するスカラーのvalueを出力するモデルを想定する。局面テンソルをstate_tensor (shape: 9✕9✕チャンネル)とすると、以下のようにバックボーンとヘッドがある構造で表現される。
```
feat = DNN_backbone(state_tensor) # shape (batch, 9, 9, feature_channel) ※NHWCのケース。実際の実装はNCHWかもしれない。
policy = DNN_policy_head(feat)
value = DNN_value_head(feat)
```

実際のモデルとして、AobaZero http://www.yss-aya.com/aobazero/ で配布されている実装と学習済みモデルを使いたい。

NNUEモデル: NNUEモデルをN_EXPERTS個学習させる。
NNUEに与える盤面特徴量をstate_kp、i番目のモデルの重みをNNUE_weight[i]とし、関数NNUE(state_kp, NNUE_weight[i])が局面に対するスカラーのvalueを出力する。（学習可能な重みはNNUE_weight[i]側にあり、NNUEという関数自体は学習可能なパラメータを持たない）
学習済みモデルとして、 `logs/halfkp_v1/checkpoints/83000.ckpt` =NNUE_initial_weightを全モデルの初期値として使う。

これらのモデルを連結し、以下のようなモデルを考える。
```
# input: 1つの局面に対するstate_tensor, state_kp
feat = DNN_backbone(state_tensor)
weight = softmax(DNN_adapter(feat) + noise) # shape (N_EXPERTS,), sum=1
averaged_NNUE_weight = np.zeros((shape_of_weight))
for i in range(N_EXPERTS):
  averaged_NNUE_weight += weight[i] * NNUE_weight[i]
value = NNUE(state_kp, averaged_NNUE_weight)
# output: 入力局面に対するvalue
```

つまりは、N_EXPERTS個のNNUE重みを、局面に依存した重み付け和して用いる。Mixture-of-Experts的なことができるはず。
DNN_adapterは、featを受け取って、全結合層2層程度でN_EXPERTS次元のベクトルを出力する新しいモデル。
DNN_backboneはおそらく重みを固定して、DNN_adapterとNNUE_weightだけを学習対象にするのが良いと思われる。
このモデルを、nnue-pytorchで、局面to評価値の教師データで教師あり学習する。
ここで、noiseを加えているのは、学習されない重みが出ないようにする正則化。適切な分散の正規分布を想定している。

## 対局時
対局時には、 **ルート局面に対して** 上記の方式でaveraged_NNUE_weightを計算する。この重みを探索中はずっと用いる。これにより、やねうら王に実装された高度に最適化された探索ルーチンは変更せずに利用する。
実装上、DNNの推論が挟まるため、やねうら王プロセス内に実装するのは少し難しい可能性がある。Pythonで推論するプロセスに局面を投げて、作られたNNUE重みを返却するような仕組みで構わない。

## 強化学習
教師あり学習において、重み付けの値は、あくまでルート局面の評価に最適な値であって、探索木の末端局面で最適という保証がない。
そのため、強化学習をするほうがより良いと考えられる。
例(これが最適かはわからない)
様々な局面に対して、NNUE_initial_weightで1000万局面探索して指し手を取得する。これを正解とする。
提案モデルで生成されたaverated_NNUE_weightで100万局面探索して、指し手を取得する。これが正解と一致していれば報酬1、不一致なら報酬-1。
この報酬で、DNN_adapterの重みを強化学習する。

# ロードマップ

## 1. AobaZeroとnnue-pytorchの統合

AobaZeroはpytorchを用いておらず、そのままではnnue-pytorchの学習の枠組みに取り込めない。
モデル構造と学習済み重みをpytorch向けに変換する。また、局面から入力特徴量を作成するロジックも必要。

まず、AobaZeroをオリジナルのまま環境構築し、様々な局面に対して特徴量やpolicy, valueを取得できる体制を作る。入出力対応関係をファイルに保存し、テストデータとして使う。
次に、pytorchへの変換を実装する。pytorchで推論できる実装ができたら、同様に局面から特徴量/policy/valueを取得し、テストデータとの一致を検証する。

## 2. 教師あり学習

nnue-pytorchを改造し、仕組みで述べた教師あり学習を実現する。

## 3. やねうら王での対局

学習したモデルとやねうら王を組み合わせて、対局可能にする。
NNUEの重みを動的に生成できるやねうら王の改造版を作る必要がある。

## 4. 強化学習

強化学習機構を実装し、更に強くする。
