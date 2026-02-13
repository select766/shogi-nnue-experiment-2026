# Expert Blending: 局面適応型NNUE評価関数

## 目的

将棋AIにおいて、**局面に応じた評価関数を動的に生成する**ことで、同じ探索コスト（同じ探索局面数）でより良い手を選べることを示す。

従来のNNUE評価関数は全局面に対して同一の重みで評価を行う。しかし、序盤・中盤・終盤、あるいは攻め合い・受け・入玉など、局面の性質によって「良い評価」のあり方は異なる。そこで、DNN (Deep Neural Network) を用いて局面の性質を認識し、その局面に最適なNNUE重みを動的に合成する。

## モデル構造

本モデルは3つのコンポーネントから構成される。

### 1. DNN Backbone (特徴量抽出器、重み固定)

AlphaZero方式のResNetで、局面テンソルから高次の特徴量を抽出する。

- 入力: `state_tensor` (局面の盤面情報を表すテンソル、shape: batch × C × 9 × 9)
- 出力: `feat` (特徴マップ、shape: batch × 256 × 9 × 9)
- 学習済みモデルとしてAobaZero (http://www.yss-aya.com/aobazero/) の重みを使用
- **学習時は重みを固定** (勾配を流さない)

```
feat = DNN_backbone(state_tensor)  # 重み固定、推論のみ
```

### 2. DNN Adapter (ゲーティングネットワーク、学習対象)

backboneの出力からN_EXPERTS個のexpert混合重みを計算する小さなネットワーク。

- 入力: `feat` (backboneの出力)
- 出力: `weight` (shape: batch × N_EXPERTS、softmaxにより総和1)
- 構造: Global Average Pooling → 全結合層2層程度 → N_EXPERTS次元
- **学習対象**

```
logits = DNN_adapter(feat)         # shape (batch, N_EXPERTS)
weight = softmax(logits + noise)   # 学習時のみnoiseを加える
```

noiseはガウスノイズで、学習中に特定のexpertにのみ重みが集中する退化を防ぐ正則化の役割を持つ。

### 3. NNUE Experts (N個の評価関数、学習対象)

N_EXPERTS個のNNUE重みセット。それぞれ独立した評価関数のパラメータを持つ。

- NNUE関数自体はパラメータを持たない純粋な関数 (HalfKP 256×2-32-32 の forward 計算)
- 各expertの重み `NNUE_weight[i]` が学習対象
- 全expertの初期値は同一の学習済みモデル `logs/halfkp_v1/checkpoints/83000.ckpt` から複製
- **学習対象**

### 推論の流れ

```python
# 入力: 1つの局面に対する state_tensor (DNN用) と state_kp (NNUE用HalfKP特徴量)

# Step 1: backboneで局面の高次特徴を抽出
feat = DNN_backbone(state_tensor)           # 重み固定

# Step 2: adapterで局面に適したexpert混合重みを計算
weight = softmax(DNN_adapter(feat) + noise) # shape (N_EXPERTS,), 総和=1

# Step 3: N個のNNUE重みを混合重みで加重平均
averaged_NNUE_weight = sum(weight[i] * NNUE_weight[i] for i in range(N_EXPERTS))

# Step 4: 合成した重みで局面を評価
value = NNUE(state_kp, averaged_NNUE_weight)
```

ポイント: Step 3で重みパラメータ空間上の加重平均を取っている。これは各expertの出力valueの加重平均 (アンサンブル) とは異なる。パラメータ空間上の補間により、N_EXPERTS個の離散的なexpertでは表現できない中間的な評価関数も生成できる。

### 学習

- 学習データ: nnue-pytorchの既存教師データ (packed SFEN形式、局面→評価値の対)
- 損失関数: 既存のNNUE学習と同一 (教師スコアとの交差エントロピー + 対局結果との交差エントロピーのλブレンド)
- 学習対象: DNN_adapter の重み + N_EXPERTS個の NNUE_weight
- 固定: DNN_backbone の重み

## 対局時の動作

対局時はルート局面 (現在の局面) に対してのみDNN推論を行い、averaged_NNUE_weightを1回だけ計算する。

```
1. ルート局面を受け取る
2. DNN_backbone + DNN_adapter でNNUE重みを合成 (Python側、1回だけ)
3. 合成した重みをやねうら王に渡す
4. やねうら王は通常のNNUE評価関数として、この固定重みで探索を実行
```

この設計の利点:
- やねうら王の高度に最適化された探索ルーチン (αβ探索、枝刈り等) をそのまま利用できる
- DNN推論はルート局面で1回だけなので、探索速度への影響は無視できる
- 実装上はPythonプロセスとやねうら王プロセスの間で重みデータをやり取りする構成で十分

探索中の末端局面ではルート局面で生成した重みをそのまま使う。つまり「この局面の性質に合った評価基準」を探索全体に適用する。

## 強化学習 (将来)

教師あり学習で得られるexpert混合重みは、あくまでルート局面の評価に最適化されたものであり、探索木の末端局面での評価に最適とは限らない。対局時にはルート局面で生成した重みで探索全体を評価するため、探索との相互作用を考慮した重み生成が望ましい。

強化学習の一案:
1. 様々な局面に対し、ベースライン (単一NNUE) で十分な探索 (例: 1000万局面) を行い、指し手を「正解」とする
2. 提案モデルで生成した重みを使い、少ない探索 (例: 100万局面) で指し手を取得する
3. 正解と一致すれば報酬+1、不一致なら報酬-1として、DNN_adapterを強化学習する

これにより「探索と組み合わせたときに最善手を見つけやすい評価関数」を局面ごとに生成できるようになることを期待する。

## ロードマップ

### 1. AobaZeroとnnue-pytorchの統合
AobaZeroのモデル構造と学習済み重みをPyTorchに変換し、backbone として利用可能にする。局面からDNN入力特徴量を生成するロジックも実装する。オリジナルとの出力一致をテストデータで検証する。

### 2. 教師あり学習
nnue-pytorchを改造し、Expert Blendingモデルの教師あり学習を実装する。

### 3. やねうら王での対局
合成したNNUE重みを動的にロードできるやねうら王の改造版を作り、対局可能にする。

### 4. 強化学習
探索との相互作用を考慮した強化学習を実装し、さらに強化する。
