# やねうら王ビルド (Linux)

## 標準NNUEエンジン

```bash
make -C YaneuraOu/source normal -j8
cp YaneuraOu/source/YaneuraOu-by-gcc bin/YaneuraOu-by-gcc
```

## Expert Blendingエンジン

現在の方式はPython推論サーバーを使わない。やねうら王が`backbone.onnx`と`head.bin`を
直接ロードする。

```bash
bash YaneuraOu/extra/onnxruntime/fetch_onnxruntime.sh linux
make -C YaneuraOu/source normal -j8
cp YaneuraOu/source/YaneuraOu-by-gcc bin/YaneuraOu-expert-blending
```

`YaneuraOu/source/Makefile`の`EXPERT_BLENDING = ON`によりonnxruntimeがリンクされる。
`bin/YaneuraOu-expert-blending.sh`は`LD_LIBRARY_PATH`を設定するGUI向け起動ラッパー。

## 評価関数

標準NNUEは`bin/eval/nn.bin`へ置く。Expert Blending checkpointは次のように変換する。

```bash
mkdir -p bin/eval
cp /path/to/model.nnue bin/eval/nn.bin

bash scripts/export_expert_blending.sh \
  logs/expert_blending_trial/checkpoints/100.ckpt \
  tmp/expert_blending_release \
  8
```

## 動作確認

標準NNUE:

```bash
scripts/project_python.sh -m train_nnue.run_yaneuraou
```

Expert Blending:

```bash
bin/YaneuraOu-expert-blending.sh
```

起動後、応答を待ちながら入力する。

```text
usi
setoption name EvalDir value bin/eval
setoption name ExpertBlendingDir value tmp/expert_blending_release
isready
position startpos
go nodes 1000
quit
```

`isready`の`readyok`前に`position`や`go`を送らない。自動評価はsandbox外で実行する。

## 速度計測

```bash
bash scripts/benchmark_expert_blending_speed.sh \
  logs/expert_blending_trial/checkpoints/100.ckpt \
  tmp/dlshogi-model/model_resnet10_swish-072 \
  8 1000 20 \
  > /tmp/benchmark_expert_blending.log 2>&1
```

過去の計測結果は`docs/research/results/expert-blending-speed.md`にある。
