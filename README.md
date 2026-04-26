# コンピュータ将棋のモデル学習実験


## コマンド

tensorboardの起動

```bash
(cd nnue-pytorch; . ./.venv/bin/activate; tensorboard --logdir ../logs)
```

## Expert Blending エンジン (やねうら王 + DNN backbone) の実行手順

学習済み Expert Blending モデルを「やねうら王にロードしてそのまま対局できる
評価関数ディレクトリ」に変換し、やねうら王本体だけで動かす一連の手順。
詳しい背景は `docs/expert-blending-speed.md` を参照。

### 1. onnxruntime (CPU prebuilt) を取得する

```bash
bash YaneuraOu/extra/onnxruntime/fetch_onnxruntime.sh linux
```

- やねうら王が C++ 内で DNN backbone (`backbone.onnx`) を推論するために必要。
- `linux` 指定で Linux x86_64 用 prebuilt を `YaneuraOu/extra/onnxruntime/linux/current/`
  に展開する。Windows 用なら `win`、両方なら `both`。
- 一度実行すれば `linux/current` シンボリックリンクが残り、以後はスキップされる。
- `ONNXRUNTIME_VERSION=<ver>` でバージョン指定可能 (default: 1.19.2)。

### 2. やねうら王をビルドする

```bash
(cd YaneuraOu/source && make normal -j4)
cp YaneuraOu/source/YaneuraOu-by-gcc bin/YaneuraOu-expert-blending
```

- `make normal` で配布用ビルド (`-O3 -flto`)。`-j4` は並列ジョブ数。
- Makefile の `EXPERT_BLENDING = ON` (default) により、
  `YaneuraOu/extra/onnxruntime/linux/current/{include,lib}` を `-I/-L`、
  `-lonnxruntime` を自動でリンクし、`-DEXPERT_BLENDING_ONNXRUNTIME` を有効にする。
- 同梱した dlshogi cppshogi (`YaneuraOu/source/eval/nnue/dlshogi_cppshogi/`) も
  同時にコンパイルされ、`Position::sfen()` から DNN 入力 (`features1` /
  `features2`) を作るブリッジが生える。
- 出力された `YaneuraOu-by-gcc` を `bin/YaneuraOu-expert-blending` に置き換える
  (benchmark スクリプトは `bin/` 配下を参照する)。

### 3. checkpoint → やねうら王ロード形式に変換する

```bash
source nnue-pytorch/.venv/bin/activate
PYTHONPATH=src:dlshogi-source:nnue-pytorch:$PYTHONPATH \
python -m train_nnue.export_for_yaneuraou \
    --checkpoint logs/<expert_blending_run>/checkpoints/<step>.ckpt \
    --backbone-weights tmp/dlshogi-model/model_resnet10_swish-072 \
    --features HalfKP \
    --n-experts 8 \
    --output-dir tmp/expert_blending_release
```

- `--checkpoint`        : Expert Blending の Lightning checkpoint (`.ckpt`)。
  内部の `model.adapter.*` と `model.nnue_experts.*` を取り出す。
- `--backbone-weights`  : dlshogi の事前学習済み backbone (`.npz`)。
  `DNNBackbone` (frozen) のロードに使う。
- `--features HalfKP`   : NNUE の特徴量名。num_features (= 125388) を確定させる。
- `--n-experts 8`       : チェックポイントに含まれる expert 数。
- `--output-dir`        : 出力ディレクトリ。下記 3 ファイルが書き出される。
    - `backbone.onnx`  : `DNNBackbone + DNNAdapter` を結合した ONNX。
      入力 `(N,62,9,9) (N,57,9,9)` float32、出力 `(N,8)` softmax 済み gate。
    - `head.bin`       : 128B 固定ヘッダ + 事前量子化済み 8 expert 重み
      (FT bias int16 / FT weight int16 (F,L1) 順 / FC int32+int8 padded)。
      residual モードでは末尾に `base_*` 1 セットを追加。
    - `head.json`      : 人間可読のメタ情報 (やねうら王は読まない)。

### 4. やねうら王に評価関数ディレクトリを渡して対局/解析する

#### 4a. 対話的に動かす場合

```bash
bin/YaneuraOu-expert-blending.sh
```

- 上記は `LD_LIBRARY_PATH` を内部で設定してから本体バイナリを `exec` する
  ラッパスクリプト。将棋 GUI (ShogiGUI / 将棋所など、環境変数を渡せない GUI)
  から登録するときも、この `.sh` を「エンジン実行ファイル」として指定する。
- スクリプト自身の絶対パスから `bin/YaneuraOu-expert-blending` と
  `YaneuraOu/extra/onnxruntime/linux/current/lib` を解決するので、
  どの作業ディレクトリから呼んでも動く。
- 直接バイナリを叩きたい場合は
  `LD_LIBRARY_PATH=YaneuraOu/extra/onnxruntime/linux/current/lib bin/YaneuraOu-expert-blending`
  と等価。
- 起動後、USI で次のように設定する。

```text
setoption name EvalDir value bin/eval
setoption name ExpertBlendingDir value tmp/expert_blending_release
isready
position startpos
go nodes 1000
```

- `EvalDir`              : ベースライン NNUE (`nn.bin`) の置き場。
  `ExpertBlendingDir` を併用する場合でも、起動時の通常 NNUE ロードのために
  必要 (各 go の前に上書きされる)。
- `ExpertBlendingDir`    : 手順 3 で作ったディレクトリのパス。
  `head.bin` の magic (`EBHEAD01`) を確認してロードする。
- `isready`              : `head.bin` を mmap し `backbone.onnx` を
  onnxruntime にロード。`info string ExpertBlending: ...` でロードログが出る。
- `go nodes <N>`         : 探索開始時に局面の sfen から gate を推論し、
  8 expert を gate でブレンドした NNUE 重みに差し替えてから探索する。
  `info string blending_weight=[...]` で gate ベクトルが見える。

`EXPERT_BLENDING_VERBOSE=1` を環境変数に入れると、各 go ごとに
`info string ExpertBlending timing(ms): feat=... onnx=... blend=...` が出力され、
合成パイプラインの内訳を確認できる。

#### 4b. ベンチマーク (合成 + 探索の wall-clock 計測)

```bash
bash scripts/benchmark_expert_blending_speed.sh \
    logs/<run>/checkpoints/<step>.ckpt \
    tmp/dlshogi-model/model_resnet10_swish-072 \
    8 1000 20
```

引数: `<checkpoint> <backbone_weights> <n_experts> <nodes> <iters>`。
スクリプトは内部で

1. 一時ディレクトリへ手順 3 と同じエクスポートを実行
2. `bin/YaneuraOu-expert-blending` を起動して `usi` → `setoption` → `isready`
3. 複数局面で `go nodes <N>` を `<iters>` 回回し、bestmove までの時間を集計

を行う。`LD_LIBRARY_PATH` も内部で自動設定する。詳細結果は
`docs/expert-blending-speed.md` の iter6 を参照。
