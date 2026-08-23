# Python環境

## 原則

このリポジトリには用途の異なる2環境がある。shellで`activate`せず、必ずラッパーを使う。

| 用途 | 定義 | 実行入口 |
|---|---|---|
| データ抽出、JSON処理、エンジン評価 | `pyproject.toml`, `uv.lock`, `.venv/` | `scripts/project_python.sh` |
| PyTorch学習、loss評価 | `requirements/nnue.txt`, `nnue-pytorch/.venv/` | `scripts/gpu_python.sh` |
| モデル変換、CPUでよいPyTorch処理 | 同上 | `scripts/nnue_python.sh` |

両ラッパーはリポジトリルートへ移動してからPythonを起動する。相対パスは常に
`/home/select766/shogi/train-nnue`基準で記述する。`PYTHONPATH`を手動設定しない。

## 構築

Ubuntu 24.04、Python 3.11、CUDA対応GPU、`uv`を前提とする。

```bash
git submodule update --init --recursive
bash scripts/setup_python_envs.sh
```

セットアップスクリプトはルート環境を`uv.lock`から開発用Jupyter依存を除いて同期し、学習環境へ
PyTorch 2.5.1 CUDA 12.1版と`requirements/nnue.txt`を導入する。グローバルPythonへは導入しない。

## C++データローダー

初回または`nnue-pytorch/training_data_loader.cpp`変更後にビルドする。

```bash
cmake -S nnue-pytorch -B nnue-pytorch/build \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  -DCMAKE_INSTALL_PREFIX="$PWD/nnue-pytorch"
cmake --build nnue-pytorch/build --config RelWithDebInfo --target install -j8
```

生成物は`nnue-pytorch/libtraining_data_loader.so`。ローダーは自身のファイル位置を基準に
共有ライブラリを探すため、コマンドの作業ディレクトリには依存しない。

## 確認

```bash
# importと共有ライブラリ。sandbox内でも実行可
bash scripts/check_environment.sh

# CUDA確認。学習と同様にsandbox外で実行する
bash scripts/check_environment.sh --require-gpu
```

Codexは学習、loss評価、最善手一致率評価、やねうら王評価をsandbox外で実行する。
CUDAが見えないsandbox内の結果でデータ破損やモデル不良を判断しない。
GPU必須処理は`gpu_python.sh`がCUDA tensor演算を事前実行する。CUDAが見えなければ
CPUへfallbackせず、sandbox外での再実行を示して停止する。
