# Windows 版 YaneuraOu-expert-blending ビルド手順

Expert Blending 対応やねうら王 (`bin/YaneuraOu-expert-blending.exe`) を
Windows x64 向けにクロスビルドまたはネイティブビルドする手順。

## 前提: ビルド環境

### 方法 A: MSYS2 MinGW64 (推奨・Windows ネイティブ)

Windows 機上で [MSYS2](https://www.msys2.org/) をインストールし、
**MSYS2 MinGW64** シェルを使う。

```bash
pacman -S --needed \
    mingw-w64-x86_64-clang \
    mingw-w64-x86_64-lld \
    mingw-w64-x86_64-cmake \
    make \
    unzip \
    curl
```

Makefile は `$(MSYSTEM) == MINGW64` を検知して自動的に Windows 向け設定
(`TARGET = YaneuraOu-by-gcc.exe`、`-static -Wl,--stack,25000000`) を適用する。

### 方法 B: Ubuntu から MinGW クロスコンパイル (未検証)

`x86_64-w64-mingw32-clang++` を使う経路は Makefile に条件分岐があるが、
動作確認はされていない。実績のある方法 A を優先する。

## 1. onnxruntime Windows prebuilt を取得

```bash
cd YaneuraOu/extra/onnxruntime
bash fetch_onnxruntime.sh win
```

展開後レイアウト:

```
extra/onnxruntime/win/current/
    include/   ← onnxruntime_cxx_api.h など
    lib/
        onnxruntime.lib   ← import library (リンク時に使う)
        onnxruntime.dll   ← 実行時に必要 (exe と同じフォルダに置く)
```

## 2. ビルド

MSYS2 MinGW64 シェルから:

```bash
cd /path/to/train-nnue/YaneuraOu/source
make clean
make normal -j4
```

成果物: `source/YaneuraOu-by-gcc.exe`

リポジトリルートにコピー:

```bash
cp YaneuraOu/source/YaneuraOu-by-gcc.exe bin/YaneuraOu-expert-blending.exe
```

## 3. 配布物の構成

以下を同じフォルダに置けばどこでも動く:

```
YaneuraOu-expert-blending.exe
onnxruntime.dll                     ← extra/onnxruntime/win/current/lib/ から
eval_expert_blending/
    backbone.onnx
    head.bin
    head.json
```

`onnxruntime.dll` のバージョンは `fetch_onnxruntime.sh` で取得した prebuilt
(デフォルト 1.19.2) と一致させること。

## 4. 起動確認

```
YaneuraOu-expert-blending.exe
usi
setoption name EvalDir value eval_expert_blending
isready
```

`readyok` の前に:

```
info string ExpertBlending: init dir=eval_expert_blending
info string ExpertBlending: head.bin loaded: F=125388 L1=256 mode=weighted ...
info string ExpertBlending: backbone.onnx loaded
```

が表示されれば正常。

## ビルド上の注意点

### Makefile の `-static` と onnxruntime.dll

Windows 向けビルドでは自動的に `-static` が付き、MinGW の CRT と
libstdc++ を静的リンクする。ただし **onnxruntime は DLL のまま**
(import library `.lib` 経由で動的リンク)。
実行環境に `onnxruntime.dll` を置くのを忘れないこと。

### `-lpthread` の扱い

`LDFLAGS += -lpthread` は MSYS2 MinGW64 では自動的に
`-lwinpthread` に解決されるため変更不要。

### `ORTCHAR_T` (onnxruntime Session のパス型)

Windows 版の ORT C++ API では `Ort::Session` コンストラクタの
モデルパスが `const wchar_t*` (`ORTCHAR_T`) を要求する。
`expert_blending_loader.cpp` 内で `#if defined(_WIN32)` ガードにより
`std::string` → `std::wstring` に変換してから渡す実装になっている。
**eval ディレクトリのパスは ASCII のみ** を前提としている (日本語パス不可)。

### `-fno-exceptions` / `-fno-rtti`

Makefile のデフォルトフラグ。ORT C++ API には
`#define ORT_NO_EXCEPTIONS 1` (Makefile で `-DNO_EXCEPTIONS` と同等) を
適用済みのため、例外を使うコードは書けない。

### cppshogi (dlshogi vendoring) の名前空間

`eval/nnue/dlshogi_cppshogi/cppshogi/` にある cppshogi はグローバル名前空間に
`Position`, `Bitboard` 等を定義する。
やねうら王本体の `YaneuraOu::Position` との衝突を避けるため、
cppshogi ヘッダは `dlshogi_features.cpp` 内でのみ include する設計になっている。

## モデルファイルの準備

モデルファイルはリポジトリ外 (`.gitignore`) に置かれる。
学習環境 (Linux) で変換した `eval_expert_blending/` (backbone.onnx + head.bin + head.json)
をアーカイブして Windows 機に転送する。

変換ツール (Linux 側):

```bash
uv run python -m train_nnue.export_for_yaneuraou \
    --checkpoint logs/<run>/checkpoints/<epoch>.ckpt \
    --backbone-weights tmp/dlshogi-model/model_resnet10_swish-072 \
    --output-dir bin/eval_expert_blending
```
