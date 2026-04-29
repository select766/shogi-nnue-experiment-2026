# エキスパートライブビジュアライザー

対局中にどのエキスパートがどの割合で使われているかをブラウザでリアルタイム表示するツール。

## 前提

- Python 3.10 以上（標準ライブラリのみ使用）
- やねうら王 (Expert Blending 対応ビルド) が対局時に以下の行を stdout に出力すること

```
info string blending_weight=[0.123, 0.456, ...]
```

## 起動手順

### 1. YaneuraOu の出力をファイルに書き出す

将棋 GUI（ShogiGUI / 将棋所など）の「エンジンの出力をファイルに保存」機能、  
または tee で stdout をログファイルに書き出す。

```bash
./YaneuraOu | tee /tmp/yaneuraou.log
```

Windows では、将棋 GUI のエンジン実行ファイルとして以下を登録する。

```
bin\YaneuraOu-expert-blending.bat
```

このバッチファイルは `YaneuraOu-expert-blending.exe` の stdout を GUI に返しながら、デフォルトで `%TEMP%\yaneuraou-expert-blending.log` に追記する。
別のログパスを使う場合は、起動前に環境変数 `YANEURAOU_EXPERT_BLENDING_LOG` を設定する。

### 2. サーバーを起動する

**プロジェクトルート**（`train-nnue/`）から実行する。

```bash
python3 tools/expert_live_visualizer/server.py \
    --log /tmp/yaneuraou.log \
    --port 8765
```

`--log` を省略するとデモモード（シナリオアニメーション）で起動する。

Windows コマンドプロンプトの場合:

```bat
python tools\expert_live_visualizer\server.py --log "%TEMP%\yaneuraou-expert-blending.log" --port 8765
```

Windows PowerShell の場合:

```powershell
python .\tools\expert_live_visualizer\server.py --log "$env:TEMP\yaneuraou-expert-blending.log" --port 8765
```

ログパスを変更する場合は、バッチファイル側とサーバー側で同じ値を指定する。

コマンドプロンプト:

```bat
set "YANEURAOU_EXPERT_BLENDING_LOG=C:\path\to\yaneuraou-expert-blending.log"
python tools\expert_live_visualizer\server.py --log "%YANEURAOU_EXPERT_BLENDING_LOG%" --port 8765
```

PowerShell:

```powershell
$env:YANEURAOU_EXPERT_BLENDING_LOG = "C:\path\to\yaneuraou-expert-blending.log"
python .\tools\expert_live_visualizer\server.py --log $env:YANEURAOU_EXPERT_BLENDING_LOG --port 8765
```

### 3. ブラウザで開く

```
http://localhost:8765/
```

MacBook の右 1/3 の領域にウィンドウを配置し、残り 2/3 に将棋 GUI の盤面を表示する。

## オプション

| オプション | デフォルト | 説明 |
|---|---|---|
| `--log FILE` | なし | YaneuraOu の stdout ログファイル |
| `--port PORT` | 8765 | HTTP ポート番号 |

## 動作モード

| モード | 条件 | 表示 |
|---|---|---|
| ライブモード | サーバーが起動し `--log` のファイルに有効な行がある | ステータスが緑点滅「ライブ表示中」 |
| デモモード | サーバー未起動、または `--log` 省略 | ステータスが「デモ表示中」で局面シナリオを自動切替 |

ブラウザは 1 秒ごとに `/api/weights` をポーリングし、ライブ／デモを自動切替する。

## ファイル構成

```
tools/expert_live_visualizer/
  server.py    # Python HTTP サーバー（標準ライブラリのみ）
  index.html   # ブラウザ表示ページ
  README.md    # このファイル
```

静的データ（SVG 代表局面）は以下を参照している:

```
results/visualize_experts_8experts_lambda05_180/expert_*_top_5.svg
```
