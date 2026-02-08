# 将棋エンジンの指し手正解率評価

将棋エンジンをデータセット上の局面で動作させ、データセットの指し手との一致率（正解率）を評価する。
学習した評価関数の強さを定量的に比較するために用いる。

## データセット

山岡の評価用データセット（floodgate棋譜由来、856,923局面）を使用。

- https://huggingface.co/datasets/takaoyamaoka/floodgate.hcpe
- バイナリ形式（HCPE: HuffmanCodedPosAndEval）

取得方法:

```
cd tmp
git clone https://huggingface.co/datasets/takaoyamaoka/floodgate.hcpe
```

### HCPEフォーマットの構造

`cshogi.HuffmanCodedPosAndEval` をdtypeとして `np.fromfile` で読み込む。各レコードのフィールド:

| フィールド | 取得方法 | 説明 |
|---|---|---|
| 局面 (SFEN) | `board.set_hcp(hcpe['hcp']); board.sfen()` | 手数は常に1 |
| 指し手 | `cshogi.move_to_usi(board.move_from_move16(hcpe['bestMove16']))` | USI形式 (例: `G*7c`) |
| 手番 | `board.turn` | 0=先手, 1=後手 |
| 勝敗 | `hcpe['gameResult']` | 0=引き分け, 1=先手勝ち, 2=後手勝ち（手番非依存） |
| 評価値 | `hcpe['eval']` | 手番側から見た値（有利なら正） |

注意: `board.move_from_move16` は `board.set_hcp` で局面をセットした後でないと正しく動作しない（指し手の合法性を局面に基づいて判定するため）。

### データの抽出

全856,923件から、シード固定の乱数でtrain/val/testに各1000件を重複なく抽出し、JSONL形式で保存する。

```
uv run python -m train_nnue.extract_hcpe_subset \
  --input tmp/floodgate.hcpe/floodgate.hcpe \
  --output-dir data/accuracy_eval
```

オプション:

| 引数 | デフォルト | 説明 |
|---|---|---|
| `--seed` | 42 | 乱数シード |
| `--count-per-split` | 1000 | 各split(train/val/test)のレコード数 |

出力例（1行）:
```json
{"sfen": "lr6l/3kg4/3ss4/p1pp5/1N2p3p/P1PP1S2P/1PN1P4/1K+B1g3L/L4P3 b BG5Prgs2np 1", "bestmove": "G*7c", "turn": 0, "gameResult": 2, "eval": -1914}
```

## 正解率の評価

### 設定ファイル

エンジンの設定をJSON形式で記述する（`configs/accuracy_eval_example.json`）。

```json
{
  "engine_path": "bin/YaneuraOu-by-gcc",
  "engine_options": {
    "Threads": 1,
    "EvalDir": "bin/suisho5"
  },
  "go_params": {
    "nodes": 1000000
  },
  "num_workers": 4
}
```

| キー | 説明 |
|---|---|
| `engine_path` | エンジンバイナリのパス |
| `engine_options` | USIオプション。`Threads`、`EvalDir`など |
| `go_params` | `Engine.go()` に渡すパラメータ。`nodes`（ノード数指定）または `byoyomi`（秒読み）|
| `num_workers` | 並列実行するEngineインスタンス数 |

パスの解決: `engine_path` と `EvalDir` はプロジェクトルートからの相対パスで記述する。相対パスの場合、`--project-root`（デフォルト: カレントディレクトリ）を基準に絶対パスに変換してからエンジンに渡される。絶対パスの場合はそのまま使用。

### 設計上のポイント

- **再現性**: `Threads=1`（シングルスレッド）かつ `nodes` 指定により、同じ局面に対して同じ結果を返す。
- **並列化**: `Engine.go()` はブロッキング呼び出しのため、`concurrent.futures.ThreadPoolExecutor` で複数のEngineインスタンスを並列実行する。各ワーカースレッドが専用のEngineプロセスを持ち、局面をラウンドロビンで分配する。
- **cshogi.usi.Engine**: コンストラクタにはエンジンバイナリのフルパスを渡す必要がある（相対パスでは動作しないケースがある）。`position()` の `sfen` 引数には `"sfen ..."` 形式（sfen接頭辞付き）で渡す。`go()` はキーワード引数（`nodes=`, `byoyomi=`）を受け取り、`(bestmove, ponder)` のタプルを返す。

### 実行

```
uv run python -m train_nnue.eval_accuracy \
  --config configs/accuracy_eval_example.json \
  --dataset data/accuracy_eval/test.jsonl \
  --output results/accuracy_eval_suisho5.json
```

stderrに進捗（100局面ごと）と最終結果が出力される。

### 出力形式

```json
{
  "accuracy": 0.602,
  "matches": 602,
  "total": 1000,
  "config": { ... },
  "dataset_path": "data/accuracy_eval/test.jsonl",
  "details": [
    {
      "index": 0,
      "sfen": "...",
      "expected": "P*3d",
      "actual": "P*3d",
      "match": true
    }
  ]
}
```

## 基準評価関数（水匠5）

動作検証用の基準として水匠5を使用する。

取得方法:
```
wget https://github.com/yaneurao/YaneuraOu/releases/download/suisho5/Suisho5.7z
p7zip -d Suisho5.7z
mv nn.bin bin/suisho5/
```

```
$ sha256sum bin/suisho5/nn.bin
768068f0d534a0603a5d38bcd143de6bbca820d5f1c95a14d40863e5b7892d76  nn.bin
```

水匠5 + nodes=1,000,000 でtest 1000局面を評価した結果: **正解率 60.2%** (602/1000)
