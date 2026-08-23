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

旧評価データは全856,923件から、シード固定の乱数でtrain/val/testに各1,000件を
重複なく抽出した。次期実験では`data/accuracy_eval_10k/`を使う。

| split | 件数 | 用途 |
|---|---:|---|
| `validation.jsonl` | 10,000 | ハイパーパラメータとcheckpointの選抜 |
| `test.jsonl` | 10,000 | validationで選抜した少数候補の最終比較 |

固定データはSFEN単位で一意で、split間および旧3,000局面との重複がない。seed、入力と
出力のSHA-256は`data/accuracy_eval_10k/manifest.json`に保存している。testを温度や
正則化係数の反復調整に使わない。

再生成:

```bash
scripts/project_python.sh -m train_nnue.extract_hcpe_subset \
  --input tmp/floodgate.hcpe/floodgate.hcpe \
  --output-dir data/accuracy_eval_10k \
  --seed 20260823 \
  --split validation=10000 \
  --split test=10000 \
  --exclude-jsonl data/accuracy_eval/train.jsonl \
  --exclude-jsonl data/accuracy_eval/val.jsonl \
  --exclude-jsonl data/accuracy_eval/test.jsonl
```

オプション:

| 引数 | デフォルト | 説明 |
|---|---|---|
| `--seed` | 42 | 乱数シード |
| `--count-per-split` | 1000 | `--split`省略時のtrain/val/testの件数（旧互換） |
| `--split NAME=COUNT` | なし | 出力split名と件数。複数指定可 |
| `--exclude-jsonl PATH` | なし | SFEN単位で除外する既存JSONL。複数指定可 |

出力例（1行）:
```json
{"sfen": "lr6l/3kg4/3ss4/p1pp5/1N2p3p/P1PP1S2P/1PN1P4/1K+B1g3L/L4P3 b BG5Prgs2np 1", "bestmove": "G*7c", "turn": 0, "gameResult": 2, "eval": -1914, "source_index": 0}
```

HCPEは対局手数を保持しない。HCPから復元したSFEN末尾は常に1なので、これを
`game_ply`として使用しない。評価器は将来`game_ply`または`ply`を含むJSONLを与えた場合だけ
序盤（1--40）、中盤（41--100）、終盤（101以上）を集計する。

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

パスの解決: `engine_path`、`EvalDir`、`ExpertBlendingDir`はプロジェクトルートからの
相対パスで記述する。`scripts/eval_accuracy.sh`がプロジェクトルートを明示し、評価実装が
エンジン起動前に絶対パスへ変換する。絶対パスの場合はそのまま使用する。

### 設計上のポイント

- **再現性**: `Threads=1`（シングルスレッド）かつ `nodes` 指定により、同じ局面に対して同じ結果を返す。
- **並列化**: `Engine.go()` はブロッキング呼び出しのため、`concurrent.futures.ThreadPoolExecutor` で複数のEngineインスタンスを並列実行する。各ワーカースレッドが専用のEngineプロセスを持ち、局面をラウンドロビンで分配する。
- **cshogi.usi.Engine**: コンストラクタにはエンジンバイナリのフルパスを渡す必要がある（相対パスでは動作しないケースがある）。`position()` の `sfen` 引数には `"sfen ..."` 形式（sfen接頭辞付き）で渡す。`go()` はキーワード引数（`nodes=`, `byoyomi=`）を受け取り、`(bestmove, ponder)` のタプルを返す。

### 実行

```bash
bash scripts/eval_accuracy.sh \
  configs/accuracy_eval_example.json \
  data/accuracy_eval/test.jsonl \
  results/accuracy_eval_suisho5.json
```

評価はsandbox外で実行する。進捗とエンジン出力は
`/tmp/eval_accuracy_accuracy_eval_suisho5.log`に保存される。

評価器自体はGPUを使わない。標準NNUEは通常のやねうら王CPU評価、現行Expert Blendingは
`bin/YaneuraOu-expert-blending.sh`から起動したやねうら王プロセス内でCPU版ONNX Runtimeを
使い、`ExpertBlendingDir`の`backbone.onnx`と`head.bin`を直接ロードする。旧方式の
`DNNServerCmd`やPython推論サーバー、CUDAは使用しない。sandbox外実行が必要なのは
やねうら王を起動する評価に対する運用上の制約であり、GPU要件ではない。

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
  ],
  "wilson_95": {"lower": 0.5714, "upper": 0.6318},
  "strata": {
    "game_ply": {"available": false, "reason": "..."},
    "absolute_teacher_eval": {"available": true, "groups": {"...": {}}},
    "entering_king": {"available": true, "groups": {"...": {}}}
  }
}
```

各結果には全体のWilson 95%区間と、教師評価値絶対値（0--500、501--2000、2001以上）、
入玉の有無、利用可能なら手数帯の層別集計を保存する。

## ベースラインとの対応あり比較

ベースラインと候補を必ず同じJSONLで評価した後、局面順を検証して比較する。

```bash
bash scripts/compare_accuracy.sh \
  results/accuracy_eval_10k_halfkp_v1.json \
  results/accuracy_eval_10k_candidate.json \
  data/accuracy_eval_10k/test.jsonl \
  results/accuracy_comparison_10k_candidate.json
```

出力には両モデルの正解率とWilson区間、`both_correct`、`baseline_only`、
`candidate_only`、`both_wrong`、正解率差、両側の正確McNemar検定p値を保存する。
層別にも同じ対応あり集計を行う。`go_params`が異なる、`Threads=1`でない、またはSFEN、
教師手、indexのいずれかがずれた結果は比較を拒否する。

## 最終候補の固定ノード自己対局

開始局面JSONLはtestとは別に用意する。各開始局面を先後入れ替えて1局ずつ指し、同じ固定ノード
条件で比較する。実行はsandbox外で行うが、現行Expert Blendingを含めて評価はCPUで完結する。

```bash
bash scripts/eval_match.sh results/match_candidate.json \
  --engine1 bin/YaneuraOu-expert-blending \
  --engine1-options 'Threads=1,ExpertBlendingDir=tmp/expert_blending_release' \
  --engine2 bin/YaneuraOu-by-gcc \
  --engine2-options 'Threads=1,EvalDir=bin/eval' \
  --nodes 1000000 \
  --openings data/match_eval/openings.jsonl
```

`--games`省略時は開始局面数の2倍を実行する。明示する場合は先後ペアを保つため正の偶数とし、
開始局面数の2倍以下にする。結果JSONには勝敗、score rate、Elo推定値・標準誤差・95%区間、
各対局の開始局面と先後を保存する。Elo区間はscoreの標準誤差をEloへ変換した近似値である。

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
