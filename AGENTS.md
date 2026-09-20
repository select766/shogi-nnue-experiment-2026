# AGENTS.md - train-nnue

## 最初に読む

1. `docs/README.md`
2. `docs/operations/python-environments.md`
3. `docs/operations/training-and-evaluation.md`
4. 研究作業なら`docs/research/update-report-202608.md`と`docs/research/hypotheses.md`

`docs/archive/`と`scripts/archive/`は参照専用であり、掲載コマンドを実行しない。

## プロジェクト

将棋NNUE (HalfKP 256x2-32-32) と Expert Blending のデータ作成、PyTorch Lightning学習、
モデル変換、やねうら王評価を管理する。

## 実行環境

Pythonを手動でactivateしない。`python`、`uv run`、`PYTHONPATH=...`を直接組み立てない。

- データ処理・JSON・エンジン評価: `scripts/project_python.sh`
- GPU必須の学習・loss評価: `scripts/gpu_python.sh`
- CPUでよいPyTorch処理・モデル変換: `scripts/nnue_python.sh`
- 初回構築: `bash scripts/setup_python_envs.sh`
- 環境確認: `bash scripts/check_environment.sh`
- GPU確認: `bash scripts/check_environment.sh --require-gpu`

ラッパーは作業ディレクトリをリポジトリルートに固定する。相対パスはすべてリポジトリルート基準。

## 標準出力とGPU

- 学習出力をエージェントが直接受け取るとログが数十GBになり得る。必ず`/tmp/*.log`へ
  `> LOG 2>&1`で保存する。`scripts/train_expert_blending.sh`と`eval_accuracy.sh`は自動で保存する。
- 学習、`check_loss_per_gameply`、`check_loss_per_expert`、`eval_accuracy`、やねうら王を起動する
  評価はsandbox外で実行する。
- GPU必須処理を`scripts/nnue_python.sh`から直接起動しない。`scripts/gpu_python.sh`の
  CUDA実演算preflightを必ず通し、CPU fallbackを許可しない。
- sandbox内ではCUDAが見えずCPU fallbackや`nan`が起きる。データ破損を疑う前に実行環境を確認する。

## 研究ループの運用

- 5分を超える計算はCodexセッション内で起動・監視せず、研究ジョブキューへ渡す。
  セッション内の短い検査には`timeout 300`を使う。
- 管理入口は`bash scripts/research_loop.sh`。詳細は`docs/operations/research-loop.md`。
  ユーザーがsandbox外で`run`を起動し、prepareとreviewは毎回新規Codexセッションで実施する。
- 外部セッションは`status`、`pause`、`pause --now`、`enqueue`で確認・制御できる。
  `pause`後はworker終了を確認してから実験コードを編集する。state.jsonを直接編集しない。
- 仮説の優先順位は従来どおり`docs/research/hypotheses.md`を唯一の正本とする。
  実行状態は`.research-loop/`、引継ぎは課題ごとのattemptフォルダに保存する。
- 中断後の計算を自動再実行しない。成果物を確認して`retry --phase`を明示する。

## 現行データ

- 主学習: `dataset/split_v1_paired_uniform_50/train/{dnn.bin,nnue.bin}`
- validation: `dataset/split_v1_paired_uniform_50/val1/{dnn.bin,nnue.bin}`
- 再生成元: `dataset/tanuki-.nnue-pytorch-2024-07-30.1/`
- 初期NNUE: `logs/halfkp_v1/checkpoints/83000.ckpt`
- DNN backbone: `tmp/dlshogi-model/model_resnet10_swish-072`

旧`split_v1`、旧paired、`uniform_50_old`は削除済み。archiveの旧パスを使わない。

## 現行コマンド

```bash
# Expert Blending学習。完全な引数例はoperations文書を参照
bash scripts/train_expert_blending.sh --help

# checkpointをC++やねうら王形式へ変換
bash scripts/export_expert_blending.sh CHECKPOINT OUTPUT_DIR 8

# 最善手一致率
bash scripts/eval_accuracy.sh CONFIG DATASET OUTPUT

# qsearch pairedデータ生成
bash scripts/run_paired_shuffle.sh INPUT_DIR OUTPUT_DIR 8 0 50 \
  > /tmp/paired_shuffle.log 2>&1
```

## サブモジュール

| パス | 用途 |
|---|---|
| `nnue-pytorch/` | 学習フレームワークとC++データローダー |
| `YaneuraOu/` | Expert Blending対応やねうら王 |
| `tanuki-learner/` | qsearch shuffle |
| `dlshogi-source/` | DNN特徴量とbackbone実装 |

サブモジュールを編集した場合は、サブモジュール内で先にコミットし、その後親リポジトリで
参照更新をコミットする。ユーザーの既存変更は戻さない。

## 主要実装

- `src/train_nnue/train_expert_blending.py`: 学習entry point
- `src/train_nnue/expert_blending_model.py`: gateとNNUE experts
- `src/train_nnue/expert_blending_dataset.py`: paired loader
- `src/train_nnue/export_for_yaneuraou.py`: ONNX + head.bin export
- `src/train_nnue/eval_accuracy.py`: 最善手一致率
- `src/train_nnue/check_loss_per_gameply.py`: validation loss診断

## 研究仮説の更新

`docs/research/hypotheses.md`を仮説の状態と優先順位の唯一の台帳とする。実験終了時は、結果文書の追加、
対象仮説の判定、未完了仮説の優先順位振り直し、結果から生じた新仮説の追加を同じ変更に含める。
最後に`scripts/project_python.sh scripts/check_research_hypotheses.py`を実行する。結果文書が台帳に未登録、
仮説IDが不正、または優先順位が連番でない状態では、研究作業を完了扱いにしない。検査通過後、
実験の実装、結果文書、仮説レジストリをコミットするまでを実験完了条件とする。サブモジュールを
変更した実験では、サブモジュール内のコミットを先に作り、その参照更新を親リポジトリの実験コミットに含める。

## shuffle_kifu

- 入力ディレクトリ内の全ファイルをバイナリとして読む。`.bin`以外を置かない。
- 実行バイナリは`bin/shuffle/tanuki-learner`、qsearch用モデルは`bin/shuffle/eval/nn.bin`。
- ビルドは`make -C tanuki-learner/source evallearn BLAS=NONE -j8`。
