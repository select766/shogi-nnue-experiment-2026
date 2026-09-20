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

- 研究ループが起動するprepare/reviewのCodexセッションでは、5分を超える計算を
  起動・監視せず外部計算phaseまたは次ジョブへ渡す。短い検査には`timeout 300`を使う。
  この時間制限はループ外でユーザーが直接依頼する整備・デバッグ・調査セッションには適用しない。
  ループ外では依頼内容と作業規模に応じて実行時間を判断する。安全・権限・GPU等の規則は共通。
- 管理入口は`bash scripts/research_loop.sh`。詳細は`docs/operations/research-loop.md`。
  ユーザーがsandbox外で`run`を起動し、prepareとreviewは毎回新規Codexセッションで実施する。
- 外部セッションは`status`、`pause`、`pause --now`、`enqueue`で確認・制御できる。
  `pause`後はworker終了を確認してから実験コードを編集する。state.jsonを直接編集しない。
- 仮説の優先順位は従来どおり`docs/research/hypotheses.md`を唯一の正本とする。
  実行状態は`.research-loop/`、引継ぎは課題ごとのattemptフォルダに保存する。
- 中断後の計算を自動再実行しない。成果物を確認して`retry --phase`を明示する。
- 通常runは空キューでも無期限待機する。`docs/operations/research-autonomy.md`を必ず読み、
  通常不足は代替策へ再計画（同一仮説累計2回まで）か手法打切りとし、人間待ちにしない。
  `needs_human`は契約・支払・破壊的変更・既存棋譜の来歴質問だけ。独立課題と日次は継続する。
  認証・セッション故障・検査/コミット失敗は実行エラーとしてblockedで停止する。
  `configs/research_data.json`の整備済みデータを利用し、不可能な完全分離証明を再要求しない。
- 約1日1回、最良候補の固定正解率と対基準勝率を日次ジョブで測る。
  詳細は`docs/operations/daily-benchmark.md`。日次計算中もCodexを起動しない。
- 最良候補は`configs/research_champion.json`。独立した選抜根拠を記録して更新・コミットする。
  最新checkpointやlossだけで交代させず、日次監視データを候補選抜・係数調整に使わない。
- 日次の固定データ・対戦相手・条件はseries内で変更しない。変更時は新seriesにする。
  機械的な日次観測は個別仮説の判定と別に蓄積し、仮説台帳の毎日更新は不要。
- 日次評価を遅らせないよう、新規の長い研究計算はできるだけ6時間以内のchunkへ分割する。

## 計算の並列化

- 共通ツール・実験固有スクリプトとも、結果を変えず独立実行できる作業は必ず並列化する。
  prepareで並列化単位、worker数、CPU/メモリ上限、同等性検証を記録する。
  逐次実行を選ぶ場合は依存関係・資源制約・測定条件など具体的理由を記録する。
- 固定nodes対局はThreads=1を維持し、先後ペアを独立workerに配分する。標準4 workers。
  エンジン・ログ・乱数・保存先を分離し、結果は固定ID順で集計する。失敗/中断時は
  workerと子エンジンを終了し、監査済み完全ペアだけを明示再開で再利用する。
- 同等性は棋譜、裁定、得点、実nodes等で検証する。処理順依存や乱数seed、浮動小数点の
  集計順を変えない。並列数だけから同等性や線形speedupを仮定しない。
- 時間制限対局、速度/費用測定、共有状態を持つ処理、GPU競合は資源競合で測定値が変わるため、
  同等性を確認せず並列化しない。逐次対照は検証のために維持してよい。
- 研究loopのprepare/execute/reviewは引き続き直列とし、計算phase内で並列化する。
  実行中の固定manifest・旧成果物を書き換えず、変更は版と移行記録を残して次attemptへ適用する。
  詳細は`docs/operations/match-parallelism.md`。

## 現行データ

- 主学習: `dataset/split_v1_paired_uniform_50/train/{dnn.bin,nnue.bin}`
- validation: `dataset/split_v1_paired_uniform_50/val1/{dnn.bin,nnue.bin}`
- 再生成元: `dataset/tanuki-.nnue-pytorch-2024-07-30.1/`
- 初期NNUE: `logs/halfkp_v1/checkpoints/83000.ckpt`
- DNN backbone: `tmp/dlshogi-model/model_resnet10_swish-072`
- 実手数・元棋譜付き評価: `dataset/floodgate2025/curated-v1/`（原棋譜の双方rating>=3500）。
  固定追試入力は`dataset/floodgate2025/match-plan-v1/`。詳細は`docs/operations/floodgate2025.md`。

テスト用棋譜は取得済みデータを優先する。来歴不明点は調査後にまとめてユーザーへ質問し、
回答を永続記録する。人間による継続的なデータ補充を前提にせず、不足時は既存エンジンの
自動対局で生成し来歴を記録してよい。外部棋譜の追加取得はユーザーが明示的に依頼した場合に限る。
pretraining祖先までの完全分離証明を一律の開始条件にせず、確認済み範囲と限界を報告する。

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
