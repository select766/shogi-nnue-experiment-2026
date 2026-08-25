# スクリプト索引

すべてリポジトリルートから実行する。Pythonを直接起動したり、仮想環境を
activateしたりしない。

## Python実行環境

| スクリプト | 用途 |
|---|---|
| `setup_python_envs.sh` | ルート環境と学習環境を構築・更新 |
| `check_environment.sh` | 依存関係、native loader、任意でCUDAを検査 |
| `check_research_hypotheses.py` | 仮説ID、優先順位、全結果文書の台帳登録を検査 |
| `project_python.sh` | データ処理・評価用Pythonの唯一の入口 |
| `nnue_python.sh` | PyTorch学習・解析用Pythonの唯一の入口 |
| `gpu_python.sh` | CUDA実演算を検査してGPU必須処理を起動 |

## 現行ワークフロー

| スクリプト | 用途 | ログ |
|---|---|---|
| `run_paired_shuffle.sh` | DNN/NNUEのpairedデータ生成 | 呼び出し側で指定 |
| `run_grouped_paired_shuffle.sh` | 1 root対K qsearch末端のgroupedデータ生成 | 呼び出し側で指定 |
| `train_expert_blending.sh` | Expert Blending学習と自動再開 | 必ず`/tmp`または指定ファイル |
| `export_expert_blending.sh` | checkpointをやねうら王用にexport | 標準出力 |
| `eval_accuracy.sh` | 最善手一致率評価 | 必ず`/tmp`または指定ファイル |
| `compare_accuracy.sh` | ベースラインとの対応あり統計・層別比較 | 標準出力 |
| `eval_match.sh` | 最終候補の固定探索自己対局 | `/tmp/eval_match_*.log` |
| `diagnose_gate.sh` | gate疎性・expert機能差・oracle routing診断 | `/tmp/diagnose_gate_*.log` |
| `run_gate_regularization_sweep.sh` | entropy＋balance正則化の3 epoch短期比較 | `/tmp/train_nnue_gate_reg_*.log` |
| `run_entmax15_sweep.sh` | 1.5-entmax＋balanceの3 epoch短期比較 | `/tmp/train_nnue_gate_entmax15_*.log` |
| `generate_router_teacher_caches.sh` | expert単独lossのrouter teacher cache生成 | `/tmp/router_teacher_cache_*.log` |
| `run_router_distillation_sweep.sh` | hard/soft router蒸留の3 epoch短期比較 | `/tmp/train_nnue_router_distill_*.log` |
| `run_router_distillation_combined_sweep.sh` | task＋soft router蒸留の勾配調整比較 | `/tmp/train_nnue_router_distill_combined_*.log` |
| `generate_optimized_gate_teacher_caches.sh` | blend loss最適化gate teacher生成 | `/tmp/optimized_gate_teacher_*.log` |
| `run_optimized_gate_distillation_sweep.sh` | 最適化gate teacher蒸留の3 epoch短期比較 | `/tmp/train_nnue_router_gate_teacher_*.log` |
| `generate_root_grouped_teachers.sh` | 複数末端で共有するroot gate teacher生成・A/B評価 | `/tmp/root_grouped_teacher_*.log` |
| `run_root_grouped_router_sweep.sh` | root-grouped task/teacher学習比較 | `/tmp/train_nnue_root_grouped_*.log` |
| `summarize_training_curve.py` | TensorBoardの学習曲線と暫定飽和判定をJSON化 | 指定したJSON |
| `benchmark_expert_blending_speed.sh` | 推論速度測定 | スクリプト内で指定 |

引数と実行例は[学習・評価手順](../docs/operations/training-and-evaluation.md)を参照する。
過去の実験専用スクリプトは`archive/`にあり、現行処理からは呼び出さない。

学習とGPU評価は`gpu_python.sh`を通す。CUDAが使えない場合はCPUへfallbackせず、
sandbox外での再実行を求めて停止する。
