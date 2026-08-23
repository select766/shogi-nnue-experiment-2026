# train-nnue

将棋NNUEと Expert Blending のデータ作成、学習、変換、評価を管理するリポジトリ。

作業を始めるときは、最初に [docs/README.md](docs/README.md) を読むこと。古い実験資料は
`docs/archive/`、古い実行スクリプトは `scripts/archive/` に分離してあり、現行手順には使わない。

## 実行入口

```bash
# 初回または依存更新後
bash scripts/setup_python_envs.sh

# 環境確認。GPU確認はsandbox外で実行する
bash scripts/check_environment.sh
bash scripts/check_environment.sh --require-gpu

# Expert Blending学習。全出力はラッパーが/tmpへ保存する
bash scripts/train_expert_blending.sh --help

# 最善手一致率評価
bash scripts/eval_accuracy.sh \
  configs/accuracy_eval_halfkp_v1.json \
  data/accuracy_eval/test.jsonl \
  results/accuracy_eval_example.json
```

Python仮想環境を手動でactivateしない。データ処理は`scripts/project_python.sh`、PyTorch学習・
モデル変換は`scripts/nnue_python.sh`を使う。どちらも作業ディレクトリをリポジトリルートへ
固定するため、コマンド内の相対パスはすべてリポジトリルート基準になる。
