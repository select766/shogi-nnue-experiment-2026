# 評価設定索引

`scripts/eval_accuracy.sh`へ渡す現行設定だけをこのディレクトリ直下に置く。
相対パスはすべてリポジトリルート基準で解決される。

| 設定 | 用途 |
|---|---|
| `accuracy_eval_halfkp_v1.json` | ベースラインHalfKP (`bin/eval/nn.bin`) |
| `accuracy_eval_expert_blending_current.json` | 現行Expert Blending release |
| `accuracy_eval_proxy_gap_control510.json` | Proxy gap棋力検証のcheckpoint 510 control |
| `accuracy_eval_proxy_gap_actual500k.json` | 実探索leaf 500k候補 |
| `accuracy_eval_example.json` | 新しい設定を作るための例 |

実行方法とJSONの仕様は[精度評価](../docs/operations/accuracy-evaluation.md)を参照する。
過去モデル専用の設定は`archive/`にあり、現行評価には使わない。
