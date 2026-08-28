# Root decision-aligned目的 結果 (2026-08-26)

> 2026-08-28追記: 事前登録した追加8,000局を完了し、合計12,000局で`+6.89 Elo`、
> 95%区間`[+0.78,+13.00]`となったため支持へ更新した。詳細は
> [24時間 深掘り検証](../deep-validation-24h-20260828/README.md)を参照。
> 支持判定後の未使用2,000局感度追試を含む14,000局でも`+6.68 Elo`、区間
> `[+1.02,+12.34]`を維持した。

## 結論

`H-DECISION-ALIGNED-LOSS`は事前規則により**測定完了**とした。root候補手utilityの勝者expert軸を
直接分類する目的は、独立validationとtestの固定bestmoveをそれぞれ`+0.13`、`+0.11`ポイントの
正方向とした。固定10万nodesの4,000局自己対局も`+7.30 Elo`だったが、95%区間
`[-3.31,+17.90]`が0を跨ぎ、支持条件の区間下端`>0`を満たさなかった。

leaf平均KLより意思決定に近い目的へ変える方向は一貫して正だったが、この教師量とhard targetでは
棋力改善を0から分離できない。仮説を棄却せず測定完了とする範囲は、既存2,000 search-utility root、
expert軸`+1`、10cp閾値、adapter-only学習である。

## 条件

- 計画: [Root decision-aligned目的](../../experiments/decision-aligned-loss-20260826.md)
- 教師: `tmp/search_utility_v1/pilot/details.jsonl`の2,000 root
- train/validation: 1,600/400 root、選抜にはvalidation先頭384 root
- initial: M0、experts/backbone固定、adapter-only、batch 64、10 epoch、seed 42
- control: leaf group loss weight 1、decision weight 0
- candidates: leaf group loss weight 1、decision CE weight `0.01, 0.05`
- accuracy: 固定validation/test各10,000局面、100万nodes、Threads=1
- match: 2,000開始局面を先後反転した4,000局、10万nodes、Threads=1

## 教師と選抜

utility最大expert軸のhard targetへ変更されたrootはtrainで453/1,600（28.31%）、validationで
105/400（26.25%）だった。10cp未満のrootはM0 gateを保持した。

| 条件 | 選抜epoch | validation group loss | decision CE |
|---|---:|---:|---:|
| task-only control | 9 | 0.01247298 | 1.86660 |
| decision weight 0.01 | 9 | 0.01248230 | 1.84502 |
| decision weight 0.05 | 9 | 0.01249223 | **1.82374** |

weight 0.05はcontrol比group loss`+0.00001925`で許容上限`+0.0001`内、decision CEは
`-0.04286`だったため最終候補に選んだ。

## 固定bestmove

| split | control | candidate | 差 | controlのみ / candidateのみ | McNemar p |
|---|---:|---:|---:|---:|---:|
| validation | 6,167/10,000 (61.67%) | 6,180/10,000 (61.80%) | +0.13pt | 856 / 869 | 0.772648 |
| test | 6,239/10,000 (62.39%) | 6,250/10,000 (62.50%) | +0.11pt | 898 / 909 | 0.814026 |

validationが正方向だったため、事前分岐に従って独立testと自己対局へ進めた。testでも方向は再現したが、
効果量は小さく、対応検定では0から分離しなかった。

## 4,000局自己対局

candidateは1,982勝1,898敗120分、score rate 51.05%、`+7.297 Elo`、標準誤差5.412 Elo、
95%区間`[-3.309,+17.904]`だった。点推定は正だが区間下端が0以下なので、事前規則の
「支持」ではなく「測定完了」とした。

## 成果物

- 教師cache: `scripts/build_decision_teacher_cache.py`
- 学習/選抜: `scripts/run_decision_aligned_stage1.sh`、`scripts/select_decision_aligned_checkpoint.py`
- accuracy: `scripts/eval_decision_aligned.sh`、`scripts/eval_decision_aligned_test.sh`
- match: `scripts/run_decision_aligned_match.sh`
- 集計: `results/decision_aligned_selection.json`、
  `results/accuracy_comparison_decision_aligned_{validation,test}.json`、
  `results/match_decision_aligned_4000.json`
- logs: `/tmp/train_nnue_decision_aligned_*.log`、`/tmp/accuracy_eval_decision_aligned_*.log`、
  `/tmp/eval_match_match_decision_aligned_chunk_*.log`
