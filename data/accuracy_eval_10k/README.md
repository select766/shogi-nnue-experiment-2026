# 固定正解率評価データ

次期実験で使う固定データである。

- `validation.jsonl`: 10,000局面。温度や正則化係数などの候補選択に使ってよい。
- `test.jsonl`: 10,000局面。validationで選抜した少数候補の最終比較だけに使う。
- `manifest.json`: 入力、seed、除外データ、出力件数とSHA-256を記録する。

両splitはSFEN単位で一意かつ相互に重複せず、旧`data/accuracy_eval/{train,val,test}.jsonl`
の3,000局面とも重複しない。再生成コマンドは次の通り。

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

HCPEレコードには対局手数がない。復元SFEN末尾の手数は常に1であり、`game_ply`として
扱わない。手数層別が必要なら、棋譜系列から実手数付きの別データを作る。
