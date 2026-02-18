# val_loss 上昇 仮説検証レポート

## 実行コマンド

```bash
PYTHONPATH=src python -m train_nnue.validate_val_loss_hypotheses --logdir logs/expert_blending_8experts_v4_paired_noise0 --train-bin dataset/split_v1_paired/train.bin --val-bin dataset/split_v1_paired/val1.bin --paired --paired-cache-dir tmp/paired_nnue_cache --batch-size 256 --max-val-positions 100000 --sample-n 64 --seed 42 --output docs/val_loss_hypotheses_report.md
```

- TensorBoard event: `logs/expert_blending_8experts_v4_paired_noise0/lightning_logs/version_0/events.out.tfevents.1771371960.cufantubuntu.505509.0`

## 結果

### H1: 検証集合が毎epochでずれており val_loss が比較不能
- 検証方法: `create_data_loaders()` の式と同じ計算で val 1epoch あたりの消費局面数を算出し、循環読み出し時の開始オフセットをシミュレーション。
- 結果: val_records=10000, batch_size=256, num_val_batches=40, consumed/epoch=10240, shift/epoch=240, cycle=125。 先頭オフセット遷移(6epoch): [0, 240, 480, 720, 960, 1200]
- 判定: `supported`

### H2: train/val のログ粒度差で見え方が歪む
- 検証方法: TensorBoardイベントを直接読んで `train_loss` と `val_loss` の記録頻度を比較。
- 結果: train_loss points=1675, val_loss points=21, mean step delta(train)=50.0, mean step delta(val)=3907.0。
- 判定: `supported`

### H3: 実際に過学習傾向が出ている
- 検証方法: val境界(step)ごとに train_loss を平均化し、epoch方向の一次傾き(train/val/gap)を計測。
- 結果: slope(train)=-0.000166, slope(val)=+0.000396, slope(gap)=+0.000562; first(val-train)=-0.001474, last(val-train)=+0.008264
- 判定: `supported`

### H4: pairedデータ境界で DNN/NNUE がずれている
- 検証方法: paired後半40Bと cache の同位置40Bをランダムサンプル比較。
- 結果: sample_n=64, mismatches(second_vs_cache)=0, same(first_vs_second)=0
- 判定: `not_supported`

## 判定キー

- `supported`: 仮説を支持する結果
- `not_supported`: 仮説を支持しない結果
- `partially_supported`: 一部のみ支持
- `inconclusive`: データ不足で判断不能
