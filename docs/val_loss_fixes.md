# val_loss 上昇問題 修正内容と検証

## 修正対象

- H1: 検証集合が毎epochでずれて比較不能
- H2: train/valのログ粒度差で見え方が歪む
- H3: 早期過学習疑い（学習順序の相関が強い可能性）

## 実装変更

1. validationを毎epoch先頭から読み直すように修正
- `src/train_nnue/expert_blending_dataset.py`
- `FixedNumBatchesDataset` に `reset_on_epoch_start` を追加
- val loaderは `reset_on_epoch_start=True` に設定

2. train/valをepoch粒度で比較可能に修正
- `src/train_nnue/train_expert_blending.py`
- `train_loss_step`（stepログ）を追加
- `train_loss` / `val_loss` は `on_epoch=True` 集計で記録

3. train順序相関を緩和するバッチ順シャッフルを追加
- `src/train_nnue/expert_blending_dataset.py`
- `FixedNumBatchesDataset` に `shuffle_buffer_size` を追加
- `create_data_loaders(..., train_shuffle_buffer_size, seed)` を追加
- `train_expert_blending.py` に `--train-shuffle-buffer-size` を追加
- `scripts/run_train_expert_blending_8experts_v4_paired_noise0.sh` で `--train-shuffle-buffer-size 64` を指定

## 検証方法と結果

### 検証1: valリセットが効くか
コマンド:
```bash
cd nnue-pytorch
source .venv/bin/activate
PYTHONPATH=../src:. python - <<'PY'
from torch.utils.data import IterableDataset
from train_nnue.expert_blending_dataset import FixedNumBatchesDataset

class SeqDS(IterableDataset):
    def __iter__(self):
        i=0
        while True:
            yield i
            i+=1

val = FixedNumBatchesDataset(SeqDS(), 5, reset_on_epoch_start=True, shuffle_buffer_size=0, seed=1)
print('val epoch1', [val[i] for i in range(len(val))])
print('val epoch2', [val[i] for i in range(len(val))])
PY
```
結果:
- `val epoch1 [0, 1, 2, 3, 4]`
- `val epoch2 [0, 1, 2, 3, 4]`

判定:
- 毎epoch同じ先頭から開始できることを確認

### 検証2: trainバッファシャッフルが効くか
コマンド:
```bash
cd nnue-pytorch
source .venv/bin/activate
PYTHONPATH=../src:. python - <<'PY'
from torch.utils.data import IterableDataset
from train_nnue.expert_blending_dataset import FixedNumBatchesDataset

class SeqDS(IterableDataset):
    def __iter__(self):
        i=0
        while True:
            yield i
            i+=1

tr = FixedNumBatchesDataset(SeqDS(), 8, reset_on_epoch_start=False, shuffle_buffer_size=4, seed=1)
print('train e1', [tr[i] for i in range(len(tr))])
print('train e2', [tr[i] for i in range(len(tr))])
PY
```
結果:
- `train e1 [1, 0, 4, 2, 7, 8, 9, 10]`
- `train e2 [5, 3, 13, 6, 15, 16, 11, 18]`

判定:
- 連番順ではなく、バッファ内で順序が崩れることを確認

### 検証3: CLIに新オプションが反映されたか
コマンド:
```bash
cd nnue-pytorch
source .venv/bin/activate
PYTHONPATH=../src:. python -m train_nnue.train_expert_blending --help | rg 'train-shuffle-buffer-size|max-val-positions|paired'
```
結果:
- `--train-shuffle-buffer-size` が help に表示されることを確認

判定:
- 学習スクリプトからシャッフル強度を操作可能

## 補足

- H3（過学習）はデータ分布/検証セット規模にも依存するため、上記修正後は同条件で再学習し、
  `train_loss (epoch)` と `val_loss (epoch)` の再計測が必要。
