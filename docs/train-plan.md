# NNUEモデル学習プラン

大容量のデータセットと適切なハイパーパラメータで学習を試みる。

# データセット
オリジナルのデータを https://huggingface.co/datasets/nodchip/tanuki-.nnue-pytorch-2024-07-30.1 より取得して以下に配置した。
./dataset/tanuki-.nnue-pytorch-2024-07-30.1/
このディレクトリは読み取り専用とする。

これを分割・シャッフルしてから学習する必要がある。
今後様々なキャリブレーションを想定しているので、
train 90%, 残り2%ずつをval1, val2, val3, val4, testに分割。

データは約300GBあり、 ./dataset 以下で操作する必要がある。
加工した大容量データは、 ./dataset 以下にサブディレクトリを作成して整理して配置すること。

# ハイパーパラメータ

以下のコマンドに掲載されているハイパーパラメータで良い。パスは書き換えること。

```bash
python train.py --features "HalfKP" --batch-size 16384 --max_epochs 1000000 --enable_progress_bar False --default_root_dir logs/20240526_halfkp_256x2-32-32 --threads 8 --lr 0.5 0.05 --num-workers 8 --lambda 1.0 0.5 --label-smoothing-eps 0.001 --accelerator gpu --devices 1 --score-scaling 361 --min-newbob-scale 1e-5 --epoch-size 1000000 --num-epochs-to-adjust-lr 500 --momentum 0.9 --network-save-period 1000 --resume-from-model "" "path\to\shogi_hao_depth9\train_shuffled\shuffled.bin" "path\to\shogi_hao_depth9\val_shuffled\shuffled.bin"
```

# 再開について

学習設定ごとにシェルスクリプトを作成し、途中でctrl-cで止めたあと、同じスクリプトを実行すれば再開できるようにしたい。
`--resume-from-model`は、そのまま使えない可能性がある。読み込んだモデルファイルを初期値として使うが、learning rateの変化などが保存されないかも。再開に必要な実装をする。また、1時間に1回程度は保存されるようなハイパーパラメータにしておく。
