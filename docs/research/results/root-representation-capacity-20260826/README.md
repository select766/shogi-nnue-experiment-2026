# Root-only adapter容量 結果 (2026-08-26)

## 結論

`H-ROOT-REPRESENTATION`のうち、配備条件と入力を固定した単純なadapter幅拡大は**棄却**した。
hidden 128のmatched controlに対し、256は選抜用valBで悪化した。512はvalBで僅かに改善したものの、
未使用valAでは平均group lossが悪化し、事前のbootstrap条件を満たさなかった。したがって速度測定と
固定10,000局面評価へ進める候補はない。

これは「rootでDNNを一度だけ実行し、探索中はblendを固定する」設計を否定しない。また、現DNN出力に
含まれないroot情報を追加する仮説も未検証である。後者は`H-ROOT-FEATURES`として分離した。

## 条件

- 計画: [Root-only adapter容量](../../experiments/root-representation-capacity-20260826.md)
- 初期値: M0のhidden 128 unitを2倍/4倍に複製し、出力重みを分割して初期logitを保存
- train: 実探索leaf 99,840 root、各root 8 leaf
- validation: valB/valA各19,968 root
- experts/backbone固定、mean group loss、adapter-only、1 epoch
- matched control: M0から同条件で追加学習したhidden 128モデル

初期logit保存は回帰テストで確認した。DNN backboneは各rootで一度だけ実行し、探索中の固定blendという
配備制約は全条件で同じである。

## 結果

差は候補からhidden 128対照を引いた値であり、負が改善を表す。

| validation | hidden | 平均group loss差 | bootstrap 95%区間 | 判定 |
|---|---:|---:|---:|---|
| valB | 256 | +0.00000780 | [+0.00000528, +0.00001046] | 選抜で脱落 |
| valB | 512 | -0.00000091 | [-0.00000441, +0.00000261] | valAへ進行 |
| valA | 512 | +0.00000274 | [-0.00000036, +0.00000597] | loss条件不通過 |

512のvalA tail loss差も`-0.00000035`、95%区間`[-0.00000987, +0.00000895]`で実質同等だった。
単純なパラメータ容量不足が主要因なら独立rootで改善が残るはずだが、その証拠は得られなかった。

事前規則ではvalAの平均loss差95%区間上端が0未満の場合だけ速度測定へ進む。候補がこの条件を
満たさなかったため、CPU速度と固定bestmoveを測らず終了した。

## 成果物

- widening: `scripts/widen_adapter_checkpoint.py`
- 学習: `scripts/run_root_representation_capacity.sh`
- valB: `results/root_representation_valB.json`
- valA: `results/root_representation_valA.json`
- logs: `/tmp/train_nnue_root_representation_hidden{256,512}_from_m0.log`
