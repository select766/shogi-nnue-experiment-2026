# Search utility候補幾何 実験計画 (2026-08-26)

対象は`H-SU-GEOMETRY`である。候補数を基準gateを含む9個に固定し、候補探索と独立utility探索を
ともに1,000,000 nodesへ揃えて候補方向だけを比較する。

## 候補集合

- `axis-1`: 基準と各expert logitへ`+1`した8候補。現行対照。
- `axis-2`: 基準と各expert logitへ`+2`した8候補。半径だけを広げる。
- `cyclic-contrast-2`: 基準と、expert `i`へ`+2`、`(i+1) mod 8`へ`-2`した8候補。
  gate simplex上でexpert間を移動し、全logit共通成分を増やさない。

全候補でDNNはrootに一度だけ適用し、blendを探索終了まで固定する。

## Screeningと事前判定

`tmp/search_utility_v1/val/roots.bin`の128--191番、同一64 rootを使用する。`axis-1`は直前の
H-EXPERT-DIVERSITYで得たM0結果を再利用し、他2条件だけ追加収集する。

基準は有効教師率15.625%、oracle gain p90 35.5cp、candidate bestmove多様度1.65625である。
次の全条件を満たす幾何があれば、学習可能な信号が増えたとしてheld-out教師再現性の追試へ進む。

1. 有効教師率が相対25%以上、すなわち19.53125%以上。
2. oracle gain p90が相対25%以上、すなわち44.375cp以上。
3. candidate bestmove多様度が相対10%以上、すなわち1.821875以上。

通過候補がなければ`棄却`する。通過候補がある場合は未使用rootでteacherを追加生成し、同じadapter-only
蒸留を行う。held-out teacher cross entropyをM0より改善すれば`支持`、改善しなければ`棄却`とする。
