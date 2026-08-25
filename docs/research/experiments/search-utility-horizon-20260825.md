# Search utility探索量安定性 実験計画 (2026-08-25)

対象は`H-SU-HORIZON`である。DNNは各候補についてrootで1回だけ実行し、そのblendを探索終了まで
固定する。探索中のleafごとにDNNを呼ぶ方式は使わない。

## 仮説と比較

現行M0のroot gateを候補0とし、各expert logitへ個別に`+1.0`した候補1--8を加える。固定した
同一128 rootについて、各候補を10,000、100,000、1,000,000 nodesで探索する。各horizonで得た
全unique bestmoveは、候補とは独立した固定`nn.bin`により、同一rootから1,000,000 nodesの
制限手探索で一度だけ評価する。このroot手番視点scoreをutilityとする。

したがって比較するのは候補探索自身のscoreではなく、同じ独立評価器で測った9候補のutility順位である。
同じ指し手を返した候補には同じutilityを与える。

## 指標の定義

- **候補順位相関**: 両horizonでutility範囲が0より大きいrootにおける、9候補utilityのtie補正
  Spearman相関。root平均とbootstrap 95%区間を報告する。
- **oracle候補集合重複率**: 各horizonで最大utilityを得た候補番号集合に共通要素があるrootの割合。
- **候補指し手一致率**: 同じroot・同じ候補が両horizonで同じbestmoveを返す割合。
- **有効教師率**: 最大候補utilityが候補0を10cp以上上回るrootの割合。
- **教師L1差**: 最大utility候補群のgate平均。ただし10cp未満なら候補0 gateを使うというpilotと同じ
  教師定義について、両horizonの8次元分布間L1距離を測る。

## 事前判定規則

主比較は10,000対1,000,000 nodesとする。次の全条件を満たせば探索量安定性を`支持`とする。

1. informative rootが全体の10%以上ある。
2. 候補順位相関のroot平均bootstrap 95%区間の下端が0.30より大きい。
3. oracle候補集合重複率のWilson 95%区間の下端が0.40より大きい。
4. 1,000,000 nodesの有効教師率のWilson 95%区間の下端が0.05より大きい。

条件2または3の95%区間上端が閾値未満、または条件4の上端が0.05未満なら`棄却`する。それ以外は
`不確定`とする。中間の100,000 nodes比較、候補指し手一致率、教師L1差は解釈と次の候補幾何設計に使う。

## 実行条件

- root: `tmp/search_utility_v1/val/roots.bin`先頭128件。pilot学習には使っていないheld-out側。
- candidate: `bin/YaneuraOu-expert-blending`、M0 release
  `tmp/proxy_gap_actual500k_release`、`Threads=1`、定跡off。
- reference: `bin/YaneuraOu-by-gcc`、`bin/eval/nn.bin`、`Threads=1`、定跡off。
- workers: 4、各探索前に`isready`を通してTTを初期化する。
- bootstrap: root単位10,000回、seed 42。
