# 実験結果

- [Root-grouped router実験](root-grouped-router-20260823/README.md)
- [Root-grouped router大規模追試](root-grouped-router-scale-20260824/README.md)
- [Proxy gap検証](proxy-gap-20260824/README.md)
- [実探索leaf routerの棋力・探索量転移](search-budget-and-strength-20260824/README.md)
- [Search-aware objective / on-policy pilot](objective-onpolicy-20260825/README.md)
- [Search utility直接蒸留pilot](search-utility-distillation-20260825/README.md)

- [固定10,000局面の最善手一致率](accuracy-eval-10k-current/README.md)
- [Checkpoint 180のgate・oracle診断](gate-diagnostics-lambda05-180/README.md)
- [Gateモデル変更実験: entropy・balance・1.5-entmax](gate-model-change-20260823/README.md)
- [Router蒸留実験](router-distillation-20260823/README.md)
- [DNN gateの手数別loss](check-loss-per-gameply-dnn-backbone-v4/README.md)
- [NNUE gateの手数別loss](check-loss-per-gameply-nnue-backbone/README.md)
- [旧pairedデータの手数別loss](check-loss-per-gameply/README.md)
- [Expert Blending実行速度](expert-blending-speed.md)

これらは測定結果の証跡であり、掲載コマンドは当時のもの。再実行には
`../../operations/training-and-evaluation.md`の現行ラッパーを使う。
