# ドキュメント索引

このファイルをCodexと開発者の入口とする。`operations/`だけが現行の実行手順、
`research/`は現在の研究判断、`archive/`は参照専用の履歴である。

## 最初に読む

1. [Python環境](operations/python-environments.md): 2つの環境の責務と再構築方法。
2. [学習と評価](operations/training-and-evaluation.md): 現行データによる学習、変換、loss評価、最善手一致率評価。
3. [データ](operations/data-preparation.md): 現行データの場所、形式、再生成方針。
4. [2026年8月研究報告](research/update-report-202608.md): これまでの根拠と実験判断。
5. [研究仮説レジストリ](research/hypotheses.md): 未検証仮説の状態、優先順位、実験結果との対応。
6. [2026年9月研究レビュー](research/results/research-review-20260920/README.md)と
   [次期実験計画](research/experiments/research-next-20260920.md): 既存結果の再点検と現在の実験方針。

## Operations

- [研究ジョブキューと外部セッションからの操作](operations/research-loop.md)
- [最良候補の日次定点評価と成長グラフ](operations/daily-benchmark.md)
- [Python環境](operations/python-environments.md)
- [学習と評価](operations/training-and-evaluation.md)
- [データ準備](operations/data-preparation.md)
- [最善手一致率の仕様](operations/accuracy-evaluation.md)
- [Expert分析と可視化](operations/expert-analysis.md)
- [やねうら王ビルド (Linux)](operations/engine-build-linux.md)
- [やねうら王ビルド (Windows)](operations/engine-build-windows.md)

## Research

- [2026年8月の依頼](research/update-plan-202608.md)
- [調査報告と次期実験計画](research/update-report-202608.md)
- [研究仮説レジストリ](research/hypotheses.md)
- [実験結果索引](research/results/README.md)

## Archive

[archive/README.md](archive/README.md)以下は、過去の設計判断や障害調査の証跡である。
コマンド、パス、依存バージョンは現行環境と互換性がない。
