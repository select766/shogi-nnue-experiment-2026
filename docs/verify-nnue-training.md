# 評価関数の学習の正当性検証

目的: 将棋において常識的な指し手が出るまで学習を行い、学習機に欠陥がないことを確認する。

学習機に問題がない場合、epoch数を増やして、１０分程度学習すれば以下の検証をパスするはず。

# 検証
学習した評価関数をやねうら王に読ませる。2つの局面で検証する。

初手に対して1秒思考したら、"2g2f"か"7g7f"が出力されるはず。
input:
position startpos
go byoyomi 1000
output:
bestmove 2g2f

2g2fが指された局面に対して1秒思考したら、"3c3d"か"8c8d"が出力されるはず。
input:
position startpos moves 2g2f
go byoyomi 1000
output:
bestmove 3c3d

