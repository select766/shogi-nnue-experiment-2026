# 実探索leaf loss棋力追試 計画 (2026-08-26)

対象は`H-LEAF-LOSS-STRENGTH`である。既存の固定10万nodes自己対局400局では、実探索leaf学習候補が
checkpoint 510対照に200勝185敗15分、`+13.0 Elo`、95%区間`[-20.4,+46.5]`だった。この方向が
標本を増やして0より上へ分離するかを、モデルを変更せず測る。

## 固定条件

- candidate: `tmp/proxy_gap_actual500k_release`
- control: `tmp/proxy_gap_control510_release`
- engine: `bin/YaneuraOu-expert-blending.sh`
- 各engine `Threads=1`、固定100,000 nodes/着手
- 開幕: `data/accuracy_eval_10k/validation.jsonl`から`abs(teacher_eval)<=500`をseed `20260824`でshuffle
- 既存200開幕（400局）の後続1,800開幕を使い、各開幕を先後交換して3,600局追加
- 既存分と合わせて固定4,000局

同じseedと`start=200`により既存開幕との重複を避ける。50開幕/100局の独立chunkに分ける。第1 waveの
初動実測後に並列数を8から12へ増やしたが、`systemd-oomd`がuser sliceのメモリ圧迫を検知してCodexを
含むscopeを終了した。12並列時は空きメモリが約5.8GiBまで低下した。6並列での再開直後も空きが12GiB
まで低下したため完了chunkがない時点で停止し、最終的に並列上限4、開始時の`MemAvailable`下限20GiBへ
固定した。固定nodes・各engineの`Threads=1`・開幕・局数・判定規則は変更していない。candidateの勝ちを
win、負けをloss、引き分けを0.5点として集計する。再実行時は完了chunkを再利用し、排他lockで二重起動を
拒否する。集約時に全4,000局、各SFENの先後1局ずつ、SFENとcandidate手番の重複なしを検査する。

## 事前判定

途中結果では停止せず、固定4,000局を完走してElo差と通常の95%区間を計算する。

- 95%区間下端が0より大きい: `支持`
- 95%区間上端が0以下: `棄却`
- 95%区間が0を跨ぐ: `測定完了`（方向は未確定）

最後の場合も追加対局だけを際限なく続けず、実探索leaf lossを単独のモデル選抜基準にはしない。
bestmove一致率は既に候補が対照比`-0.20`ポイントだったため、この追試の判定には再利用しない。
