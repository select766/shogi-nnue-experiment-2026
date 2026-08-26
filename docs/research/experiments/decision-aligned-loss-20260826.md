# Root decision-aligned目的 実験計画 (2026-08-26)

対象は`H-DECISION-ALIGNED-LOSS`である。同じroot、experts、データ量でleaf平均KLだけの対照と、root候補手
utilityから直接作る方向教師を併用したrouterを比較する。

## 教師

既存`tmp/search_utility_v1/pilot/details.jsonl`の2,000 rootを再利用する。各rootはM0 gateと8 expert軸
`+1`候補を探索し、候補bestmoveを独立固定NNUEで制限手評価済みである。基準より10cp以上改善した場合、
utility最大のexpert軸（同点は一様）を8次元hard targetにする。それ未満はM0 gateを保持する。

既存soft教師は変更rootでもbias後gateを目標にしたため平均L1変化が約0.05だった。本実験は候補手utilityの
勝者方向を直接分類し、単なる同じ教師のデータ増量ではない。

## 学習と選抜

- train/validation: 同じ1,600/400 root、validation先頭384を選抜に使用。
- initial M0、experts固定、adapter-only、batch 64、10 epoch、seed 42。
- leaf-KL control: task weight 1、decision weight 0。
- candidates: task weight 1、decision CE weight `0.01, 0.05`。

validation hard teacher CEがM0より`0.005`以上改善し、leaf group lossがtask-only対照比`+0.0001`以内の
候補だけ、固定validation 10,000局面・100万nodesへ進める。複数ならCE最小を1つ選ぶ。

## 事前判定

固定bestmoveがtask-only対照より正方向なら固定testと事前固定4,000局自己対局へ進む。自己対局Eloの
95%区間下端が0より上なら`支持`する。bestmoveが0以下なら`棄却`し、test・自己対局へ進めない。
bestmoveは正だが対局区間が0を跨ぐ場合は`測定完了`とする。
