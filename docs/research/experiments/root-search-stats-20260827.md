# 浅いroot探索統計 実験計画 (2026-08-27)

対象は`H-ROOT-SEARCH-STATS`である。静的combined 7特徴対照へ、本探索前に固定HalfKPで一度だけ
得る1024-node MultiPV統計6次元を追加し、未知leaf集合のlossと固定bestmoveを比較する。

## 特徴

`bin/YaneuraOu-by-gcc`、`EvalDir=bin/eval`、`Threads=1`、`MultiPV=4`、`go nodes 1024`を使う。
探索前にhashをclearし、root手番から見た最終MultiPV scoreを降順に並べる。合法手が4未満なら
得られた本数だけを使う。

1. top1 scoreを2000cpで割り`[-1,1]`へclip
2. top1-top2 marginを1000cpで割り`[0,1]`へclip（1本なら0）
3. top1-last widthを1000cpで割り`[0,1]`へclip
4. score標準偏差を1000cpで割り`[0,1]`へclip
5. `softmax(score/200cp)`のentropyを`log(MultiPV本数)`で正規化（1本なら0）
6. 得られたMultiPV本数/4

mate scoreは符号付き`32000-ply`cpへ写像する。特徴抽出器、学習cache、accuracy時のonline抽出は
同じ関数を使う。候補入力は静的combined 7特徴＋この6特徴の計13次元である。

## 学習と選抜

- initial: M0。追加13列を0初期化し初期gateを保存する。
- train: `tmp/proxy_gap_v2/train` 99,840 root x 8 leaf、1 epoch。
- selection: valB 19,968 root。候補は1条件だけで、matched controlは既存の静的combined 7特徴
  （同じM0、同じ99,840 root、同じ1 epoch）。
- experts/backbone固定、adapter-only、mean group loss、batch 256、seed 42。

valBの対応group loss差が負でbootstrap 95%区間上端が0以下なら、未使用valA 19,968 rootへ進む。
通らなければ`棄却`し、固定bestmoveへ進めない。

## 配備評価と事前判定

valAでもloss差の95%区間上端が0以下ならONNX/C++形式へexportし、固定validation 10,000局面・
100万nodesで静的combined対照と対応比較する。浅い探索時間を本探索時間へ加えた追加率も記録する。

1. valA lossを改善し、固定validation bestmove差が正、CPU追加時間が10%以内なら`支持`。
2. valA loss差の95%区間下端が0以上、bestmove差が0以下、またはCPU追加時間が10%超なら`棄却`。
3. それ以外は`測定完了`。validation bestmoveが正なら固定test 10,000局面も追試として記録するが、
   仮説判定は事前固定validation条件で行う。
