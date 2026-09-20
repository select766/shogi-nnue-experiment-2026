# 最良候補の定点評価と成長グラフ

研究ループに、約24時間ごとの固定ベンチマークを組み込む。準備・解釈のCodexセッションは
使わず、管理プロセスが専用計算ジョブを投入して評価・集計・描画する。
日次計算もホスト側で実行し、5分超の計算をインタラクティブCodexに待たせない。

## 最良候補の決め方

`configs/research_champion.json`を「現在までに得た最良候補」の正本とする。
checkpointの更新日時や最小lossから自動選択しない。研究reviewセッションが、独立した
選抜実験に基づく根拠文書を残して、モデルID・checkpoint・配備済みファイル・選抜理由を更新する。
不確定なら現候補を維持し、候補の比較を別ジョブとして登録する。

初期候補は`decision-aligned-weight005-epoch9`。旧task-only対照に約+6.7 Eloの信号があるため
暫定採用したもので、標準HalfKPへの優位が証明済みという意味ではない。

配備済みONNX/headとエンジン、checkpointのSHA-256をモデルIDごとに登録する。
同じIDの重み・実行ファイル・説明を書き換えたら停止する。モデル交代は新しいIDを使う。
予約時の候補をジョブへ保存するため、後から最良候補のポインタを変えても予約済み評価は変わらない。
重み等を実行途中で変更した場合も、終了時のhash検査が不通過になり曲線へ登録しない。

モデルの選抜に日次ベンチマークの値を使わない。日次データでの係数調整や候補の多数比較もしない。
これらは開発の進捗を見る監視指標であり、新規独立test・正式な棋力検証の代わりではない。

## 固定条件（growth-v1）

| 項目 | 初期設定 |
|---|---|
| 正解率 | `data/accuracy_eval_10k/test.jsonl`全10,000局面 |
| 正解 | 保存されたbestmoveとの完全一致 |
| 正解率の探索量 | 1,000,000 nodes / 局面 |
| 基準相手 | HalfKP checkpoint83000由来の`bin/eval/nn.bin`、`bin/YaneuraOu-by-gcc` |
| 対局開始局面 | validationからabs(eval)<=500の200局面をseed20260920で固定抽出 |
| 対局数 | 各開始局面を先後反転した400局 |
| 対局の探索量 | 100,000 nodes / 着手 |
| 共通条件 | Threads=1、USI_Hash=256、定跡なし、最大512着手 |
| 並列数 | 4（上限4） |

対局開始局面は正解率用データとの盤面・手番・持駒の重複を除外する。
これは元棋譜や学習データからの独立性を保証するものではない。
正解率データ、対局開始局面、基準相手の実行ファイル・評価関数、評価コード、cshogiバージョンを
固定する。条件変更時は`configs/daily_benchmark.json`のseriesを新しくし、別曲線にする。
既存seriesへ異なる条件の点を混ぜない。各モデル固有のengine/dependencyはassetsに列挙する。
外部ファイルを暗黙に読む拡張エンジンは、そのファイルもassetsへ追加する。

日次対局は毎局isready/usinewgameを送り、開始SFENと全着手履歴をエンジンへ渡す。
両者とも毎手TTを消去する。旧`run_match.py`の手順は変更せず、日次用の固定手順を別に実装した。
勝敗理由・着手列を保存し、違法手や通信失敗はモデルの負けにせず評価失敗として停止する。
最大手数到達は引分。千日手・連続王手と宣言勝ちも扱う。

## 表示する指標

- 正解率: 一致数/10,000。Wilson 95%区間を表示。
- 対基準勝率: 勝数/全対局数。引分は勝ちに数えない。
- 補助線: `(勝数 + 0.5 × 引分数) / 全対局数`。引分の増減を区別できるよう併記。
- 対局の区間: 開始局面ごとの先後2局平均を1観測とする標本分散からの名目95%正規近似。
- 日付、モデル交代ラベル、標本数、勝/敗/分を保存。変化を見やすくするため縦軸は区間に合わせる。

400局では数Eloの差を確証できない。グラフの小差を即座に「成長した」と判定しない。
同じ固定開始局面の再評価を独立標本とみなして日を跨いで局数合算しない。
点推定の過去最大だけを描く処理も行わない。同じ候補の日も測定し、変化がなければ水平になる。

## 操作

```bash
# 固定データとhashの登録・空のグラフ作成。エンジンを起動しない
bash scripts/research_loop.sh benchmark-init

# 通常のループ。期限到来時に研究ジョブより先に定点評価する
bash scripts/research_loop.sh run

# 次回期限を待たず、定点評価を1件だけ予約（計算はrunが実施）
bash scripts/research_loop.sh benchmark-enqueue

# 状態・停止は通常ジョブと共通
bash scripts/research_loop.sh status
bash scripts/research_loop.sh pause
bash scripts/research_loop.sh pause --now
```

設定はresearch_loop.jsonのdaily_benchmarkで、既定enabled=true、interval_seconds=86400、
timeout_seconds=43200。通常のrunは研究キューが空でもプロセスを維持して次回を待つ。
この待機中はCodexを起動しない。`run --max-jobs N`はN件の研究課題で終了し、定点評価はNに数えない。
キューが空の場合はこの上限付きrunは定点評価後に終了する。workerが終了/停止中は評価しない。

初回は最初の研究課題より前に実施。その後は前回の予約から約24時間、研究課題の完了境界で実施する。
実行中の研究計算を割り込んで止めないので、長い課題では日次実行が遅れる。
新規研究計算はできるだけ6時間以内のchunkに分け、遅延を抑える。
停止期間に溜まった日数分を連続で実行せず、再開後に1件測る。

失敗・中断時は通常と同じblocked/pausedになり、部分結果を曲線に入れない。
ログとworker-N.jsonを確認し、`retry DAILY_JOB_ID --phase execute`、resume、runで再実行する。
完了JSONが既に保存済みで描画だけ失敗した場合、retryは再対局せず描画だけをやり直す。
未完了の場合は同じ候補と条件で全体を再実行する。prepare/reviewの再試行は不要。
設定/hash違反が予約時に見つかった場合はstatusのbenchmark_errorに理由がある。

## 保存先

```text
.research-loop/growth/growth-v1/
  protocol.json              # 条件、基準相手/データ/評価コードのhash
  accuracy.jsonl
  openings.jsonl
  champions/MODEL_ID.json     # 候補の根拠、配備パス、hash
  measurements/DAILY_ID.json  # 完遂した測定だけ。追記で履歴を保持
  growth.html                # ブラウザで開く入口
  growth.png / growth.svg    # 共有・保存可能な図
  history.csv                # 時系列データ
.research-loop/experiments/DAILY_ID/attempt-001/
  benchmark.json             # 予約時の候補・series
  worker-N.json              # 完了または途中までの正解率・対局明細
  progress-N.json
  accuracy.json / matches.json
  execution.json / execute.log
  benchmark-complete.json
```

このディレクトリはGit対象外なので定期バックアップに含める。/tmpのログだけに測定値を残さない。
日次の機械的な観測は個別の仮説判定とは分け、毎回Codexによる結果文書/台帳コミットは行わない。
日次推移から新しい研究判断をする場合は、別ジョブで仮説・検証・台帳・コミットの通常ルールを適用する。

## 検査

```bash
timeout 300 scripts/project_python.sh -m unittest tests.test_daily_benchmark tests.test_research_loop
timeout 300 scripts/nnue_python.sh -m unittest tests.test_growth_plot
```

固定条件/モデルの改変拒否、24時間期限と重複予約、予約時候補の固定、未完了除外、引分集計、
先後ペア、履歴送信、Codexなしの実行経路、描画・CSV・HTMLを検査する。
