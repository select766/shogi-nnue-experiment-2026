# 研究ジョブキュー

ユーザーがホストのターミナルで実行管理スクリプトを起動する。

```text
仮説台帳の優先順位でキュー選択
 → 新規Codex: 実装・計画・run.sh・引継ぎを作成して終了
 → bash run.sh: 学習/評価を実行（Codexプロセスなし）
 → 新規Codex: 結果解釈・台帳更新・コミット・次課題を提案して終了
 → 管理側が提案を検査してキューへ反映 → 次の課題
```

LLMを起動するのはprepare/reviewだけ。計算中の待機は通常のPythonプロセスが行い、
LLMへの問い合わせはない。全phaseは直列。新しい課題の優先順位は
`docs/research/hypotheses.md`の未完了仮説priorityから毎回読む。キューへ順位を複製しない。
途中まで準備/計算した課題は、別課題を開始する前にその解釈まで終える。

約24時間ごとの[日次定点評価](daily-benchmark.md)は研究課題の間に優先して実行する。
これはCodexを使わない組込み計算ジョブで、最良候補の正解率・勝率と成長グラフを更新する。
日次評価が有効な通常runは、研究キューが空でもモデルを起動せず次の期限を待つ。

## 初回設定と起動

ホスト環境でCodex CLIへログイン済み、プロジェクトPython環境が構築済みであることが前提。
Codexの起動方法は公式の[非対話モード](https://learn.chatgpt.com/docs/non-interactive-mode)に従う。
ローカル`codex-cli 0.155.1`で`exec --ephemeral --json --output-schema --output-last-message`と
`--approve-for-me`を確認した。resume/forkは使わず、モデルはユーザーのCLI設定を引き継ぐ。

```bash
bash scripts/research_loop.sh init
bash scripts/research_loop.sh enqueue configs/research_loop_seed.json
bash scripts/research_loop.sh status

# ユーザーのターミナル（sandbox外）で起動。日次評価が有効なら空キューでも待機
bash scripts/research_loop.sh run

# 最初の1課題だけ完了して終了する場合
bash scripts/research_loop.sh run --max-jobs 1
```

初期課題は`H-MATCH-PROTOCOL`の1時間以内の短期検証。大規模対局は後続ジョブに分ける。
initは既存キューを上書きしない。初期化済みなら最初の2コマンドは繰り返さない。
runはforegroundなので、長い運用にはtmux等を使う。Codexインタラクティブセッションに
長時間runの監視を依頼しない。

GPU・エンジンを起動するrun.shは、ユーザーが起動した管理プロセスのホスト権限を引き継ぐ。
Codex側は`--approve-for-me`によるworkspace-writeと自動承認レビューを使う。
無制限のsandbox回避は指定しない。承認失敗・認証エラーはblockedで停止する。

## 外部セッションから確認・停止・再開

以下は短時間で終了する管理操作なので、別のCodexインタラクティブセッションから実行できる。
statusはモデルを起動しない。

```bash
bash scripts/research_loop.sh status

# 現在のphase完了後、次のCodex/計算を開始せずworkerを終了する
bash scripts/research_loop.sh pause

# 現在のCodexまたは計算のプロセス群を終了し、ジョブをblockedにする
bash scripts/research_loop.sh pause --now

# 停止要求の解除。終了したworkerは次のrunで起動する
bash scripts/research_loop.sh resume
bash scripts/research_loop.sh run
```

計算中のpauseは計算完了後・review起動前に止まる。resume/runで新規reviewから継続する。
pause --nowやCtrl-C/SIGTERMはプロセス群へTERMを送り、猶予後KILLする。
OSのsuspendではなく中断なので、途中のモデル/データが不完全な可能性がある。
run.shのdaemon化・setsidによるプロセス群からの離脱は禁止。

statusのJSONには各課題のphase/status/priority/依存、実験フォルダ、worker PID、
子PID、開始時刻、heartbeat、現在のログパスがある。`worker_present`はlockの状態。
`active`が残りworkerがいない場合はrecover手順を使う。計算の細かい進捗は本体が書く
summary/artifacts/logで確認する。

## 中断・失敗・異常終了

計算の非zero終了・timeoutはexecution.jsonへ記録し、新規reviewへ渡す。
Codexの失敗、不正な出力、検査不通過ではキュー全体をpaused、課題をblockedにする。
自動再試行はしない。

```bash
# まず実験フォルダと終了理由・ログ・成果物を確認する
bash scripts/research_loop.sh status

# 以下3つは代替操作。状況に合う1つを選ぶ。
bash scripts/research_loop.sh retry match-protocol-pilot --phase review
bash scripts/research_loop.sh retry match-protocol-pilot --phase execute
bash scripts/research_loop.sh retry match-protocol-pilot --phase prepare

bash scripts/research_loop.sh resume
bash scripts/research_loop.sh run
```

reviewは既存結果を新規セッションへ再度渡す。executeはrun.shを再実行するため、
handoffの再開仕様を先に確認する。prepareは新attemptを作る。旧回答・プロセス記録は保存する。
最初の準備が失敗して残った当該課題の変更はretry prepareで修復可能。
無関係なユーザー変更が混ざった場合は先に手動で整理する。

SIGKILL・ホスト停止後は子とプロセス群の停止を確認する。worker lockは子へ継承され、
子が動いている間の二重実行を防ぐ。`recover`は生きた記録済みプロセス群があれば拒否する。

```bash
bash scripts/research_loop.sh recover
# activeだった課題がblockedになる。成果物を確認してretryのphaseを選ぶ。
```

異常終了とファイル更新を跨ぐ厳密なexactly-once実行は保証しない。曖昧な完了状態は人が確認する。
不要な課題は`cancel JOB_ID`で取り消せる。成果物は削除せず、依存ジョブは自動解除しない。

## 実験フォルダと引継ぎ契約

```text
.research-loop/
  state.json                         # 管理側だけが更新するキューと稼働状態
  state.lock / worker.lock / workspace.lock
  experiments/JOB_ID/attempt-001/
    job.json                         # 課題、仮説ID、元commit、以前のattempt
    plan.md                          # 対照、標本、成功基準、予算、停止規則
    handoff.md                       # 実装、出力の読み方、再実行/再開仕様
    run.sh                           # cwd=repo root, RUN_DIR=実験フォルダ
    prepared.json                    # 上記3ファイルのSHA-256と計算timeout
    prepare-prompt-001.md             # 実際のセッション指示
    prepare.schema.json / prepare.json
    execute-process-001.json         # argv、終了理由、exit code、時間、ログ
    execution.json                   # 最新の計算結果メタデータ
    result-summary.md                # 本体が書く64KiB以内の要約
    artifacts/                       # metrics、結果、詳細ログ参照等
    review-prompt-001.md
    review.schema.json / review.json # 完了/保留、commit、report、次課題
    analysis.md                      # 解釈と次の課題
    prepare.log / execute.log / review.log  # /tmpの実ログへのsymlink
```

準備済み3ファイルが実行前に変わった場合は停止する。変更したい場合はprepareをやり直す。
大きいモデルは通常のlogs/等へ保存し、handoff/summaryにパスを残してよい。
本体stdout/stderrとCodexイベントは/tmpへ保存し、管理側のメモリへ読み込まない。
/tmpは消える可能性があるため、重要な値と再現条件は恒久結果文書/JSONへ保存する。
.research-loop/はGit対象外なので、継続運用・別ホスト移行にはこのフォルダもバックアップする。

reviewは恒久結果文書と台帳を更新して当該変更をコミットする。管理側は実在する新commit、
元commitとの包含、結果文書・台帳のcommit内容一致、clean worktree、台帳検査、
次課題ID/依存/循環を確認し、doneと次課題を同じstate更新で反映する。
completeは記録完了の意味で、仮説の支持とは限らない。

## 課題の追加と設定

ジョブJSONの4キー。hypothesisは台帳の未完了ID、depends_onは既存または同時提案内のjob ID。

```json
{
  "id": "decision-replication-stage1",
  "hypothesis": "H-DECISION-REPLICATION",
  "task": "固定候補追試用の新規開始局面とmanifestを作る。正式対局は次ジョブへ分ける。",
  "depends_on": ["match-protocol-pilot"]
}
```

外部セッションはこのファイルを作り`enqueue PATH.json`で追加できる。reviewのnext_jobsも同じ形式。
順位変更は仮説台帳で行い、state.jsonは直接編集しない。実験コードの編集前はpauseしworker終了を確認する。
キュー操作とstatusは実行中も可能。実行可能課題がない場合、日次評価が無効ならworkerは終了する。
日次評価が有効なら次の期限を待つ。max-jobs指定時は空キューで終了する。

`configs/research_loop.json`の設定:

| 項目 | 初期値 | 意味 |
|---|---:|---|
| session_timeout_seconds | 1800 | prepare/review全体の上限。超過はblocked |
| max_compute_seconds | 86400 | 外部計算上限。prepareはこの範囲内で指定 |
| poll_seconds | 1 | 停止・子終了の確認間隔。LLM呼出しではない |
| terminate_grace_seconds | 10 | TERM後にKILLするまでの猶予 |
| codex_command | argv配列 | 必要ならモデル/プロファイルを追加。shell文字列ではない |

セッション内の個々のコマンドはtimeout 300を使い、5分超は外部へ送るよう指示する。
これは実行規約であり、任意の生成コードの挙動を静的に証明するsandboxではない。
管理側が強制するのはセッション全体と外部計算全体のtimeout。1日超の計算はchunkや
checkpointでジョブを分割するか、ユーザーが上限を変更する。
設定変更は次のworkerから有効。`--state-dir`/`--config`はサブコマンドの前へ指定する。
別state-dirでも同一repositoryのworkerは1つに制限する。

## 動作検証

```bash
timeout 300 scripts/project_python.sh -m unittest tests.test_research_loop
```

模擬Codexと短い実プロセスで、fresh session分離、計算中にprepareプロセスがないこと、
優先順位・依存、JSON不正、計算失敗、timeout、境界停止・即時中断、二重起動、
dirty worktree保護、キュー更新の原子性を検査する。学習やAPI呼出しは行わない。
