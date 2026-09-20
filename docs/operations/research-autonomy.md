# 自律研究: 不足を人間待ちにしない

この方針は旧ジョブ・旧計画の「完全な除外証明がなければ保留」等に優先する。
科学的な未確定（仮説台帳のheldを含む）と、人間の回答待ちを区別する。
弱い証拠を支持と呼ぶこと、実験を実行済みと偽ること、安全制約の回避は許さない。

## 利用できる入力

`configs/research_data.json`を利用可能データのカタログとする。Floodgate 2025最終版は
双方R3500以上の38,351棋譜・5,710,803局面。開発・選抜・最終test・対局の分割を守る。
固定費用測定8棋譜、本番10,000棋譜、予備1,324棋譜は整備済み。
詳細な形式と来歴は`floodgate2025.md`。旧試行版は使わない。

人間による定期的なデータ増量を計画の前提にしない。ディスク上の棋譜を優先し、
来歴が不明な場合だけ対象・未確認事項をまとめて質問する。既知の説明を再質問しない。
既存エンジンでの自動対局は可。エンジン/重みhash、設定、seed、探索費用、開始局面、
全着手、結果、生成コード版と分割を保存する。5分超は外部計算phaseへ渡す。
追加の外部棋譜取得はユーザーの明示依頼がある場合だけ。

理想的教師・全pretraining祖先の完全な非重複証明は要求しない。
既知の重複照合範囲と未確認範囲を記録し、それに見合う主張に限定する。
既存test・日次seriesを都合よく差し替えず、未知の来歴を創作しない。

## 不足への順序と打切り

1. カタログ・対象実験の証跡・同じ仮説の過去の障害を確認する。
2. 既存入力での修復、代替測定、生成対局、費用削減、主張の限定を検討する。
3. 準備セッションで解消できない通常不足は`replan`でreviewへ渡す。計算は起動しない。
4. reviewは未実行/失敗を記録し、途中実装も検査・コミットしてから、`replan`または
   `abandoned`を返す。後者は当該手法の終了であって仮説の棄却ではない。
5. 同一仮説の自動再計画はジョブIDを跨いで累計2回まで。障害・試行・次の具体策を
   stateと次attemptのjob.jsonへ保存する。上限を超えるreplanは管理側でabandonedにする。
   仮説IDやジョブIDを付け替えて同じ不足を反復してはならない。

再計画は同じジョブの新attemptで新規Codexを起動する。正常に完了した段階からの
次実験は通常の`complete` + `next_jobs`であり、再計画回数とは別。
成績を見た後に成功閾値・抽出条件を緩めない。費用不足による縮小は探索的と明記する。
abandonedの依存ジョブはcancelledになり、永久待機させない。必要な後続はreviewで
独立に再設計する。同一仮説・未解決ジョブへの依存をnext_jobsで作ることは管理側も拒否する。

## 本当に人間が必要な場合

reviewの`needs_human`は以下だけに限定する（途中変更の検査・コミットは必要）。

- `license_contract`: 契約・権利上の判断。
- `payment`: 支払い・費用承認。
- `destructive_change`: 既存データを不可逆に変える等の高リスク操作。
- `unknown_data_provenance`: 使用予定のディスク上の棋譜の来歴不明。

単なる標本不足、理想的な独立性を証明できないこと、有意差なし、通常のコード修正は対象外。
安全な代替があるならまずそれを使う。承認待ちは当該ジョブと依存先だけを待機させ、
clean worktreeに戻して独立実験と日次評価は続ける。

replan/needs_humanのreviewはattempt内に`resolution.json`を作る:

```json
{
  "obstacle": "足りないものと証拠",
  "attempted": "試した手段と結果（再計画では前回との差も）",
  "next_task": "新しいprepareで行う具体的課題",
  "category": "unknown_data_provenance",
  "question": "人間にしか答えられない具体的な質問"
}
```

category/questionはneeds_humanだけ必須。内容は恒久結果文書とanalysis.mdにも記録する。
外部セッションで`status`を見ると質問・履歴が表示される。回答はテキストに保存し、
workerをpauseして終了を確認してから次を実行する。

```bash
bash scripts/research_loop.sh answer JOB_ID /absolute/path/answer.txt
bash scripts/research_loop.sh resume
bash scripts/research_loop.sh run
```

回答はstateのhuman_answersと次のjob.jsonに引き継ぐ。全ジョブの回答も各prepareの
human_decisionsへ渡し、同じ来歴を別実験で再質問しない。回答は包括的承認ではない。
否認の場合は安全な代替またはabandonedにする。既存のsandbox/権限を変更しない。

## 実行障害は別扱い

計算の非zero終了やtimeoutはreviewへ渡し、安全な修正再計画を可能にする。
Codex自体の認証障害・不正出力・コミット/検査失敗、手動中断、日次基盤故障は
実行エラーとしてblocked/pausedを維持する。検査していない変更の上で次実験を走らせない。
これらは通常の研究不足による保留ではなく、ログと成果物を確認して明示retryする。
全課題が完了/打切り/承認待ちの場合もrunは終了せず、Codexなしでenqueueまたは日次を待つ。

再開前の読み取り専用検査:

```bash
scripts/project_python.sh scripts/check_research_readiness.py
```

各prepareはその時点のカタログをattemptのdata-catalog.jsonにも保存する。
実験のplanには実際に使用した入力hashと分割・用途を記録する。
