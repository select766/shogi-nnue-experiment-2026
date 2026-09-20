# 研究ループ再開準備: 不可能な入力要求の撤回

対象: H-DECISION-REPLICATION、H-FLOODGATE-2025。2026-09-20。
これは運用修正と準備監査であり、新しい棋力実験ではない。新規対局・学習は0件。

## 停止原因と解消

旧`decision-replication-manifest/attempt-001`のprepareは約292秒で正常終了したが、
新規10,000棋譜と旧train/教師/選抜/対局/accuracyの**完全な**除外証跡を要求してblockedを返した。
原prepare.json、plan.md、handoff.md、input-audit.json、run.shは
`.research-loop/experiments/decision-replication-manifest/attempt-001/`に保存する。
旧run.shは未実行。execution.jsonのpreparation_deferred/exit_code nullは非実行の証跡である。
当時生成されたinventory実装と3検査も履歴としてコミットする。実装は常にheldを返す
旧監査専用であり、現行実験の開始判定には使用しない。

ユーザーの指定でその後整備したFloodgate 2025は、双方R3500以上38,351棋譜、
5,710,803局面と固定対局用10,000棋譜を持つ。既知の24,995固有局面を照合除外し、
棋譜・局面・履歴の全件検証を通過済み。詳細は[データ整備結果](../floodgate2025-preparation/README.md)。
今回`configs/research_data.json`に利用可能として登録し、2つのmanifestのhashを固定した。
これは全pretraining祖先が非重複であるという証明ではなく、その証明を前提条件にはしない。

旧ジョブは管理コマンドcancelで終了させ、途中成果物を残す。
後継は`configs/research_loop_replication_cost.json`の
`decision-replication-cost-floodgate2025`。既存の手順qualification後に固定cost8の16局で
費用と手順を確認し、正式20,000局を別chunkとして事前登録する。
旧ジョブの不足条件は後継へ引き継がない。H-DECISION-REPLICATIONは未検証へ戻す。
仮説の優先順位1〜6と科学的主張は維持し、運用不具合から新しい棋力仮説は作らない。

## 恒久的な予防

- 通常不足は安全な代替案へ再計画。同じ仮説ではIDを跨いで累計2回まで。
- 再計画時の障害・試行・次課題を保存し、新規prepareへ引き継ぐ。
- 実施不能な手法はabandoned。依存ジョブを中止し、独立実験と日次を続ける。
- 人間待ちは契約、支払い、破壊的変更、既存棋譜の来歴質問のみ。
- 科学的未確定と人間待ちは別。全祖先の完全分離や理想的教師を要求しない。
- セッション/認証/検査/コミットの実行障害は従来どおり停止し、安全確認前に自動再実行しない。

具体的形式・操作は[自律研究方針](../../../operations/research-autonomy.md)。
再開準備検査は`scripts/project_python.sh scripts/check_research_readiness.py`。
これはmanifest hash、入力存在、queue、clean worktree、台帳を調べ、Codex/engineを起動しない。
巨大データの再hash・CUDA・認証の実行確認は含まない。

## 検証

オフラインの模擬Codexで、新規セッション分離・計算中Codexなし、通常不足からの再計画、
上限到達・ID変更でも上限共有、依存の連鎖中止、承認待ち中の独立課題継続、回答引継ぎ、
不適切な人間待ち分類の拒否、旧deferred互換、既存のpause/recover/error処理を検査する。
併せて日次評価、データ整備、仮説台帳、旧inventoryとカタログ検査の回帰テストを実施する。
実際の有料Codexセッションや長期workerの起動は今回の検査には含めない。

検査結果: project環境で71件、描画用nnue環境でgrowth plotの1件、合計72件通過。
最初に描画テストもproject環境で呼んだ際はmatplotlib不在でimportに失敗したため、
日次実装が実際に使用するnnue wrapperで再実行した。環境への追加インストールは行っていない。
台帳検査は34仮説・33結果文書を確認。git diff --checkも通過。

```bash
scripts/project_python.sh -m unittest tests.test_research_loop tests.test_daily_benchmark \
  tests.test_research_hypotheses tests.test_prepare_floodgate \
  tests.test_decision_replication_manifest tests.test_research_readiness tests.test_match_protocol
scripts/nnue_python.sh -m unittest tests.test_growth_plot
scripts/project_python.sh scripts/check_research_hypotheses.py
```
