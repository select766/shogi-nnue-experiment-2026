# Floodgate 2025: 双方R3500以上の実手数付き評価データ

## 入力と採用条件

ユーザー指定の[公式2025年アーカイブ](https://wdoor.c.u-tokyo.ac.jp/shogi/archive/wdoor2025.7z)を
2026-09-20に取得。351,630,898 bytes。[配布元の公開SHA256](https://wdoor.c.u-tokyo.ac.jp/shogi/)と
`423504903f211316ec40f9c3d8dcc2e5faf392fc4d1334ede1489242a9d09baa`の一致を確認した。
原アーカイブを`dataset/floodgate2025/raw/wdoor2025.7z`へ保存する。

採用の必須条件は、原CSAの`'black_rate:`と`'white_rate:`が**両方3500以上**であること。
3500ちょうどを含む。レーティングの欠損、非数、重複記載、対局者名との不一致は採用しない。
現在のレーティングを遡って適用したり、相手の値から推定したりしない。

研究用派生データでは追加で、2025年開始・EVENTと原ファイル名の一致・平手初期局面・
全着手合法・正常終局を要求する。時間切れ、通信異常、中断などは除外し原本は残す。
初期局面と全USI着手列のhashが一致する棋譜を重複として除外する。
採用・除外の全件について、archive内パスまたは理由を保存する。

## 整備済みデータ

`dataset/floodgate2025/curated-v1/`:

| ファイル | 内容 | 件数 |
|---|---|---:|
| `games.jsonl.gz` | 棋譜ID、原本hash、日時、対局者・rating、初期局面、全着手、終局、各手評価値・時間 | 38,351棋譜 |
| `positions.jsonl.gz` | 各着手前のSFEN、実手数、着手、棋譜ID、score、旧局面との重複フラグ | 5,710,803局面 |
| `development.jsonl` | 教師診断・開発用、1棋譜1局面 | 11,454 |
| `validation.jsonl` | 候補選抜用、1棋譜1局面 | 7,678 |
| `test.jsonl` | 最終確認用、1棋譜1局面 | 7,739 |
| `match.jsonl` | 対局開始局面用、1棋譜1局面 | 11,332 |
| `rejections.jsonl.gz` | 除外した原棋譜と理由 | 90,492棋譜 |
| `manifest.json` | 原本・設定・実装・出力hash、月別/手数帯/対局者分布、除外元一覧 | — |

全局面には序盤から終盤まで残す。`game_ply=1`は初手を指す直前、`played_plies=0`。
SFEN末尾を便宜的に1へ置き換えず、全着手の再生から実際の手数を得る。
`bestmove`は原棋譜の実着手であり、別途100万nodes探索して生成した教師ではない。
既存評価器へ入力した場合は「100万nodesで探索した手と原棋譜の手の一致率」を測る。

`reported_score_black`は原CSAの整数評価コメント。CSAの[仕様](https://www.computer-shogi.org/protocol/record_v3.html)に
従い先手視点を保持し、互換フィールド`eval`は手番視点へ変換する。
評価値は対局者が着手時に報告したものなので、尺度・探索条件は統一されていない。
欠損は`null`かフィールド省略で表し、0として補完しない。mate相当値も勝手にcpへ直さない。

## 分割と重複管理

初期局面+全着手列のhashに固定seedを加えて、棋譜単位でdevelopment 30%、validation 20%、
test 20%、match 30%へ振り分ける。split間に同一棋譜は入れない。
この比率はモデル成績を見ずに決めた。testの件数を無理に1万へ増やすための再抽出はしない。

固定JSONLは各棋譜から最大1局面。24手目以降、消費時間が1秒以上の着手を候補にし、
局面選択はhash順で決める。ゼロ秒除外は定跡手を減らすproxyであり、完全な定跡除外ではない。
matchのみ、報告された評価値の絶対値が500以下を要求する。これは互角性の近似フィルタで、
共通エンジンによる評価ではない。

全固定JSONL間で「盤面・手番・持駒」の重複を除外し、設定に列挙した過去のaccuracy 23,000行、
utility教師2,000行、proxy-gap開始局面200行、旧14,000対局の開始局面との重複も除外する。
これらを合わせた既知の固有局面は24,995。教師生成・過去評価の全ファイルや巨大pretraining祖先を
網羅した検査とは主張しない。詳細な照合範囲と入力hashはmanifestに保存する。

全局面ファイルは定跡・反復・転置も保持する。学習・評価に無条件で混ぜずsplitを守り、
手数別分析の区間推定には棋譜単位のクラスタリングを用いる。
既存`data/accuracy_eval_10k/`や日次growth-v1は変更していない。

## 追試用の固定集合

`dataset/floodgate2025/match-plan-v1/`:

- `cost8.jsonl`: 費用測定専用の8棋譜・8局面。
- `openings10000.jsonl`: 棋力比較用10,000棋譜・10,000局面。先後反転なら20,000局。
- `reserve.jsonl`: 残り1,324棋譜。対局結果が悪いことを理由に差し替えない。
- `manifest.json`: 抽出seed、元poolと各出力のhash。

固定集合はhash順で抽出し、費用測定と本番を分離した。各局面に`initial_sfen`と`history_usi`が
あり、元棋譜の開始から再生できる。対局runnerへ元棋譜履歴を渡すか、抽出SFENから新しい対局を
始めるかは、試験計画で先に決める。既存runnerがこれらの追加フィールドを自動利用するわけではない。
これらは**入力の整備**であって、対局・強さの測定・完全な祖先分離の証明はまだ行っていない。

## 再生成・検証

```bash
bash scripts/download_floodgate2025.sh
scripts/project_python.sh -m train_nnue.prepare_floodgate
scripts/project_python.sh scripts/verify_floodgate2025.py dataset/floodgate2025/curated-v1
scripts/project_python.sh scripts/select_floodgate_openings.py
scripts/project_python.sh -m unittest tests.test_prepare_floodgate
```

`prepare_floodgate`は7zをストリームで読み、各memberのCRC・sizeを検査する。原ファイルを
大量に展開する必要はない。既存出力ディレクトリは上書き拒否。設定変更や再試行には新しい
`--output`を指定する。途中終了時は`complete: true`のmanifestがないので完了扱いしない。
`--limit N`は動作確認専用で、正式成果物として扱えない。gzip出力は時刻を固定し、同一入力・実装で再現可能。

`verify_floodgate2025.py`は全採用棋譜と全局面の対応を再生検証し、各固定集合の履歴・合法手・
双方rating・棋譜分離・局面重複・旧集合除外・ファイルhashを検査する。
原本と派生データはGit対象外なので、データバックアップへ含める。
取得時に中断した`raw/extracted/`は未使用の作業途中ファイルであり、入力にしない。
原棋譜の正本はhash確認済みの`raw/wdoor2025.7z`だけである。
`curated-v1-initial/`と`match-plan-v1-initial/`は終局形式対応前の試行版で、正式評価に使わない。
正常な入玉宣言の二重記録、summaryのみの最大手数引分・連続王手終局を保持する最終版を使用する。

ループ内prepare/reviewの5分制限は維持するが、ユーザーが直接依頼するループ外の整備・
デバッグには適用しない。最終版の全件生成は108.00秒。全件検証の記録は結果文書に保存した。
