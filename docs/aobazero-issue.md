# AobaZero PyTorch 移植: 現状と課題

## 概要

AobaZero (256x19 ResNet, Swish) の重みを PyTorch に変換し、棋譜データに対する指し手一致率・勝敗一致率で検証を行っている。

**目標:** 指し手一致率 > 30%, 勝敗一致率 > 70%
**現状:** 指し手一致率 ~4%, 勝敗一致率 ~58% (ほぼランダム)

## 完了済みの作業

| ファイル | 内容 | 状態 |
|---------|------|------|
| `src/train_nnue/aobazero_model.py` | PyTorch モデル定義 (256x19, Swish, policy 2187, value 1) | 完了 |
| `src/train_nnue/convert_aobazero_weights.py` | テキスト重み → PyTorch state_dict 変換 | 完了 |
| `src/train_nnue/aobazero_features.py` | 362ch 入力特徴量エンコーダ | **バグあり** |
| `src/train_nnue/verify_aobazero.py` | 指し手/勝敗一致率の評価スクリプト | **バグあり** |
| `aobazero/weights/aobazero_w4636.pt` | 変換済み重み (23.4M パラメータ) | 完了 |
| `data/accuracy_eval/test.jsonl` | テスト用棋譜データ (1000局面) | 完了 |

## 発見されたバグ

### Bug 1 (CRITICAL): チャンネルオフセットが14ずれている

**場所:** `aobazero_features.py` の `encode_features()`

**原因:** C++ の `set_dcnn_channels()` では、TWO_HOT エンコーディングであっても駒配置セクションに28チャンネル分を確保する (`add_base = 28`)。Python 実装では14チャンネルしか確保していないため、持ち駒以降のすべてのチャンネルが14ずれている。

**C++ のチャンネルレイアウト (正しい):**
```
  0- 13: 駒配置 (TWO_HOT: 自分+1, 相手-1)
 14- 27: [未使用、ゼロ]  ← Python で欠落
 28- 41: 持ち駒 (7種 × 2 = 14ch)
 42- 44: 千日手カウント (3ch)
 45-269: 過去手ステップのパディング (225ch)
270-297: 駒種別利き (14種 × 2 = 28ch)
298:     王手フラグ (1ch)
299-305: 駒落ちone-hot (7ch)
306-314: パディング (9ch)
315-324: 利き集約度 (5+5 = 10ch)
325-359: パディング (35ch)
360:     手番 (1ch)
361:     手数/512 (1ch)
```

**Python の現在のレイアウト (誤り):**
```
  0- 13: 駒配置
 14- 27: 持ち駒         ← 本来28-41
 28- 30: 千日手         ← 本来42-44
 31-255: パディング     ← 本来45-269
256-283: 駒種別利き     ← 本来270-297
  ...全部14ずれている
```

**修正方針:** `base += 14` を `base += 28` に変更する (14ch 分のゼロパディングを確保)。

### Bug 2: 打ち駒のポリシーインデックスが1ずれている

**場所:** `verify_aobazero.py` の `move_to_policy_index()`

**原因:** `cshogi.move_drop_hand_piece()` は 0-based の値 (HPAWN=0, HLANCE=1, ..., HROOK=6) を返すが、コードが `drop_piece - 1` としているため、歩の打ち駒が ch=19 (成り+dir9) にマッピングされる。

```python
# 現在 (誤り)
dir_idx = drop_piece - 1  # HPAWN=0 → dir_idx=-1 → ch=19

# 修正後
dir_idx = drop_piece       # HPAWN=0 → dir_idx=0  → ch=20
```

**検証結果:**
```
HPAWN=0, HLANCE=1, HKNIGHT=2, HSILVER=3, HGOLD=4, HBISHOP=5, HROOK=6
```

### Bug 3: 手番チャンネルが反転している

**場所:** `aobazero_features.py` の `encode_features()`

**原因:** C++ では `sideToMove == BLACK` のとき 1.0 を設定するが、Python では `turn == WHITE` のとき 1.0 を設定している。

```cpp
// C++ (正しい): 先手番のとき 1.0
if ( sideToMove == BLACK ) set_dcnn_data(stock_num, data, base, y,x);
```

```python
# Python (誤り): 後手番のとき 1.0
if turn == cshogi.WHITE:
    data[base, :, :] = 1.0

# 修正後
if turn == cshogi.BLACK:
    data[base, :, :] = 1.0
```

## 軽微な差異 (要調査)

### 影利き (kage) の減算

C++ の駒種別利き計算では、`kb_m[z][i] & 0x80` で影利き (遮蔽された利き) を検出して減算している。Python 実装ではスライディング駒が最初の駒で停止するため、通常のケースでは同等の結果になるはず。ただし、厳密な一致は保証できない。

```cpp
// C++ per-piece kiki (影利き減算あり)
kage = 0;
for (int i=0; i<kiki_m[z]; i++) if ( kb_m[z][i] & 0x80 ) kage++;
n0 -= kage;
```

利き集約度チャンネル (channels 315-324) では、C++ 側で `allkaku()` の結果をそのまま使用 (影利き減算なし、コメントアウト済み)。Python 側でも影利きを含まない値を使用しているため、ここでの差異は小さいと思われるが、`allkaku()` の内部動作を確認する必要がある。

## 座標系の整理

### cshogi
- `sq = file_index * 9 + rank_index`
- file_index: 0=1筋 ~ 8=9筋
- rank_index: 0=一段 ~ 8=九段
- `rank = sq % 9`, `file = sq // 9`

### AobaZero テンソル
- `data[ch][rank][file]` (= `data[ch][y][x]`)
- `make_z(x+1, y+1) = (y+1)*16 + (x+1)` → 上位ニブル=rank, 下位ニブル=file
- 後手番 (flip) のとき: `rank = 8 - rank`, `file = 8 - file` (180度回転)

### 駒番号の対応
| AobaZero | 番号 | cshogi | 番号 | チャンネル |
|----------|------|--------|------|-----------|
| 歩 (fu) | 1 | PAWN | 1 | 0 |
| 香 (kyo) | 2 | LANCE | 2 | 1 |
| 桂 (kei) | 3 | KNIGHT | 3 | 2 |
| 銀 (gin) | 4 | SILVER | 4 | 3 |
| **金 (kin)** | **5** | **GOLD** | **7** | **4** |
| **角 (kaku)** | **6** | **BISHOP** | **5** | **5** |
| **飛 (hi)** | **7** | **ROOK** | **6** | **6** |
| 王 (ou) | 8 | KING | 8 | 7 |
| と (to) | 9 | PROM_PAWN | 9 | 8 |
| 成香 | 10 | PROM_LANCE | 10 | 9 |
| 成桂 | 11 | PROM_KNIGHT | 11 | 10 |
| 成銀 | 12 | PROM_SILVER | 12 | 11 |
| (欠番) | 13 | - | - | - |
| **馬 (uma)** | **14** | **HORSE** | **13** | **12** |
| **龍 (ryu)** | **15** | **DRAGON** | **14** | **13** |

金・角・飛の番号が異なる点、馬・龍の番号が異なる点に注意。`_CSHOGI_TO_AOBA_PIECE` マッピングでは正しく変換済み。

## ポリシーエンコーディング

- 出力サイズ: 2187 = 27 × 9 × 9
- チャンネル 0-9: 非成り (方向 0-9)
- チャンネル 10-19: 成り (方向 0-9)
- チャンネル 20-26: 打ち (歩=20, 香=21, ..., 飛=26)
- `policy_index = ch * 81 + to_rank * 9 + to_file` (現在のプレイヤー視点)

方向定義 (`dx = from_file - to_file`, `dy = from_rank - to_rank`):
```
dir 0: dx>0, dy==0     dir 4: dx<0, dy==0
dir 1: dx>0, dy>0      dir 5: dx<0, dy<0
dir 2: dx==0, dy>0     dir 6: dx==0, dy<0
dir 3: dx<0, dy>0      dir 7: dx>0, dy<0
dir 8: 桂 (dx==1, dy==2)
dir 9: 桂 (dx==-1, dy==2)
```

## 実行方法

```bash
cd nnue-pytorch && source .venv/bin/activate
PYTHONPATH=../src:$PYTHONPATH python -m train_nnue.verify_aobazero \
    --weights ../aobazero/weights/aobazero_w4636.pt \
    --dataset ../data/accuracy_eval/test.jsonl \
    --num-positions 200 --verbose
```

## 次のステップ

1. **Bug 1-3 を修正する** (チャンネルオフセット、打ち駒インデックス、手番フラグ)
2. 修正後に精度を再評価する
3. 精度が目標に達しない場合:
   - 影利きの差異を詳細に調査する
   - w4636 が初期段階のチェックポイントである可能性を考慮し、より後の重み (w4591 等) での検証を検討する
4. 検証をパスしたら、ロードマップ2 (Expert Blending) に進む
