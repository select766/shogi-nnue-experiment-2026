#!/usr/bin/env python3
"""postprocess.py: figure_raw.svg を後処理して figure.svg を生成する"""

import re, sys
from dataclasses import dataclass
from typing import List, Tuple, Optional

# --------------------------------------------------------------------------- #
# 設定
# --------------------------------------------------------------------------- #
INPUT_SVG  = "figure_raw.svg"
OUTPUT_SVG = "figure.svg"
PHYS_W_CM  = 16.2

# --------------------------------------------------------------------------- #
# 重なり検証ヘルパー
# --------------------------------------------------------------------------- #

@dataclass
class Rect:
    x: float; y: float; w: float; h: float; label: str = ""
    def right(self): return self.x + self.w
    def bottom(self): return self.y + self.h
    def overlaps(self, other) -> bool:
        return (self.x < other.right() and self.right() > other.x and
                self.y < other.bottom() and self.bottom() > other.y)
    def contains_point(self, px, py) -> bool:
        return self.x <= px <= self.right() and self.y <= py <= self.bottom()

def parse_filled_rects(svg_content: str) -> List[Rect]:
    """stroke と fill が両方 transparent でないノード rect を返す"""
    rects = []
    for m in re.finditer(r'<rect ([^>]+)/>', svg_content):
        attrs = m.group(1)
        def ga(name):
            mm = re.search(rf'{name}="([^"]+)"', attrs)
            return mm.group(1) if mm else ""
        fill = ga('fill'); stroke = ga('stroke')
        if fill in ('transparent', 'none', '') and stroke in ('transparent', 'none', ''):
            continue  # コンテナ・背景は除外
        if fill == '#FFFFFF' and stroke in ('transparent', 'none', ''):
            continue  # 白背景も除外
        try:
            x=float(ga('x')); y=float(ga('y')); w=float(ga('width')); h=float(ga('height'))
            if w < 10 or h < 10: continue
            # コンテナグループ（面積大）は除外（路線がコンテナを横断するのは許容）
            if w * h > 50000: continue
            rects.append(Rect(x, y, w, h))
        except ValueError:
            pass
    return rects

def sample_path_points(d: str, n_samples: int = 80) -> List[Tuple[float, float]]:
    """SVG path d 属性からポイントをサンプリングする"""
    points = []
    tokens = re.findall(r'[MLCZz]|[-+]?[0-9]*\.?[0-9]+', d)
    i = 0; cx = 0; cy = 0
    while i < len(tokens):
        cmd = tokens[i]; i += 1
        if cmd in ('Z', 'z'):
            continue
        coords = []
        while i < len(tokens) and not tokens[i].isalpha():
            coords.append(float(tokens[i])); i += 1
        if cmd == 'M' and len(coords) >= 2:
            cx, cy = coords[0], coords[1]
            points.append((cx, cy))
        elif cmd == 'L' and len(coords) >= 2:
            tx, ty = coords[0], coords[1]
            for t in [k/n_samples for k in range(n_samples+1)]:
                points.append((cx + t*(tx-cx), cy + t*(ty-cy)))
            cx, cy = tx, ty
        elif cmd == 'C' and len(coords) >= 6:
            x1,y1,x2,y2,tx,ty = coords[0],coords[1],coords[2],coords[3],coords[4],coords[5]
            for t in [k/n_samples for k in range(n_samples+1)]:
                u=1-t
                px=u**3*cx+3*u**2*t*x1+3*u*t**2*x2+t**3*tx
                py=u**3*cy+3*u**2*t*y1+3*u*t**2*y2+t**3*ty
                points.append((px, py))
            cx, cy = tx, ty
    return points

def estimate_text_rect(x: float, y: float, text: str, font_size: float = 16,
                       anchor: str = "middle") -> Rect:
    char_w = sum(0.6 if ord(c) < 128 else 1.0 for c in text) * font_size * 0.9
    h = font_size * 1.3
    if anchor == "middle":
        return Rect(x - char_w/2, y - font_size, char_w, h, text)
    elif anchor == "start":
        return Rect(x, y - font_size, char_w, h, text)
    else:
        return Rect(x - char_w, y - font_size, char_w, h, text)

def check_overlaps(svg_content: str) -> List[str]:
    violations = []
    node_rects = parse_filled_rects(svg_content)

    # エッジパスがノード内部を通過していないか
    for m in re.finditer(r'<path ([^>]+)/>', svg_content):
        attrs = m.group(1)
        if 'marker-end' not in attrs: continue
        dm = re.search(r'\bd="([^"]+)"', attrs)
        if not dm: continue
        path_d = dm.group(1)
        if 'opacity:0' in attrs: continue
        pts = sample_path_points(path_d)
        for rect in node_rects:
            inner = [p for p in pts[5:-5] if rect.contains_point(*p)]
            if len(inner) > 3:
                violations.append(
                    f"OVERLAP: arrow path passes through box at "
                    f"x={rect.x:.0f} y={rect.y:.0f} w={rect.w:.0f} h={rect.h:.0f} "
                    f"(path starts near {pts[0]})"
                )
                break

    # テキストラベルがノード rect と重なっていないか
    for m in re.finditer(r'<text ([^>]+)>([^<]+)</text>', svg_content):
        attrs = m.group(1); text = m.group(2).strip()
        if not text or text in ('…',): continue
        if 'fill-N1' in attrs: continue
        xm = re.search(r'\bx="([^"]+)"', attrs)
        ym = re.search(r'\by="([^"]+)"', attrs)
        if not xm or not ym: continue
        anchor = "middle"
        if 'text-anchor:start' in attrs: anchor = "start"
        elif 'text-anchor:end' in attrs: anchor = "end"
        fs_m = re.search(r'font-size:([0-9.]+)', attrs)
        fs = float(fs_m.group(1)) if fs_m else 16.0
        tr = estimate_text_rect(float(xm.group(1)), float(ym.group(1)), text, fs, anchor)
        for rect in node_rects:
            if tr.overlaps(rect):
                violations.append(
                    f"OVERLAP: label '{text}' at ({tr.x:.0f},{tr.y:.0f}) "
                    f"overlaps box at x={rect.x:.0f} y={rect.y:.0f} "
                    f"w={rect.w:.0f} h={rect.h:.0f}"
                )
                break

    return violations

# --------------------------------------------------------------------------- #
# メイン処理
# --------------------------------------------------------------------------- #

with open(INPUT_SVG, 'r', encoding='utf-8') as f:
    content = f.read()

# 1. 外側 SVG の viewBox 幅・高さを取得
m = re.search(r'<svg\b[^>]*viewBox="0 0 ([0-9.]+) ([0-9.]+)"', content)
if not m:
    print("ERROR: outer viewBox not found", file=sys.stderr)
    sys.exit(1)
vb_w = float(m.group(1))
vb_h = float(m.group(2))
phys_h_cm = vb_h * PHYS_W_CM / vb_w
print(f"viewBox: {vb_w:.0f} x {vb_h:.0f}  →  physical: {PHYS_W_CM}cm x {phys_h_cm:.2f}cm")

# 2. 外側 SVG に物理サイズを付与
content = re.sub(
    r'(<svg\b)([^>]*?viewBox="0 0)',
    f'\\1 width="{PHYS_W_CM}cm" height="{phys_h_cm:.2f}cm"\\2',
    content, count=1
)

# 3. 内側 SVG (d2-svg) の width/height を viewBox に合わせる
content = re.sub(
    r'(<svg\b[^>]*d2-svg[^>]*?)width="[^"]+" height="[^"]+"',
    f'\\1width="{vb_w:.0f}" height="{vb_h:.0f}"',
    content
)

# 4. text-italic → text （エッジラベルのイタリック除去）
content = content.replace('class="text-italic"', 'class="text"')
content = content.replace('"text-italic ', '"text ')

# 5. transparent → none  (Inkscape 互換)
content = content.replace('fill="transparent"', 'fill="none"')
content = content.replace('stroke="transparent"', 'stroke="none"')

# 6. animated エッジを静的 dasharray に変換
# animated クラスが付いた path の stroke-dasharray を設定し animation を削除
# d2 animated パスは CSS に animation が付く → 後処理で dasharray を直接指定
# (現在このフローでは animated エッジなし)

# 7. カスタム SVG 要素を追加
#    コーディネートはすべて figure_raw.svg の viewBox 座標系 (0 0 834 1780)
#    に基づく

# SVG の中で使う arrowhead マーカー定義
custom_defs = """\
<defs>
  <marker id="arrow-fwd" markerWidth="10" markerHeight="12"
          refX="9" refY="6" viewBox="0 0 10 12"
          orient="auto" markerUnits="userSpaceOnUse">
    <polygon points="0,0 10,6 0,12" fill="#0D32B2"/>
  </marker>
  <marker id="arrow-bwd" markerWidth="10" markerHeight="12"
          refX="9" refY="6" viewBox="0 0 10 12"
          orient="auto" markerUnits="userSpaceOnUse">
    <polygon points="0,0 10,6 0,12" fill="#D32F2F"/>
  </marker>
</defs>
"""

# ランダム局面 B (right=473, center-y=113) → NNUE推論 (right=514, center-y=1463)
# 右に抜けて下に降り NNUE推論の右へ入る
path_random_b_to_nnue = (
    '<path d="M 473 113 L 820 113 L 820 1463 L 514 1463" '
    'stroke="#0D32B2" fill="none" stroke-width="2" '
    'marker-end="url(#arrow-fwd)"/>'
)
# ラベル: 第1水平セグメント上 (training_data の右外側)
label_b_to_nnue = (
    '<text x="648" y="100" style="font-size:16;text-anchor:middle" '
    'class="text" fill="#555555">局面 B (HalfKP特徴量)</text>'
)

# 損失計算 (left=245, center-y=1641) → DNN Adapter (left=66, center-y=652)
# 左に抜けて上昇し Adapter の左へ入る
path_loss_to_dnn_ad = (
    '<path d="M 245 1641 L 15 1641 L 15 652 L 66 652" '
    'stroke="#D32F2F" fill="none" stroke-width="2" stroke-dasharray="8 4" '
    'marker-end="url(#arrow-bwd)"/>'
)
# ラベル: DNN chain と NNUE Experts の間の空白帯 (x=17, y=848-864)
label_backward_1 = (
    '<text x="17" y="848" style="font-size:13;text-anchor:start" '
    'class="text" fill="#D32F2F">誤差逆伝播</text>'
)
label_backward_2 = (
    '<text x="17" y="864" style="font-size:13;text-anchor:start" '
    'class="text" fill="#D32F2F">(Adapter + NNUE Experts を更新)</text>'
)

# 損失計算 (right=432, center-y=1641) → NNUE Experts (right=814, center-y=679)
# 右に抜けて上昇し Experts の右へ入る
path_loss_to_experts = (
    '<path d="M 432 1641 L 825 1641 L 825 679 L 814 679" '
    'stroke="#D32F2F" fill="none" stroke-width="2" stroke-dasharray="8 4" '
    'marker-end="url(#arrow-bwd)"/>'
)

custom_elements = "\n".join([
    "<!-- === postprocess.py 追加要素 === -->",
    custom_defs,
    "<!-- ランダム局面 B → NNUE推論 -->",
    path_random_b_to_nnue,
    label_b_to_nnue,
    "<!-- 逆伝播: 損失計算 → DNN Adapter -->",
    path_loss_to_dnn_ad,
    label_backward_1,
    label_backward_2,
    "<!-- 逆伝播: 損失計算 → NNUE Experts -->",
    path_loss_to_experts,
])

# 内側 </svg> の直前に挿入
# SVG の末尾: </mask></svg></svg>\n
# 2番目の </svg> が内側 SVG 終了タグ
close_positions = [m.start() for m in re.finditer(r'</svg>', content)]
if len(close_positions) < 2:
    print("ERROR: expected at least 2 </svg> tags", file=sys.stderr)
    sys.exit(1)
inner_close_pos = close_positions[-2]  # second-to-last = inner SVG close
content = content[:inner_close_pos] + custom_elements + "\n" + content[inner_close_pos:]

# 8. 出力
with open(OUTPUT_SVG, 'w', encoding='utf-8') as f:
    f.write(content)

print(f"Written: {OUTPUT_SVG}")
print(f"Physical size: {PHYS_W_CM}cm x {phys_h_cm:.2f}cm")

# 9. 重なり検証
violations = check_overlaps(content)
if violations:
    print("\n=== OVERLAP VIOLATIONS DETECTED ===")
    for v in violations:
        print(" ", v)
    print(f"Total: {len(violations)} violation(s)")
    sys.exit(1)
else:
    print("=== No overlaps detected ===")
