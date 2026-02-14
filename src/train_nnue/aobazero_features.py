"""
AobaZero 362-channel input feature encoder.

Encodes a shogi board position into a (362, 9, 9) tensor matching AobaZero's
input format. Uses cshogi for board representation.

Channel layout (362 total):
  0-13:   Piece positions (two-hot: own=+1, opponent=-1) [14 ch]
  14-27:  [Unused, zeros] (reserved for non-TWO_HOT mode) [14 ch]
  28-41:  Hand pieces (normalized counts, 7 types * 2 sides) [14 ch]
  42-44:  Repetition count (0/1/2+) [3 ch]
  45-269: Padding for past 5 time steps (all zeros) [225 ch]
  270-297: Per-piece-type attack presence (14 types * 2 sides) [28 ch]
  298:    In check [1 ch]
  299-305: Handicap one-hot (always channel 0 for even games) [7 ch]
  306-314: Padding [9 ch]
  315-324: Aggregate attack degree (5 own + 5 opponent levels) [10 ch]
  325-359: Padding [35 ch]
  360:    Side to move (1.0 if sente/BLACK) [1 ch]
  361:    Ply / 512 [1 ch]

Coordinate system:
  AobaZero tensor: data[ch][rank][file]  (rank=row, file=column)
  cshogi square:   sq = file_index * 9 + rank_index
  Mapping:         rank = sq % 9,  file = sq // 9

Note: The board is always viewed from the current player's perspective.
When it's gote's turn, the board is rotated 180 degrees.

Reference: aobazero/repo/learn/yss_dcnn.cpp lines 1123-1558
"""

import numpy as np
import cshogi


def _sq_to_rank_file(sq: int):
    """Convert cshogi square index to (rank, file) for AobaZero tensor."""
    return sq % 9, sq // 9


# Piece movement patterns for attack computation
# Each entry: list of (d_rank, d_file, is_sliding) tuples
# Sente moves forward = decreasing rank (toward rank 0 = 一段)
_PIECE_MOVES = {
    cshogi.PAWN: [(-1, 0, False)],
    cshogi.LANCE: [(-1, 0, True)],
    cshogi.KNIGHT: [(-2, -1, False), (-2, 1, False)],
    cshogi.SILVER: [(-1, -1, False), (-1, 0, False), (-1, 1, False),
                    (1, -1, False), (1, 1, False)],
    cshogi.GOLD: [(-1, -1, False), (-1, 0, False), (-1, 1, False),
                  (0, -1, False), (0, 1, False), (1, 0, False)],
    cshogi.BISHOP: [(-1, -1, True), (-1, 1, True), (1, -1, True), (1, 1, True)],
    cshogi.ROOK: [(-1, 0, True), (1, 0, True), (0, -1, True), (0, 1, True)],
    cshogi.KING: [(-1, -1, False), (-1, 0, False), (-1, 1, False),
                  (0, -1, False), (0, 1, False),
                  (1, -1, False), (1, 0, False), (1, 1, False)],
    # Promoted pieces
    cshogi.PROM_PAWN: None,    # Same as GOLD
    cshogi.PROM_LANCE: None,   # Same as GOLD
    cshogi.PROM_KNIGHT: None,  # Same as GOLD
    cshogi.PROM_SILVER: None,  # Same as GOLD
    13: [(-1, -1, True), (-1, 1, True), (1, -1, True), (1, 1, True),  # HORSE
         (-1, 0, False), (1, 0, False), (0, -1, False), (0, 1, False)],
    14: [(-1, 0, True), (1, 0, True), (0, -1, True), (0, 1, True),  # DRAGON
         (-1, -1, False), (-1, 1, False), (1, -1, False), (1, 1, False)],
}
for pt in [cshogi.PROM_PAWN, cshogi.PROM_LANCE, cshogi.PROM_KNIGHT, cshogi.PROM_SILVER]:
    _PIECE_MOVES[pt] = _PIECE_MOVES[cshogi.GOLD]

# Maps cshogi piece_type to AobaZero channel index (0-13)
_CSHOGI_TO_AOBA_PIECE = {
    cshogi.PAWN: 0, cshogi.LANCE: 1, cshogi.KNIGHT: 2, cshogi.SILVER: 3,
    cshogi.GOLD: 4, cshogi.BISHOP: 5, cshogi.ROOK: 6, cshogi.KING: 7,
    cshogi.PROM_PAWN: 8, 10: 9, 11: 10, 12: 11, 13: 12, 14: 13,
}

# AobaZero kiki piece types (1..15, skipping 13) -> cshogi piece types
_AOBA_KIKI_PIECE_TYPES = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15]
_AOBA_KIKI_TO_CSHOGI = {
    1: cshogi.PAWN, 2: cshogi.LANCE, 3: cshogi.KNIGHT, 4: cshogi.SILVER,
    5: cshogi.GOLD, 6: cshogi.BISHOP, 7: cshogi.ROOK, 8: cshogi.KING,
    9: cshogi.PROM_PAWN, 10: 10, 11: 11, 12: 12, 14: 13, 15: 14,
}

# Hand piece divisors for normalization
_HAND_DIVISORS = [18, 4, 4, 4, 4, 2, 2]  # pawn, lance, knight, silver, gold, bishop, rook


def _compute_attacks(board: cshogi.Board):
    """Compute attack maps for each piece type and aggregate attacks.

    All coordinates use AobaZero convention: (rank, file).

    Returns:
        piece_type_attacks: dict mapping (cshogi_piece_type, color) -> 9x9 bool array
        sente_total: 9x9 int array of total sente attacks per square
        gote_total: 9x9 int array of total gote attacks per square
    """
    # Build board array: (rank, file) -> (piece_type, color)
    board_array = {}
    for sq in range(81):
        piece = board.piece(sq)
        if piece != 0:
            rank, file = _sq_to_rank_file(sq)
            if piece >= 17:
                color = cshogi.WHITE
                pt = piece - 16
            else:
                color = cshogi.BLACK
                pt = piece
            board_array[(rank, file)] = (pt, color)

    sente_total = np.zeros((9, 9), dtype=np.int32)
    gote_total = np.zeros((9, 9), dtype=np.int32)
    piece_type_attacks = {}

    for (p_rank, p_file), (pt, color) in board_array.items():
        moves = _PIECE_MOVES.get(pt)
        if moves is None:
            continue

        # Sente moves forward (decreasing rank), gote moves backward (increasing rank)
        dir_mult = 1 if color == cshogi.BLACK else -1

        attacked = set()
        for d_rank, d_file, is_sliding in moves:
            dr = d_rank * dir_mult
            df = d_file * dir_mult
            nr, nf = p_rank + dr, p_file + df
            if is_sliding:
                while 0 <= nr < 9 and 0 <= nf < 9:
                    attacked.add((nr, nf))
                    if (nr, nf) in board_array:
                        break
                    nr += dr
                    nf += df
            else:
                if 0 <= nr < 9 and 0 <= nf < 9:
                    attacked.add((nr, nf))

        key = (pt, color)
        if key not in piece_type_attacks:
            piece_type_attacks[key] = np.zeros((9, 9), dtype=np.bool_)
        for (ar, af) in attacked:
            piece_type_attacks[key][ar, af] = True

        target = sente_total if color == cshogi.BLACK else gote_total
        for (ar, af) in attacked:
            target[ar, af] += 1

    return piece_type_attacks, sente_total, gote_total


def encode_features(board: cshogi.Board, ply: int = 0) -> np.ndarray:
    """Encode a board position into AobaZero's 362-channel input tensor.

    The board is always viewed from the current player's perspective.

    Args:
        board: cshogi.Board with the position to encode
        ply: Move number (used for ply/512 feature)

    Returns:
        numpy array of shape (362, 9, 9), dtype float32
    """
    data = np.zeros((362, 9, 9), dtype=np.float32)
    turn = board.turn
    flip = (turn == cshogi.WHITE)
    base = 0

    # === 1. Piece positions (14 channels, two-hot) ===
    for sq in range(81):
        piece = board.piece(sq)
        if piece == 0:
            continue

        rank, file = _sq_to_rank_file(sq)

        if piece >= 17:
            color = cshogi.WHITE
            pt = piece - 16
        else:
            color = cshogi.BLACK
            pt = piece

        ch = _CSHOGI_TO_AOBA_PIECE.get(pt)
        if ch is None:
            continue

        row, col = rank, file
        if flip:
            row = 8 - row
            col = 8 - col

        if (color == cshogi.BLACK) != flip:
            data[base + ch, row, col] = 1.0
        else:
            data[base + ch, row, col] = -1.0

    base += 28  # base=28 (C++ reserves 28 channels even with TWO_HOT using only 14)

    # === 2. Hand pieces (14 channels) ===
    pih = board.pieces_in_hand
    sente_hand = pih[cshogi.BLACK]
    gote_hand = pih[cshogi.WHITE]

    for i in range(7):
        own_count = sente_hand[i] if not flip else gote_hand[i]
        opp_count = gote_hand[i] if not flip else sente_hand[i]
        f_own = float(own_count) / _HAND_DIVISORS[i]
        f_opp = float(opp_count) / _HAND_DIVISORS[i]
        data[base + i, :, :] = f_own
        data[base + 7 + i, :, :] = f_opp

    base += 14  # base=42

    # === 3. Repetition (3 channels) ===
    base += 3  # base=45

    # === 4. Padding for past time steps (225 channels) ===
    base += 225  # base=270

    # === 5. Per-piece-type attack presence (28 channels) ===
    piece_type_attacks, sente_total, gote_total = _compute_attacks(board)

    for ki, aoba_pt in enumerate(_AOBA_KIKI_PIECE_TYPES):
        cshogi_pt = _AOBA_KIKI_TO_CSHOGI[aoba_pt]

        sente_atk = piece_type_attacks.get((cshogi_pt, cshogi.BLACK), None)
        gote_atk = piece_type_attacks.get((cshogi_pt, cshogi.WHITE), None)

        if flip:
            sente_atk, gote_atk = gote_atk, sente_atk

        for atk, ch_off in [(sente_atk, 0), (gote_atk, 1)]:
            if atk is not None:
                for r in range(9):
                    for f in range(9):
                        if atk[r, f]:
                            rr, ff = (8 - r, 8 - f) if flip else (r, f)
                            data[base + ch_off, rr, ff] = 1.0
        base += 2
    # base=298

    # === 6. In check (1 channel) ===
    if board.is_check():
        data[base, :, :] = 1.0
    base += 1  # base=299

    # === 7. Handicap (7 channels) ===
    data[base + 0, :, :] = 1.0
    base += 7  # base=306

    # === 8. Padding (9 channels) ===
    base += 9  # base=315

    # === 9. Aggregate attack degree (10 channels) ===
    M = 4
    own_atk = sente_total if not flip else gote_total
    opp_atk = gote_total if not flip else sente_total

    for r in range(9):
        for f in range(9):
            rr, ff = (8 - r, 8 - f) if flip else (r, f)

            n0 = min(int(own_atk[r, f]), M)
            n1 = min(int(opp_atk[r, f]), M)

            if n0 == 0:
                data[base + 0, rr, ff] = 1.0
            for i in range(n0):
                data[base + i + 1, rr, ff] = 1.0

            if n1 == 0:
                data[base + 0 + M + 1, rr, ff] = 1.0
            for i in range(n1):
                data[base + i + 1 + M + 1, rr, ff] = 1.0

    base += 10  # base=325

    # === 10. Padding (35 channels) ===
    base += 35  # base=360

    # === 11. Side to move + ply (2 channels) ===
    if turn == cshogi.BLACK:
        data[base, :, :] = 1.0
    data[base + 1, :, :] = float(ply) / 512.0
    base += 2  # base=362

    assert base == 362, f"Channel count mismatch: {base} != 362"
    return data
