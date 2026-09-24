import math
from functools import lru_cache

import cv2
import numpy as np
import streamlit as st
from PIL import Image, ImageDraw, ImageFont

# -----------------------------
# 基本設定
# -----------------------------
st.set_page_config(page_title="オセロせんせい", page_icon="⚫", layout="centered")

EMPTY = 0
BLACK = 1
WHITE = -1
DIRS = [
    (-1, -1), (-1, 0), (-1, 1),
    (0, -1),           (0, 1),
    (1, -1),  (1, 0),  (1, 1),
]
COLS = "ABCDEFGH"

# 角を強く評価し、角のすぐ隣（X/C打ち）は基本的に避ける。
POSITION_WEIGHTS = np.array([
    [120, -25, 20,  5,  5, 20, -25, 120],
    [-25, -45, -5, -5, -5, -5, -45, -25],
    [ 20,  -5, 15,  3,  3, 15,  -5,  20],
    [  5,  -5,  3,  3,  3,  3,  -5,   5],
    [  5,  -5,  3,  3,  3,  3,  -5,   5],
    [ 20,  -5, 15,  3,  3, 15,  -5,  20],
    [-25, -45, -5, -5, -5, -5, -45, -25],
    [120, -25, 20,  5,  5, 20, -25, 120],
], dtype=np.int16)

CORNERS = {(0, 0), (0, 7), (7, 0), (7, 7)}
X_SQUARES = {(1, 1), (1, 6), (6, 1), (6, 6)}
C_SQUARES = {(0, 1), (1, 0), (0, 6), (1, 7), (6, 0), (7, 1), (6, 7), (7, 6)}

st.markdown("""
<style>
:root {
    --bg: #090909;
    --surface: #141414;
    --surface-2: #1c1c1c;
    --surface-3: #242424;
    --border: #353535;
    --text: #ffffff;
    --muted: #bdbdbd;
    --button: #ffffff;
    --button-text: #111111;
    --green: #32c26b;
    --green-soft: rgba(50,194,107,.14);
    --green-border: rgba(50,194,107,.48);
    --shadow: 0 14px 34px rgba(0,0,0,.36);
}

html, body, [class*="css"] { color: var(--text); }

.stApp {
    color: var(--text);
    background:
        radial-gradient(circle at 50% -10%, rgba(50,194,107,.08), transparent 30%),
        var(--bg);
}
section.main > div { background: transparent; }

.block-container {
    max-width: 760px;
    padding-top: 4.25rem !important;
    padding-bottom: 4rem;
}

.hero-panel {
    position: relative;
    overflow: hidden;
    border: 1px solid var(--border);
    border-radius: 24px;
    padding: 1.2rem 1rem 1.15rem;
    margin-bottom: 1.2rem;
    background: linear-gradient(180deg, #181818 0%, #101010 100%);
    box-shadow: var(--shadow);
}
.hero-panel::after {
    content: "";
    position: absolute;
    left: 0;
    right: 0;
    bottom: 0;
    height: 3px;
    background: var(--green);
}
.hero-kicker {
    text-align: center;
    font-size: .72rem;
    font-weight: 800;
    letter-spacing: .18em;
    color: var(--green);
    margin-bottom: .45rem;
}
.big-title {
    text-align:center;
    font-size:2.08rem;
    font-weight:900;
    line-height:1.2;
    margin:.10rem 0 .35rem;
    color: var(--text);
}
.sub {
    text-align:center;
    color: var(--muted);
    margin: 0 auto 1rem;
    line-height:1.6;
    max-width: 30rem;
}
.hero-chips {
    display:flex;
    gap:.5rem;
    flex-wrap:wrap;
    justify-content:center;
}
.hero-chip {
    background: #202020;
    border: 1px solid #373737;
    color: #ffffff;
    padding: .42rem .7rem;
    border-radius: 999px;
    font-size: .88rem;
    font-weight: 700;
}

.small-note {font-size:.88rem; color:var(--muted);}
h3 {
    color: var(--text) !important;
    font-size: 1.65rem !important;
    font-weight: 800 !important;
    margin-bottom: .25rem !important;
}

.result-card,
.lesson-head,
[data-testid="stCameraInput"],
[data-testid="stFileUploader"],
[data-testid="stExpander"] {
    border:1px solid var(--border) !important;
    border-radius:20px !important;
    background: var(--surface) !important;
    box-shadow: var(--shadow);
}
.result-card { padding:16px; margin:10px 0; }
.result-main {font-size:1.45rem; font-weight:900; text-align:center; color:var(--green);}
.kid-text {font-size:1.15rem; line-height:1.7; text-align:center; color:var(--text);}

/* ボタンは白地＋黒文字で統一。緑は枠・フォーカス・状態表示に限定。 */
div.stButton > button,
[data-testid="baseButton-secondary"],
[data-testid="baseButton-primary"] {
    border-radius:16px !important;
    min-height:54px;
    font-weight:850;
    letter-spacing:.01em;
    border: 1px solid #ffffff !important;
    background: #ffffff !important;
    color: var(--button-text) !important;
    box-shadow: 0 8px 20px rgba(0,0,0,.26);
}
div.stButton > button:hover,
[data-testid="baseButton-secondary"]:hover,
[data-testid="baseButton-primary"]:hover {
    background: #f1f1f1 !important;
    border-color: var(--green) !important;
}
div.stButton > button:focus,
[data-testid="baseButton-secondary"]:focus,
[data-testid="baseButton-primary"]:focus {
    box-shadow: 0 0 0 3px rgba(50,194,107,.28), 0 8px 20px rgba(0,0,0,.26) !important;
}

[data-testid="stCameraInput"] {margin-top:.35rem; padding: .85rem;}
[data-testid="stFileUploader"] {margin-top:.25rem; padding: .85rem;}
[data-testid="stExpander"] { padding: .35rem .5rem; }
[data-testid="stExpander"] summary {
    background: #181818 !important;
    border-radius: 16px !important;
    color: var(--text) !important;
}

/* Streamlit標準の文字色を明示して見切れ・同化を防ぐ */
.stApp [data-testid="stMarkdownContainer"],
.stApp [data-testid="stMarkdownContainer"] p,
.stApp [data-testid="stMarkdownContainer"] li,
.stApp [data-testid="stWidgetLabel"],
.stApp [data-testid="stWidgetLabel"] p,
.stApp label,
.stApp small,
[data-testid="stExpander"] summary,
[data-testid="stExpander"] summary p,
[data-testid="stExpander"] summary span {
    color: var(--text) !important;
}
[data-testid="stCaptionContainer"],
[data-testid="stCaptionContainer"] p { color: var(--muted) !important; }

/* 白ボタン内は必ず黒文字 */
div.stButton > button,
div.stButton > button p,
div.stButton > button span,
[data-testid="baseButton-secondary"],
[data-testid="baseButton-secondary"] p,
[data-testid="baseButton-secondary"] span,
[data-testid="baseButton-primary"],
[data-testid="baseButton-primary"] p,
[data-testid="baseButton-primary"] span,
[data-testid="stCameraInput"] button,
[data-testid="stCameraInput"] button p,
[data-testid="stFileUploader"] button,
[data-testid="stFileUploader"] button p {
    color: var(--button-text) !important;
}

[data-testid="stCameraInput"] p,
[data-testid="stCameraInput"] span,
[data-testid="stFileUploader"] p,
[data-testid="stFileUploader"] span { color: var(--text) !important; }

/* selectbox / radio / input */
[data-baseweb="select"] > div,
[data-testid="stRadio"] > div,
[data-testid="stTextInput"] input {
    background: var(--surface-2) !important;
    color: var(--text) !important;
    border: 1px solid #3a3a3a !important;
    border-radius: 14px !important;
}
[data-baseweb="select"] span,
[data-testid="stRadio"] label,
[data-testid="stRadio"] p,
[data-testid="stTextInput"] input { color: var(--text) !important; }
[role="listbox"], [role="option"] {
    background: #1c1c1c !important;
    color: #ffffff !important;
}
[role="option"]:hover { background: #292929 !important; }

.mode-card,
.think-card,
.challenge-card {
    border: 1px solid var(--border);
    border-radius: 20px;
    padding: 1rem;
    margin: .65rem 0;
    background: linear-gradient(180deg, #171717 0%, #111111 100%);
    box-shadow: var(--shadow);
}
.mode-title {
    color: var(--green);
    font-size: 1.15rem;
    font-weight: 900;
    margin-bottom: .3rem;
}
.mode-text { color: var(--muted); line-height: 1.6; }
.turn-badge {
    display:inline-block;
    border-radius:999px;
    padding:.35rem .65rem;
    background:var(--green-soft);
    border:1px solid var(--green-border);
    color:#bff2d2;
    font-weight:800;
    margin:.2rem 0 .65rem;
}
.feedback-good,
.feedback-bad,
.feedback-neutral {
    border-radius:18px;
    padding: .9rem 1rem;
    margin:.7rem 0;
    line-height:1.65;
    border:1px solid var(--border);
    color: var(--text);
}
.feedback-good {background:rgba(50,194,107,.11); border-color:var(--green-border);}
.feedback-bad {background:#1b1b1b; border-color:#555;}
.feedback-neutral {background:#181818; border-color:#3f3f3f;}
.feedback-good b {color:#83e6aa;}
.feedback-bad b {color:#ffffff;}
.feedback-neutral b {color:#ffffff;}
.level-meter {
    display:flex;
    gap:.35rem;
    justify-content:center;
    margin:.5rem 0 .9rem;
}
.level-dot {
    width:28px;
    height:8px;
    border-radius:99px;
    background:#343434;
}
.level-dot.on { background:var(--green); }

[data-testid="stAlert"] {
    background: #161616 !important;
    border: 1px solid #3b3b3b !important;
    color: var(--text) !important;
}
[data-testid="stAlert"] * { color: var(--text) !important; }
[data-testid="stProgress"] p,
[data-testid="stProgress"] span { color: var(--text) !important; }

.lesson-head { padding:16px 16px 12px; margin:.4rem 0 1rem; }
.lesson-stage {font-size:.9rem; color:var(--green); font-weight:800; margin-bottom:.25rem;}
.lesson-title {font-size:1.45rem; font-weight:900; line-height:1.35; color:var(--text);}
.lesson-goal {font-size:1.02rem; line-height:1.65; margin-top:.65rem; color:var(--text);}
.lesson-point {
    border-left:5px solid var(--green);
    padding:.8rem .95rem;
    margin:.8rem 0;
    border-radius:14px;
    background: var(--green-soft);
    font-size:1.08rem;
    line-height:1.7;
    color:var(--text);
}
.lesson-seq {
    font-size:1.12rem;
    font-weight:900;
    text-align:center;
    padding:.8rem;
    border-radius:14px;
    background: #202020;
    border: 1px solid #393939;
    border-bottom: 3px solid var(--green);
    margin:.65rem 0;
    color:var(--text);
}

hr { border-color: #323232; }
[data-testid="stImage"] img {
    border-radius: 18px;
    border: 1px solid #383838;
    box-shadow: var(--shadow);
}

@media (max-width: 768px) {
    .block-container {
        padding-top: calc(5.75rem + env(safe-area-inset-top)) !important;
        padding-left: 1rem !important;
        padding-right: 1rem !important;
    }
    .hero-panel { padding: 1rem .85rem 1rem; border-radius: 22px; }
    .big-title { font-size:1.82rem; margin-top:.2rem; }
    .sub { font-size:.98rem; margin-bottom: .9rem; }
    .hero-chip { font-size: .8rem; padding: .38rem .6rem; }
    div.stButton > button { min-height:56px; font-size:1.03rem; }
    h3 { font-size: 1.45rem !important; }
}
</style>
""", unsafe_allow_html=True)


# -----------------------------
# オセロロジック
# -----------------------------
def initial_board():
    b = np.zeros((8, 8), dtype=np.int8)
    b[3, 3] = WHITE
    b[3, 4] = BLACK
    b[4, 3] = BLACK
    b[4, 4] = WHITE
    return b


def flips_for_move(board, player, r, c):
    if board[r, c] != EMPTY:
        return []
    flips = []
    opp = -player
    for dr, dc in DIRS:
        rr, cc = r + dr, c + dc
        line = []
        while 0 <= rr < 8 and 0 <= cc < 8 and board[rr, cc] == opp:
            line.append((rr, cc))
            rr += dr
            cc += dc
        if line and 0 <= rr < 8 and 0 <= cc < 8 and board[rr, cc] == player:
            flips.extend(line)
    return flips


def legal_moves(board, player):
    moves = {}
    for r in range(8):
        for c in range(8):
            f = flips_for_move(board, player, r, c)
            if f:
                moves[(r, c)] = f
    return moves


def apply_move(board, player, move, flips=None):
    r, c = move
    if flips is None:
        flips = flips_for_move(board, player, r, c)
    new = board.copy()
    new[r, c] = player
    for rr, cc in flips:
        new[rr, cc] = player
    return new


def is_frontier(board, r, c):
    if board[r, c] == EMPTY:
        return False
    for dr, dc in DIRS:
        rr, cc = r + dr, c + dc
        if 0 <= rr < 8 and 0 <= cc < 8 and board[rr, cc] == EMPTY:
            return True
    return False


def corner_owner_score(board, player):
    return sum(1 if board[r, c] == player else -1 if board[r, c] == -player else 0 for r, c in CORNERS)


def evaluate(board, player):
    """player から見た盤面評価。"""
    empties = int(np.sum(board == EMPTY))
    my_moves = len(legal_moves(board, player))
    opp_moves = len(legal_moves(board, -player))

    # 終局
    if my_moves == 0 and opp_moves == 0:
        diff = int(np.sum(board == player) - np.sum(board == -player))
        return 100000 + diff * 1000 if diff > 0 else -100000 + diff * 1000 if diff < 0 else 0

    positional = int(np.sum(POSITION_WEIGHTS * (board == player)) - np.sum(POSITION_WEIGHTS * (board == -player)))
    mobility = my_moves - opp_moves
    corners = corner_owner_score(board, player)

    my_frontier = 0
    opp_frontier = 0
    for r in range(8):
        for c in range(8):
            if is_frontier(board, r, c):
                if board[r, c] == player:
                    my_frontier += 1
                elif board[r, c] == -player:
                    opp_frontier += 1
    frontier = opp_frontier - my_frontier  # 自分の境界石が少ないほどよい

    disc_diff = int(np.sum(board == player) - np.sum(board == -player))

    if empties > 40:
        return positional * 2.0 + mobility * 12 + corners * 250 + frontier * 3 - disc_diff * 1.5
    elif empties > 16:
        return positional * 2.2 + mobility * 10 + corners * 300 + frontier * 2 + disc_diff * 0.5
    else:
        return positional * 1.3 + mobility * 5 + corners * 350 + frontier + disc_diff * 8


def board_key(board):
    return tuple(int(x) for x in board.flatten())


def negamax(board, player, depth, alpha, beta, root_player):
    moves = legal_moves(board, player)
    opp_moves = legal_moves(board, -player)
    if depth <= 0 or (not moves and not opp_moves):
        return evaluate(board, root_player)

    if not moves:
        return negamax(board, -player, depth - 1, alpha, beta, root_player)

    maximizing = (player == root_player)
    ordered = sorted(moves.items(), key=lambda kv: POSITION_WEIGHTS[kv[0][0], kv[0][1]], reverse=maximizing)

    if maximizing:
        value = -math.inf
        for move, flips in ordered:
            child = apply_move(board, player, move, flips)
            value = max(value, negamax(child, -player, depth - 1, alpha, beta, root_player))
            alpha = max(alpha, value)
            if alpha >= beta:
                break
        return value
    else:
        value = math.inf
        for move, flips in ordered:
            child = apply_move(board, player, move, flips)
            value = min(value, negamax(child, -player, depth - 1, alpha, beta, root_player))
            beta = min(beta, value)
            if alpha >= beta:
                break
        return value


def best_move(board, player):
    moves = legal_moves(board, player)
    if not moves:
        return None, None, None

    empties = int(np.sum(board == EMPTY))
    # スマホ利用時の待ち時間を抑える。
    if empties > 44:
        depth = 4
    elif empties > 24:
        depth = 5
    elif empties > 12:
        depth = 6
    else:
        depth = min(empties + 2, 10)

    scored = []
    for move, flips in moves.items():
        child = apply_move(board, player, move, flips)
        score = negamax(child, -player, depth - 1, -math.inf, math.inf, player)
        scored.append((score, move, flips, child))

    scored.sort(key=lambda x: x[0], reverse=True)
    score, move, flips, child = scored[0]
    return move, flips, {"score": score, "after": child, "all": scored, "depth": depth}


def best_move_training(board, player):
    """思考実験・理解度テスト用の高速探索。スマホで繰り返し使える待ち時間を優先する。"""
    moves = legal_moves(board, player)
    if not moves:
        return None, None, None

    empties = int(np.sum(board == EMPTY))
    if empties > 40:
        depth = 3
    elif empties > 24:
        depth = 3
    elif empties > 12:
        depth = 4
    else:
        depth = min(empties + 1, 6)

    scored = []
    for move, flips in moves.items():
        child = apply_move(board, player, move, flips)
        score = negamax(child, -player, depth - 1, -math.inf, math.inf, player)
        scored.append((score, move, flips, child))
    scored.sort(key=lambda x: x[0], reverse=True)
    score, move, flips, child = scored[0]
    return move, flips, {"score": score, "after": child, "all": scored, "depth": depth}


def coord(move):
    r, c = move
    return f"{COLS[c]}{r + 1}"


def corner_adjacent_to_empty_corner(move, board):
    r, c = move
    maps = {
        (1, 1): (0, 0), (0, 1): (0, 0), (1, 0): (0, 0),
        (1, 6): (0, 7), (0, 6): (0, 7), (1, 7): (0, 7),
        (6, 1): (7, 0), (6, 0): (7, 0), (7, 1): (7, 0),
        (6, 6): (7, 7), (6, 7): (7, 7), (7, 6): (7, 7),
    }
    corner = maps.get((r, c))
    return corner is not None and board[corner] == EMPTY


def explain_move(board, player, move, flips, meta):
    after = meta["after"]
    opp_before = len(legal_moves(board, -player))
    opp_after = len(legal_moves(after, -player))
    my_next = len(legal_moves(after, player))
    empties = int(np.sum(board == EMPTY))

    # 6歳向け：理由を1つに絞る。
    if move in CORNERS:
        return "かどが とれるよ！ かどの石は、もう ひっくり返されないよ。"

    # 相手に角を許すかを確認
    opp_corners = [m for m in legal_moves(after, -player) if m in CORNERS]
    if not opp_corners and opp_after <= max(1, opp_before - 2):
        return f"ここにおくと、あいてが おけるところが {opp_after}こに へるよ。"

    if move in X_SQUARES or move in C_SQUARES:
        if not corner_adjacent_to_empty_corner(move, board):
            return "このばしょなら、かどの近くでも あぶなくなりにくいよ。"

    if empties <= 14:
        gain = len(flips)
        return f"さいごに近いから、先までよんで ここをえらんだよ。{gain}こ ひっくり返せるよ。"

    if len(flips) <= 2:
        return "いまは たくさん取らずに、つぎに うごきやすくする手だよ。"

    if opp_after <= 3:
        return "あいてが おけるばしょを少なくして、こまらせる手だよ。"

    return f"この手なら {len(flips)}こ かえしながら、つぎのチャンスものこせるよ。"


# -----------------------------
# 画像認識
# -----------------------------
def order_points(pts):
    pts = np.asarray(pts, dtype=np.float32)
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).reshape(-1)
    return np.array([
        pts[np.argmin(s)],
        pts[np.argmin(d)],
        pts[np.argmax(s)],
        pts[np.argmax(d)],
    ], dtype=np.float32)


def detect_board_quad(img_bgr):
    h, w = img_bgr.shape[:2]
    scale = 1000.0 / max(h, w) if max(h, w) > 1000 else 1.0
    small = cv2.resize(img_bgr, None, fx=scale, fy=scale) if scale != 1 else img_bgr.copy()
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(gray, 45, 140)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:30]

    img_area = small.shape[0] * small.shape[1]
    best = None
    best_score = -1
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < img_area * 0.12:
            continue
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
        if len(approx) != 4 or not cv2.isContourConvex(approx):
            continue
        pts = order_points(approx.reshape(4, 2))
        widths = [np.linalg.norm(pts[1] - pts[0]), np.linalg.norm(pts[2] - pts[3])]
        heights = [np.linalg.norm(pts[3] - pts[0]), np.linalg.norm(pts[2] - pts[1])]
        ww, hh = np.mean(widths), np.mean(heights)
        if hh <= 1 or ww <= 1:
            continue
        ratio = ww / hh
        square_score = 1.0 - min(abs(math.log(max(ratio, 1e-6))), 1.0)
        score = area / img_area + square_score * 0.35
        if 0.65 < ratio < 1.45 and score > best_score:
            best = pts
            best_score = score

    if best is None:
        # 検出失敗時は中央の正方形を仮採用。手修正につなげる。
        side = min(small.shape[0], small.shape[1]) * 0.88
        cx, cy = small.shape[1] / 2, small.shape[0] / 2
        half = side / 2
        best = np.array([[cx-half, cy-half], [cx+half, cy-half], [cx+half, cy+half], [cx-half, cy+half]], dtype=np.float32)
        detected = False
    else:
        detected = True

    best = best / scale
    return best, detected


def warp_board(img_bgr, quad, size=800):
    src = order_points(quad)
    dst = np.array([[0, 0], [size-1, 0], [size-1, size-1], [0, size-1]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(img_bgr, M, (size, size))


def classify_cells(warped):
    """各マス中央の明るさ・彩度から黒/白/空を推定する。"""
    hsv = cv2.cvtColor(warped, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    board = np.zeros((8, 8), dtype=np.int8)
    conf = np.zeros((8, 8), dtype=float)
    cell = warped.shape[0] / 8.0

    # 盤面全体の中央値を基準にし、照明差を多少吸収する。
    global_luma = float(np.median(gray))

    for r in range(8):
        for c in range(8):
            cy = int((r + 0.5) * cell)
            cx = int((c + 0.5) * cell)
            rad = int(cell * 0.25)
            yy, xx = np.ogrid[:gray.shape[0], :gray.shape[1]]
            mask = (xx - cx) ** 2 + (yy - cy) ** 2 <= rad ** 2
            vals = gray[mask]
            sats = hsv[:, :, 1][mask]
            if len(vals) == 0:
                continue
            lum = float(np.median(vals))
            sat = float(np.median(sats))
            p10 = float(np.percentile(vals, 10))
            p90 = float(np.percentile(vals, 90))

            # 黒石: 明確に暗い。白石: 明確に明るく、緑盤より彩度が低いことが多い。
            if lum < min(78, global_luma - 28) or p10 < 35:
                board[r, c] = BLACK
                conf[r, c] = min(1.0, (global_luma - lum + 25) / 90)
            elif lum > max(158, global_luma + 35) and (sat < 95 or p90 > 220):
                board[r, c] = WHITE
                conf[r, c] = min(1.0, (lum - global_luma + 25) / 100)
            else:
                board[r, c] = EMPTY
                conf[r, c] = 0.65
    return board, conf


def process_photo_file(file_obj, source_name="写真"):
    """camera_input / file_uploader の画像を盤面データへ変換する。"""
    try:
        data = np.frombuffer(file_obj.getvalue(), np.uint8)
        img_bgr = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if img_bgr is None:
            return False, "画像を読み取れませんでした。もう一度撮影してください。"

        quad, detected = detect_board_quad(img_bgr)
        warped = warp_board(img_bgr, quad)
        board, _conf = classify_cells(warped)

        st.session_state.board = board
        st.session_state.photo_processed = warped
        if detected:
            st.session_state.recognition_note = f"{source_name}から盤の四角を自動検出しました。"
        else:
            st.session_state.recognition_note = (
                f"{source_name}では盤の外枠を見つけにくかったため、中央を盤として読み取りました。"
                "必要ならマスを修正してください。"
            )
        st.session_state.page = "analyze"
        return True, ""
    except Exception:
        return False, "写真の解析に失敗しました。盤全体が入るように、なるべく真上から撮り直してください。"


# -----------------------------
# 表示用盤面
# -----------------------------
def get_font(size):
    candidates = [
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    for p in candidates:
        try:
            return ImageFont.truetype(p, size)
        except Exception:
            pass
    return ImageFont.load_default()


def render_board(board, recommended=None, legal=None, size=720):
    margin = 58
    board_size = size - margin - 10
    cell = board_size / 8
    img = Image.new("RGB", (size, size), (244, 244, 238))
    d = ImageDraw.Draw(img)
    font = get_font(25)
    small = get_font(20)

    # 盤面
    d.rectangle([margin, 8, margin + board_size, 8 + board_size], fill=(32, 132, 76), outline=(20, 80, 45), width=4)
    for i in range(9):
        x = margin + i * cell
        y = 8 + i * cell
        d.line([x, 8, x, 8 + board_size], fill=(18, 70, 42), width=2)
        d.line([margin, y, margin + board_size, y], fill=(18, 70, 42), width=2)

    # 座標
    for c in range(8):
        txt = COLS[c]
        bbox = d.textbbox((0, 0), txt, font=small)
        d.text((margin + (c + .5) * cell - (bbox[2]-bbox[0])/2, 8 + board_size + 8), txt, fill=(30, 30, 30), font=small)
    for r in range(8):
        txt = str(r + 1)
        bbox = d.textbbox((0, 0), txt, font=small)
        d.text((margin - 12 - (bbox[2]-bbox[0]), 8 + (r + .5) * cell - (bbox[3]-bbox[1])/2), txt, fill=(30, 30, 30), font=small)

    legal = set(legal or [])
    for r in range(8):
        for c in range(8):
            cx = margin + (c + .5) * cell
            cy = 8 + (r + .5) * cell
            rad = cell * .36
            if board[r, c] == BLACK:
                d.ellipse([cx-rad, cy-rad, cx+rad, cy+rad], fill=(20, 20, 20), outline=(0, 0, 0), width=2)
            elif board[r, c] == WHITE:
                d.ellipse([cx-rad, cy-rad, cx+rad, cy+rad], fill=(248, 248, 248), outline=(70, 70, 70), width=2)
            elif (r, c) in legal and recommended is None:
                rr = cell * .08
                d.ellipse([cx-rr, cy-rr, cx+rr, cy+rr], fill=(245, 235, 92))

    if recommended is not None:
        r, c = recommended
        cx = margin + (c + .5) * cell
        cy = 8 + (r + .5) * cell
        rad = cell * .43
        d.ellipse([cx-rad, cy-rad, cx+rad, cy+rad], outline=(255, 220, 40), width=8)
        # 文字化けを避け、★の代わりに大きな中心点＋リングで表示
        rr = cell * .11
        d.ellipse([cx-rr, cy-rr, cx+rr, cy+rr], fill=(255, 225, 40), outline=(120, 90, 0), width=2)
    return img


# -----------------------------
# 学習コンテンツ
# -----------------------------
# 6歳児が「1授業 = 1つの考え方」で進められるように、
# ルール → 序盤の考え方 → 定石の入口 → 名前のある定石 → 上級、の順にする。
# 定石は丸暗記を目的にせず、「同じ序盤でも形が分かれる」ことを理解する教材として扱う。
LESSON_STAGES = [
    {
        "id": "stage1",
        "title": "ステージ1　はじめのきほん",
        "short": "ルールと、いちばん大事な場所",
        "level": "初歩",
        "lesson_ids": ["01", "02", "03", "04"],
    },
    {
        "id": "stage2",
        "title": "ステージ2　序盤の考え方",
        "short": "石の数より、つぎの動き",
        "level": "初級",
        "lesson_ids": ["05", "06", "07", "08"],
    },
    {
        "id": "stage3",
        "title": "ステージ3　定石の入口",
        "short": "最初の2手でできる3つの形",
        "level": "初中級",
        "lesson_ids": ["09", "10", "11", "12"],
    },
    {
        "id": "stage4",
        "title": "ステージ4　名前のある定石",
        "short": "タイガー・キャット・イタリアン・バッファロー",
        "level": "中級",
        "lesson_ids": ["13", "14", "15", "16"],
    },
    {
        "id": "stage5",
        "title": "ステージ5　中盤から終盤へ",
        "short": "安定石・静かな手・パリティ・読み切り",
        "level": "上級",
        "lesson_ids": ["17", "18", "19", "20"],
    },
]

LESSONS = {
    "01": {
        "title": "はさめる場所におこう",
        "goal": "どこに石をおけるか、自分で見つけられるようになる。",
        "teach": [
            "じぶんの石と石で、あいての石を はさめる場所にだけ おけるよ。",
            "たて・よこ・ななめ。どの向きでも、1こ以上 はさめればOK。",
            "まずは『ここにおいたら、どの石をはさめる？』と考えよう。",
        ],
        "remember": "合言葉は『はさめる？』",
        "board": "initial",
        "quiz": {
            "q": "あいての石を1こも はさめない場所には、おける？",
            "options": ["おける", "おけない"],
            "answer": "おけない",
            "why": "オセロは、1こ以上の石をはさんで返せる場所にだけおけるよ。",
        },
    },
    "02": {
        "title": "かどは とくべつ",
        "goal": "4つの角が、どうして強いか説明できる。",
        "teach": [
            "かどの石は、となりが2方向しかないから、いちど取ると もう返されないよ。",
            "だから、かどを取れるときは とても大きなチャンス。",
            "でも『かどを取りたい』だけでなく、あいてにかどを渡さないことも大事。",
        ],
        "remember": "かどは『もう返らない石』。",
        "highlight": {"safe": [(0,0),(0,7),(7,0),(7,7)]},
        "quiz": {
            "q": "いちど取った『かど』の石は、あとで返される？",
            "options": ["返される", "返されない"],
            "answer": "返されない",
            "why": "角の外側にはマスがないので、相手に両側からはさまれないよ。",
        },
    },
    "03": {
        "title": "かどの近くは あぶないことがある",
        "goal": "Xマス・Cマスを、かどが空いているときに警戒できる。",
        "teach": [
            "かどのななめ内側を Xマス、かどの横・下を Cマス とよぶよ。",
            "かどが空いているときに、ここへ早くおくと、あいてに かどを取られやすくなることがある。",
            "いつでもダメではないよ。『かどが空いている？』を先に見るのがポイント。",
        ],
        "remember": "X・Cを見る前に『かどは空いてる？』",
        "highlight": {
            "danger": [(1,1),(1,6),(6,1),(6,6),(0,1),(1,0),(0,6),(1,7),(6,0),(7,1),(6,7),(7,6)],
            "safe": [(0,0),(0,7),(7,0),(7,7)],
        },
        "quiz": {
            "q": "かどが空いているとき、XマスやCマスは？",
            "options": ["すぐおく", "まず注意する"],
            "answer": "まず注意する",
            "why": "相手に角を取らせるきっかけになることがあるから、先を見てからおこう。",
        },
    },
    "04": {
        "title": "たくさん取る手が、いつも一番ではない",
        "goal": "序盤は『今の石数』だけで手を選ばない。",
        "teach": [
            "ゲームのはじめは、石が多いほうが勝ちとはかぎらないよ。",
            "たくさん返すと、自分の石が外へ広がって、相手がおける場所を増やすことがある。",
            "『何こ取れる？』の次に、『相手はどこへおける？』も見よう。",
        ],
        "remember": "序盤は『石の数』より『つぎの動き』。",
        "quiz": {
            "q": "ゲームのはじめ、いちばん多く返せる手を いつも選べばいい？",
            "options": ["いつも選ぶ", "ほかも見る"],
            "answer": "ほかも見る",
            "why": "相手のおける場所や、角を渡さないかも一緒に考えると強くなるよ。",
        },
    },
    "05": {
        "title": "外へ広がりすぎない",
        "goal": "空きマスに接する自分の石を増やしすぎない考え方を知る。",
        "teach": [
            "空いているマスのとなりにある石は、これから返されやすい石だよ。",
            "自分の石が外へ大きく広がりすぎると、相手が使える場所が増えやすい。",
            "序盤は、盤の中のほうに小さくまとまる形も大切。",
        ],
        "remember": "外へ出しすぎず、形を小さく。",
        "quiz": {
            "q": "序盤に自分の石が外へどんどん広がると？",
            "options": ["相手の手が増えやすい", "必ず勝てる"],
            "answer": "相手の手が増えやすい",
            "why": "空きマスの近くに自分の石が増えるほど、相手がはさめる場所も増えやすいよ。",
        },
    },
    "06": {
        "title": "あいての『おける場所』をへらそう",
        "goal": "モビリティ（合法手の数）の考え方を使える。",
        "teach": [
            "相手がおける場所が10こあるのと、2こしかないのでは、2このほうが相手はこまるよ。",
            "この『おける場所の数』を、むずかしい言葉で モビリティ というよ。",
            "強い手を探すときは、打ったあとに『相手は何か所？』を数えてみよう。",
        ],
        "remember": "相手のえらべる手を少なくする。",
        "quiz": {
            "q": "相手のおける場所は、どちらがうれしい？",
            "options": ["2か所", "10か所"],
            "answer": "2か所",
            "why": "選べる手が少ないほど、相手は好きな作戦をとりにくくなるよ。",
        },
    },
    "07": {
        "title": "へんは、かどとセットで考える",
        "goal": "辺に石を置く前に、角との関係を見る。",
        "teach": [
            "盤のはし（辺）は、まんなかより返されにくいことがある。",
            "でも、かどが空いている辺は、打ち方によって相手へ角を渡すこともある。",
            "『辺だから強い』ではなく、『この辺は角とどうつながる？』と見るよ。",
        ],
        "remember": "へんを見るときは、近くのかども見る。",
        "highlight": {"focus": [(0,3),(3,0),(7,4),(4,7)], "safe": [(0,0),(0,7),(7,0),(7,7)]},
        "quiz": {
            "q": "辺におけるなら、いつでも安全？",
            "options": ["いつでも安全", "角との関係を見る"],
            "answer": "角との関係を見る",
            "why": "辺の形しだいで、相手に角を取らせることがあるからだよ。",
        },
    },
    "08": {
        "title": "つぎのつぎまで見る",
        "goal": "自分→相手→自分の3手を順番に考える。",
        "teach": [
            "まず『ここにおく』。その次に『相手はどこへおく？』。最後に『自分はそのあとどこへおく？』。",
            "1手だけでなく、3つの順番で見ると、ワナに気づきやすくなる。",
            "全部の手を読む必要はないよ。角やXマスが関係するときだけでも、3手見る習慣をつけよう。",
        ],
        "remember": "じぶん → あいて → じぶん。",
        "quiz": {
            "q": "『つぎのつぎ』を見る順番は？",
            "options": ["じぶん→あいて→じぶん", "じぶん→じぶん→あいて"],
            "answer": "じぶん→あいて→じぶん",
            "why": "オセロは交代で打つので、相手の返しをはさんで考えるよ。",
        },
    },
    "09": {
        "title": "定石は F5 から覚える",
        "goal": "最初の4か所は対称で、定石ではF5を代表にすることを知る。",
        "teach": [
            "黒の1手目は C4・D3・E6・F5 の4か所。形は回すと同じだよ。",
            "定石を勉強するときは、同じ形を何回も覚えなくてよいように F5 から始めることが多い。",
            "ここから白の返し方で、3つの大きな入口に分かれるよ。",
        ],
        "remember": "最初は4か所とも同じ形。勉強では F5 を代表にする。",
        "sequence": ["F5"],
        "sequence_name": "黒 F5",
        "quiz": {
            "q": "黒の最初の4か所は、強さがぜんぶ違う？",
            "options": ["形は同じ", "ぜんぶ違う"],
            "answer": "形は同じ",
            "why": "盤を回して見ると同じ形になるので、定石ではF5を代表にして覚えられるよ。",
        },
    },
    "10": {
        "title": "垂直型（Perpendicular）",
        "goal": "F5 → D6 の形を見て、垂直型の入口を覚える。",
        "teach": [
            "黒 F5 のあと、白 D6 と打つ形を Perpendicular（垂直型）というよ。",
            "ここからタイガー・キャット・イタリアンなど、たくさんの定石へ分かれていく。",
            "まずは名前より、『F5のあと白がD6』という形を見て覚えよう。",
        ],
        "remember": "F5 → D6 ＝ 垂直型。",
        "sequence": ["F5", "D6"],
        "sequence_name": "垂直型　F5 → D6",
        "quiz": {
            "q": "F5 → D6 は、どの入口？",
            "options": ["垂直型", "平行型", "斜め型"],
            "answer": "垂直型",
            "why": "この2手が Perpendicular（垂直型）の基本形だよ。",
        },
    },
    "11": {
        "title": "平行型（Parallel）",
        "goal": "F5 → F4 の形を見て、平行型の入口を覚える。",
        "teach": [
            "黒 F5 のあと、白 F4 と打つ形を Parallel（平行型）というよ。",
            "垂直型とは盤の形がちがうので、そのあとにできる定石も変わる。",
            "まずは『F5の上にF4がならぶ』形として覚えよう。",
        ],
        "remember": "F5 → F4 ＝ 平行型。",
        "sequence": ["F5", "F4"],
        "sequence_name": "平行型　F5 → F4",
        "quiz": {
            "q": "F5 → F4 は、どの入口？",
            "options": ["平行型", "垂直型", "斜め型"],
            "answer": "平行型",
            "why": "この2手が Parallel（平行型）の基本形だよ。",
        },
    },
    "12": {
        "title": "斜め型（Diagonal）",
        "goal": "F5 → F6 の形を見て、斜め型の入口を覚える。",
        "teach": [
            "黒 F5 のあと、白 F6 と打つ形を Diagonal（斜め型）というよ。",
            "この入口から、バッファローなどの名前のある定石へ進むことができる。",
            "3つの入口を見分けられれば、定石の地図がかなり見やすくなるよ。",
        ],
        "remember": "F5 → F6 ＝ 斜め型。",
        "sequence": ["F5", "F6"],
        "sequence_name": "斜め型　F5 → F6",
        "quiz": {
            "q": "F5 → F6 は、どの入口？",
            "options": ["斜め型", "平行型", "垂直型"],
            "answer": "斜め型",
            "why": "この2手が Diagonal（斜め型）の基本形だよ。",
        },
    },
    "13": {
        "title": "タイガー定石",
        "goal": "垂直型からタイガーへ進む5手を、盤の形で見られる。",
        "teach": [
            "タイガーは、垂直型 F5 → D6 から始まる名前のある定石だよ。",
            "手順は F5 → D6 → C3 → D3 → C4。",
            "丸暗記だけでなく、最後のC4まで打ったときに『どちらの石が外へ出ているか』も見よう。",
        ],
        "remember": "タイガー：F5 → D6 → C3 → D3 → C4",
        "sequence": ["F5", "D6", "C3", "D3", "C4"],
        "sequence_name": "Tiger　F5 → D6 → C3 → D3 → C4",
        "quiz": {
            "q": "タイガーは、最初の2手ではどの入口？",
            "options": ["垂直型", "平行型", "斜め型"],
            "answer": "垂直型",
            "why": "最初が F5 → D6 なので、垂直型から分かれる定石だよ。",
        },
    },
    "14": {
        "title": "キャット定石",
        "goal": "タイガーと違う、垂直型の別ルートを見比べる。",
        "teach": [
            "キャットも、最初は F5 → D6 の垂直型。",
            "手順は F5 → D6 → C4 → D3 → C5。",
            "タイガーと3手目がちがうね。定石は『途中の分かれ道』を見ると覚えやすいよ。",
        ],
        "remember": "キャット：F5 → D6 → C4 → D3 → C5",
        "sequence": ["F5", "D6", "C4", "D3", "C5"],
        "sequence_name": "Cat　F5 → D6 → C4 → D3 → C5",
        "quiz": {
            "q": "タイガーとキャットが最初に分かれるのは？",
            "options": ["3手目", "1手目"],
            "answer": "3手目",
            "why": "どちらも F5 → D6 まで同じで、3手目が C3 と C4 に分かれるよ。",
        },
    },
    "15": {
        "title": "イタリアン定石",
        "goal": "キャットと途中まで同じ定石を見比べる。",
        "teach": [
            "イタリアンは F5 → D6 → C4 → D3 までキャットと同じ。",
            "5手目を E6 と打つとイタリアン、C5 と打つとキャットになる。",
            "『同じ4手から、5手目で別の名前になる』という定石の枝分かれを見よう。",
        ],
        "remember": "イタリアン：F5 → D6 → C4 → D3 → E6",
        "sequence": ["F5", "D6", "C4", "D3", "E6"],
        "sequence_name": "Italian　F5 → D6 → C4 → D3 → E6",
        "quiz": {
            "q": "F5 → D6 → C4 → D3 のあと、E6なら？",
            "options": ["イタリアン", "キャット"],
            "answer": "イタリアン",
            "why": "同じ4手から、E6ならイタリアン、C5ならキャットに分かれるよ。",
        },
    },
    "16": {
        "title": "バッファロー定石",
        "goal": "斜め型から始まる定石を1つ覚える。",
        "teach": [
            "バッファローは、F5 → F6 の斜め型から始まる定石だよ。",
            "手順は F5 → F6 → E6 → D6 → C3。",
            "垂直型のタイガーたちと比べると、最初の2手から盤の形がちがうことがわかる。",
        ],
        "remember": "バッファロー：F5 → F6 → E6 → D6 → C3",
        "sequence": ["F5", "F6", "E6", "D6", "C3"],
        "sequence_name": "Buffalo　F5 → F6 → E6 → D6 → C3",
        "quiz": {
            "q": "バッファローは、最初の2手ではどの入口？",
            "options": ["斜め型", "垂直型", "平行型"],
            "answer": "斜め型",
            "why": "最初が F5 → F6 なので、斜め型から始まるよ。",
        },
    },
    "17": {
        "title": "安定石をつなげる",
        "goal": "もう返されない石を、角から増やす考え方を知る。",
        "teach": [
            "角は安定石。そこから辺にそって、もう返されない石がつながることがある。",
            "終盤では『今何こある？』より『最後まで残る石は何こ？』を見る。",
            "角を取ったあとも、角からつながる石を少しずつ増やせるか考えよう。",
        ],
        "remember": "角から『返らない石』をつなげる。",
        "highlight": {"safe": [(0,0),(0,1),(0,2),(1,0),(2,0)]},
        "quiz": {
            "q": "角からつながって、もう返されない石は？",
            "options": ["安定石", "空きマス"],
            "answer": "安定石",
            "why": "相手がもう返せない石は、最後まで自分の石として残るよ。",
        },
    },
    "18": {
        "title": "静かな手を見つける",
        "goal": "少なく返す手でも強い理由を説明できる。",
        "teach": [
            "1こしか返さない手でも、相手のおける場所を大きく減らせるなら強いことがある。",
            "こういう目立たない手を、ここでは『静かな手』として覚えよう。",
            "見る順番は『返す数 → 相手の手の数 → 角を渡さないか』。",
        ],
        "remember": "少なく返して、相手の手も少なく。",
        "quiz": {
            "q": "1こしか返さない手は、必ず弱い？",
            "options": ["必ず弱い", "強いこともある"],
            "answer": "強いこともある",
            "why": "相手の選択肢を減らせるなら、返す石が少なくても価値が高いよ。",
        },
    },
    "19": {
        "title": "終盤のパリティ（ぐうすう・きすう）",
        "goal": "空きマスの数と最後の1手の関係を知る。",
        "teach": [
            "終盤は、空きマスがまとまりごとに何こあるかを見ることがある。",
            "2・4・6のような偶数か、1・3・5のような奇数かで、最後に打てる側が変わりやすい。",
            "6歳ではまず『空きマスを数えると、最後の順番が見える』までわかれば十分。",
        ],
        "remember": "終盤は、空きマスの数も作戦になる。",
        "quiz": {
            "q": "終盤で数えると役立つものは？",
            "options": ["空きマス", "盤の色"],
            "answer": "空きマス",
            "why": "空きマスの数で、どちらが最後に打ちやすいか考えられるよ。",
        },
    },
    "20": {
        "title": "最後は読み切る",
        "goal": "空きが少なくなったら、最後まで順番に読む習慣をつける。",
        "teach": [
            "空きがたくさんある序盤は、全部読むのはむずかしい。",
            "でも空きが6こ、4こ、2こと少なくなったら、最後まで全部の手を試しやすくなる。",
            "上級者は『今よさそう』ではなく、『最後に何こ残る？』で手をくらべるよ。",
        ],
        "remember": "空きが少なくなったら、最後まで読む。",
        "quiz": {
            "q": "空きが4こくらいなら、どうする？",
            "options": ["最後まで読んでみる", "今の石だけ数える"],
            "answer": "最後まで読んでみる",
            "why": "終盤は手の候補が少ないので、最後まで読むほど正確に選べるよ。",
        },
    },
}


def board_from_sequence(sequence):
    """F5, D6... の定石表記から盤面を作る。黒から交互に打つ。"""
    board = initial_board()
    player = BLACK
    last = None
    for text_move in sequence:
        col = COLS.index(text_move[0].upper())
        row = int(text_move[1:]) - 1
        move = (row, col)
        flips = flips_for_move(board, player, row, col)
        if not flips:
            break
        board = apply_move(board, player, move, flips)
        last = move
        player = -player
    return board, last


def render_teaching_board(board=None, focus=None, danger=None, safe=None, size=650):
    """授業用。注目マス・注意マス・安全マスを盤上で囲む。"""
    if board is None:
        board = initial_board()
    img = render_board(board, size=size)
    draw = ImageDraw.Draw(img)
    margin = 58
    board_size = size - margin - 10
    cell = board_size / 8

    def outline_cells(cells, color, width=6):
        for r, c in cells or []:
            x0 = margin + c * cell + 4
            y0 = 8 + r * cell + 4
            x1 = margin + (c + 1) * cell - 4
            y1 = 8 + (r + 1) * cell - 4
            draw.rounded_rectangle([x0, y0, x1, y1], radius=8, outline=color, width=width)

    outline_cells(safe, (65, 175, 95), 7)
    outline_cells(danger, (220, 75, 75), 7)
    outline_cells(focus, (245, 200, 55), 7)
    return img


def stage_for_lesson(lesson_id):
    for stage in LESSON_STAGES:
        if lesson_id in stage["lesson_ids"]:
            return stage
    return LESSON_STAGES[0]


def next_lesson_id(lesson_id):
    ids = list(LESSONS.keys())
    try:
        idx = ids.index(lesson_id)
    except ValueError:
        return None
    return ids[idx + 1] if idx + 1 < len(ids) else None


# -----------------------------
# 思考実験・理解度テスト用盤面
# -----------------------------
PRACTICE_LEVELS = {
    1: {"name": "レベル1　やさしい局面", "plies": 8, "hint": "おける場所が少ない局面。まずは安全そうな場所を1つずつ比べよう。"},
    2: {"name": "レベル2　序盤", "plies": 14, "hint": "取る石の数だけでなく、角・辺と相手の次の手も見よう。"},
    3: {"name": "レベル3　中盤", "plies": 24, "hint": "次の相手の手、その次の自分の手まで考えよう。"},
    4: {"name": "レベル4　むずかしい中盤", "plies": 36, "hint": "少なく返す手や、相手の自由をうばう手も候補にしよう。"},
    5: {"name": "レベル5　終盤", "plies": 50, "hint": "空きマスが少ないので、最後まで読むつもりで考えよう。"},
}


def generate_practice_position(level=1, variant=0):
    """合法手だけを使って、再現可能な練習局面を作る。"""
    level = int(max(1, min(5, level)))
    target_plies = PRACTICE_LEVELS[level]["plies"]
    board = initial_board()
    player = BLACK
    for ply in range(target_plies):
        moves = legal_moves(board, player)
        if not moves:
            if not legal_moves(board, -player):
                break
            player = -player
            moves = legal_moves(board, player)
            if not moves:
                break
        ordered = sorted(
            moves.items(),
            key=lambda kv: (POSITION_WEIGHTS[kv[0][0], kv[0][1]], len(kv[1]), -kv[0][0], -kv[0][1]),
            reverse=True,
        )
        idx = (variant * 7 + ply * 3 + level * 5) % len(ordered)
        move, flips = ordered[idx]
        board = apply_move(board, player, move, flips)
        player = -player
    if not legal_moves(board, player) and legal_moves(board, -player):
        player = -player
    return board, player


def get_practice_position(level=1, variant=0):
    """最低2択になる局面を優先して返す。"""
    fallback = None
    for offset in range(18):
        board, player = generate_practice_position(level, variant + offset)
        moves = legal_moves(board, player)
        if fallback is None and moves:
            fallback = (board, player, variant + offset)
        if len(moves) >= 2:
            if level <= 2 and len(moves) <= 6:
                return board, player, variant + offset
            if level >= 3:
                return board, player, variant + offset
    if fallback is not None:
        return fallback
    return initial_board(), BLACK, variant


def move_quality_summary(board, player, move):
    flips = flips_for_move(board, player, move[0], move[1])
    after = apply_move(board, player, move, flips)
    opp_count = len(legal_moves(after, -player))
    if move in CORNERS:
        return f"{coord(move)}は かど。{len(flips)}こ返して、相手のおける場所は {opp_count}こだよ。"
    if corner_adjacent_to_empty_corner(move, board):
        return f"{coord(move)}は かどの近く。相手に かどをわたさないか注意しよう。"
    if len(flips) <= 2:
        return f"{coord(move)}は {len(flips)}こだけ返す手。相手のおける場所は {opp_count}こだよ。"
    return f"{coord(move)}は {len(flips)}こ返す手。相手のおける場所は {opp_count}こだよ。"


def render_level_meter(level):
    dots = ''.join('<span class="level-dot on"></span>' if i <= level else '<span class="level-dot"></span>' for i in range(1, 6))
    return f'<div class="level-meter">{dots}</div>'


def move_buttons(moves, key_prefix, step=0):
    ordered = sorted(moves.keys(), key=lambda m: (m[0], m[1]))
    if not ordered:
        return None
    ncols = min(4, max(2, len(ordered)))
    cols = st.columns(ncols)
    clicked = None
    for i, move in enumerate(ordered):
        if cols[i % ncols].button(
            coord(move),
            use_container_width=True,
            key=f"{key_prefix}_{step}_{coord(move)}",
        ):
            clicked = move
    return clicked


def reset_experiment(level=None, variant=None):
    if level is None:
        level = st.session_state.get("experiment_level", 2)
    if variant is None:
        variant = st.session_state.get("experiment_variant", 0)
    board, player, actual_variant = get_practice_position(level, variant)
    st.session_state.experiment_level = level
    st.session_state.experiment_variant = actual_variant
    st.session_state.experiment_base_board = board.copy()
    st.session_state.experiment_base_player = player
    st.session_state.experiment_board = board.copy()
    st.session_state.experiment_user_player = player
    st.session_state.experiment_history = []
    st.session_state.experiment_step = 0
    st.session_state.experiment_custom = False


def set_experiment_from_board(board, player):
    st.session_state.experiment_base_board = board.copy()
    st.session_state.experiment_base_player = player
    st.session_state.experiment_board = board.copy()
    st.session_state.experiment_user_player = player
    st.session_state.experiment_history = []
    st.session_state.experiment_step = 0
    st.session_state.experiment_custom = True


def play_experiment_turn(chosen):
    board = st.session_state.experiment_board.copy()
    user_player = st.session_state.experiment_user_player
    user_moves = legal_moves(board, user_player)
    if chosen not in user_moves:
        return
    teacher_move, teacher_flips, teacher_meta = best_move_training(board, user_player)
    chosen_flips = user_moves[chosen]
    chosen_after = apply_move(board, user_player, chosen, chosen_flips)
    same_as_teacher = chosen == teacher_move
    best_opp_count = len(legal_moves(teacher_meta["after"], -user_player)) if teacher_meta else 0
    chosen_opp_count = len(legal_moves(chosen_after, -user_player))
    best_reason = explain_move(board, user_player, teacher_move, teacher_flips, teacher_meta)
    if same_as_teacher:
        compare_text = f"せんせいAIも {coord(chosen)} をえらんだよ。{best_reason}"
    else:
        compare_text = f"あなたは {coord(chosen)}。せんせいAIなら {coord(teacher_move)}。{best_reason}"
        if best_opp_count < chosen_opp_count:
            compare_text += f" 相手のおける場所も {chosen_opp_count}こから {best_opp_count}こにへらせるよ。"
        else:
            compare_text += " 今回は相手のおける場所の数だけでなく、石の位置とその先まで合わせて選んでいるよ。"
    record = {
        "user_move": coord(chosen),
        "teacher_user_move": coord(teacher_move),
        "compare": compare_text,
        "reply_move": None,
        "reply_reason": None,
        "passes": [],
    }
    board = chosen_after
    ai_player = -user_player
    safety = 0
    while safety < 4:
        ai_moves = legal_moves(board, ai_player)
        if not ai_moves:
            if not legal_moves(board, user_player):
                record["passes"].append("両方ともおけないのでゲーム終了。")
            else:
                record["passes"].append("相手はおける場所がないのでパス。")
            break
        ai_move, ai_flips, ai_meta = best_move_training(board, ai_player)
        reason = explain_move(board, ai_player, ai_move, ai_flips, ai_meta)
        board = apply_move(board, ai_player, ai_move, ai_flips)
        if record["reply_move"] is None:
            record["reply_move"] = coord(ai_move)
            record["reply_reason"] = reason
        else:
            record["passes"].append(f"あなたがパスになったので、相手AIは {coord(ai_move)} にもう一度打ったよ。")
        if legal_moves(board, user_player):
            break
        if not legal_moves(board, ai_player):
            record["passes"].append("このあと両方ともおけないのでゲーム終了。")
            break
        record["passes"].append("あなたはおける場所がないのでパス。")
        safety += 1
    st.session_state.experiment_board = board
    st.session_state.experiment_history.append(record)
    st.session_state.experiment_step += 1


def new_challenge(level=None, variant=None):
    if level is None:
        level = st.session_state.get("challenge_level", 1)
    if variant is None:
        variant = st.session_state.get("challenge_variant", 0)
    board, player, actual_variant = get_practice_position(level, variant)
    st.session_state.challenge_level = level
    st.session_state.challenge_variant = actual_variant
    st.session_state.challenge_board = board.copy()
    st.session_state.challenge_player = player
    st.session_state.challenge_feedback = None
    st.session_state.challenge_selected = None
    st.session_state.challenge_best = None
    st.session_state.challenge_next_level = level


# -----------------------------
# セッション状態
# -----------------------------
if "page" not in st.session_state:
    st.session_state.page = "home"
if "board" not in st.session_state:
    st.session_state.board = initial_board()
if "photo_processed" not in st.session_state:
    st.session_state.photo_processed = None
if "recognition_note" not in st.session_state:
    st.session_state.recognition_note = ""
if "current_lesson" not in st.session_state:
    st.session_state.current_lesson = "01"
if "completed_lessons" not in st.session_state:
    st.session_state.completed_lessons = []
if "experiment_level" not in st.session_state:
    st.session_state.experiment_level = 2
if "experiment_variant" not in st.session_state:
    st.session_state.experiment_variant = 0
if "experiment_board" not in st.session_state:
    reset_experiment(2, 0)
if "experiment_custom" not in st.session_state:
    st.session_state.experiment_custom = False
if "challenge_level" not in st.session_state:
    st.session_state.challenge_level = 1
if "challenge_variant" not in st.session_state:
    st.session_state.challenge_variant = 0
if "challenge_total" not in st.session_state:
    st.session_state.challenge_total = 0
if "challenge_correct" not in st.session_state:
    st.session_state.challenge_correct = 0
if "challenge_board" not in st.session_state:
    new_challenge(1, 0)


def go(page):
    st.session_state.page = page
    st.rerun()


# -----------------------------
# UI: ホーム
# -----------------------------
st.markdown('''
<div class="hero-panel">
    <div class="hero-kicker">PHOTO × STRATEGY × LESSON</div>
    <div class="big-title">⚫ オセロせんせい ⚪</div>
    <div class="sub">しゃしんを とったら、おすすめの1手をいっしょに考えるよ</div>
    <div class="hero-chips">
        <span class="hero-chip">📷 盤面を撮る</span>
        <span class="hero-chip">⭐ おすすめの1手</span>
        <span class="hero-chip">📘 授業で学ぶ</span>
    </div>
</div>
''', unsafe_allow_html=True)

if st.session_state.page == "home":
    st.markdown("### 📷 盤面を撮る")
    st.caption("盤全体が入るように、なるべく真上から撮ってください。")

    # 1) 通常のカメラ。ブラウザがカメラ利用を許可していればそのまま撮影できる。
    pic = st.camera_input(
        "オセロの盤面を撮影",
        resolution="720p",
        label_visibility="collapsed",
        key="board_camera",
    )
    if pic is not None:
        ok, err = process_photo_file(pic, "カメラ写真")
        if ok:
            st.rerun()
        else:
            st.error(err)

    # 2) Android のブラウザ内カメラが権限で止まった場合の実用的な逃げ道。
    #    スマホでは画像選択時に「カメラ」を選べる端末が多い。
    with st.expander("カメラが開かないとき", expanded=False):
        st.caption("ブラウザのカメラ権限が使えない場合は、こちらから撮影した写真を読み込めます。")
        fallback_pic = st.file_uploader(
            "📷 写真を撮る / 写真を選ぶ",
            type=["jpg", "jpeg", "png"],
            accept_multiple_files=False,
            key="board_photo_upload",
        )
        if fallback_pic is not None:
            ok, err = process_photo_file(fallback_pic, "選んだ写真")
            if ok:
                st.rerun()
            else:
                st.error(err)

        st.caption("※ カメラ欄に『This app would like to use your camera』と出る場合は、アプリではなくブラウザ側のカメラ権限が止まっています。")

    st.markdown("---")
    st.markdown('<div class="mode-card"><div class="mode-title">考えて強くなる</div><div class="mode-text">授業で覚えるだけでなく、同じ盤面を何度も動かして「もしここなら？」を試せます。</div></div>', unsafe_allow_html=True)
    if st.button("📘 オセロを学ぶ（授業）", use_container_width=True, type="secondary"):
        go("learn")

    c_test, c_challenge = st.columns(2)
    if c_test.button("🧪 テスト盤面", use_container_width=True):
        go("experiment")
    if c_challenge.button("🎯 挑戦", use_container_width=True):
        go("challenge")

    with st.expander("写真なしで試す"):
        st.write("最初の盤面から、おすすめ手の動きを確認できます。")
        if st.button("最初の盤面で試す", use_container_width=True):
            st.session_state.board = initial_board()
            st.session_state.photo_processed = None
            st.session_state.recognition_note = "最初の盤面を使っています。"
            go("analyze")


# -----------------------------
# UI: 解析
# -----------------------------
elif st.session_state.page == "analyze":
    if st.button("← もどる"):
        go("home")

    st.markdown("### ① 盤面をかくにん")
    st.caption(st.session_state.recognition_note)
    st.image(render_board(st.session_state.board), use_container_width=True)

    with st.expander("マスがちがうときは修正する"):
        c1, c2 = st.columns(2)
        with c1:
            col_label = st.selectbox("よこ", list(COLS), index=0)
        with c2:
            row_num = st.selectbox("たて", list(range(1, 9)), index=0)
        r = row_num - 1
        c = COLS.index(col_label)
        current = st.session_state.board[r, c]
        current_txt = "黒" if current == BLACK else "白" if current == WHITE else "空"
        st.write(f"いまの {col_label}{row_num}：**{current_txt}**")
        a, b, c3 = st.columns(3)
        if a.button("● 黒", use_container_width=True):
            st.session_state.board[r, c] = BLACK
            st.rerun()
        if b.button("○ 白", use_container_width=True):
            st.session_state.board[r, c] = WHITE
            st.rerun()
        if c3.button("空にする", use_container_width=True):
            st.session_state.board[r, c] = EMPTY
            st.rerun()

    st.markdown("### ② どっちの番？")
    p1, p2 = st.columns(2)
    black = p1.button("● くろ", use_container_width=True, type="primary")
    white = p2.button("○ しろ", use_container_width=True)

    if black or white:
        player = BLACK if black else WHITE
        st.session_state.player = player
        st.session_state.page = "result"
        st.rerun()


# -----------------------------
# UI: 結果
# -----------------------------
elif st.session_state.page == "result":
    if st.button("← 盤面をなおす"):
        go("analyze")

    board = st.session_state.board
    player = st.session_state.get("player", BLACK)
    moves = legal_moves(board, player)

    if not moves:
        opp_moves = legal_moves(board, -player)
        st.markdown('<div class="result-card"><div class="result-main">おける場所がないよ</div><div class="kid-text">こんかいは パス。あいての番だよ。</div></div>', unsafe_allow_html=True)
        st.image(render_board(board), use_container_width=True)
        if not opp_moves:
            bcnt = int(np.sum(board == BLACK))
            wcnt = int(np.sum(board == WHITE))
            st.write(f"ゲーム終了：黒 {bcnt}こ / 白 {wcnt}こ")
    else:
        with st.spinner("いちばんよい手を考えています…"):
            move, flips, meta = best_move(board, player)
        explanation = explain_move(board, player, move, flips, meta)
        player_name = "くろ" if player == BLACK else "しろ"
        st.markdown(
            f'<div class="result-card"><div class="result-main">⭐ {coord(move)} に おこう！</div>'
            f'<div class="kid-text">{explanation}</div></div>',
            unsafe_allow_html=True,
        )
        st.image(render_board(board, recommended=move), use_container_width=True)

        with st.expander("おとな向け：この手を選んだ理由"):
            after = meta["after"]
            st.write(f"探索の深さ：{meta['depth']}手先相当")
            st.write(f"この手で返る石：{len(flips)}個")
            st.write(f"相手の次の合法手：{len(legal_moves(after, -player))}か所")
            st.write("角・危険な角周辺・合法手数（モビリティ）・境界石・終盤の石数を組み合わせて評価しています。")

    c1, c2 = st.columns(2)
    if c1.button("📷 つぎの盤面を撮る", use_container_width=True, type="primary"):
        st.session_state.photo_processed = None
        go("home")
    if c2.button("📘 オセロを学ぶ（授業）", use_container_width=True):
        go("learn")
    if st.button("🧪 この盤面で『もしここなら？』を試す", use_container_width=True):
        set_experiment_from_board(board, player)
        go("experiment")


# -----------------------------
# UI: 思考実験テスト盤面
# -----------------------------
elif st.session_state.page == "experiment":
    top1, top2 = st.columns(2)
    if top1.button("← ホームへ", use_container_width=True):
        go("home")
    if top2.button("🎯 挑戦へ", use_container_width=True):
        go("challenge")

    st.markdown("### 🧪 テスト盤面")
    st.caption("あなたが1手打つ → せんせいAIが相手のおすすめ手を打つ、を何度も繰り返せます。最初の局面へ戻して別の手も比較できます。")

    level_labels = {k: v["name"] for k, v in PRACTICE_LEVELS.items()}
    selected_level = st.selectbox(
        "盤面のむずかしさ",
        list(level_labels.keys()),
        index=max(0, st.session_state.experiment_level - 1),
        format_func=lambda x: level_labels[x],
        key="experiment_level_select",
    )
    if selected_level != st.session_state.experiment_level:
        reset_experiment(selected_level, st.session_state.experiment_variant + 1)
        st.rerun()

    if st.session_state.get("experiment_custom", False):
        st.markdown('<div class="feedback-neutral"><b>写真・実戦から持ってきた盤面</b><br>この同じ局面から、何通りでも試せます。</div>', unsafe_allow_html=True)
    else:
        st.markdown(render_level_meter(st.session_state.experiment_level), unsafe_allow_html=True)
    st.markdown(
        f'<div class="think-card"><div class="mode-title">考えるポイント</div><div class="mode-text">{PRACTICE_LEVELS[st.session_state.experiment_level]["hint"]}</div></div>',
        unsafe_allow_html=True,
    )

    board = st.session_state.experiment_board
    user_player = st.session_state.experiment_user_player
    user_name = "くろ" if user_player == BLACK else "しろ"
    user_moves = legal_moves(board, user_player)
    st.markdown(f'<span class="turn-badge">あなたは {user_name}</span>', unsafe_allow_html=True)

    if user_moves:
        st.image(render_board(board, legal=list(user_moves.keys())), use_container_width=True)
        st.markdown("#### ここに置いたら、相手AIはどう返す？")
        st.caption("黄色い点がおける場所です。座標ボタンを1つ選んでください。")
        chosen = move_buttons(user_moves, "experiment_move", st.session_state.experiment_step)
        if chosen is not None:
            with st.spinner("せんせいAIが先を考えています…"):
                play_experiment_turn(chosen)
            st.rerun()
    else:
        st.image(render_board(board), use_container_width=True)
        if legal_moves(board, -user_player):
            st.info("あなたはおける場所がないのでパスです。最初の盤面へ戻すか、別の盤面を試してください。")
        else:
            st.success("この局面はゲーム終了です。")

    if st.session_state.experiment_history:
        latest = st.session_state.experiment_history[-1]
        st.markdown("#### いまの1手をふり返る")
        same = latest["user_move"] == latest["teacher_user_move"]
        cls = "feedback-good" if same else "feedback-neutral"
        title = "AIと同じ手！" if same else "別の手もくらべよう"
        st.markdown(
            f'<div class="{cls}"><b>{title}</b><br>{latest["compare"]}</div>',
            unsafe_allow_html=True,
        )
        if latest["reply_move"]:
            st.markdown(
                f'<div class="feedback-neutral"><b>相手AIの返し：{latest["reply_move"]}</b><br>{latest["reply_reason"]}</div>',
                unsafe_allow_html=True,
            )
        for note in latest["passes"]:
            st.caption(note)

        with st.expander("これまでの実験を見る", expanded=False):
            for i, rec in enumerate(reversed(st.session_state.experiment_history), start=1):
                n = len(st.session_state.experiment_history) - i + 1
                reply = rec["reply_move"] or "パス"
                st.write(f"{n}. あなた {rec['user_move']} → 相手AI {reply}")

    a, b = st.columns(2)
    if a.button("↩ 同じ最初の盤面に戻る", use_container_width=True):
        st.session_state.experiment_board = st.session_state.experiment_base_board.copy()
        st.session_state.experiment_user_player = st.session_state.experiment_base_player
        st.session_state.experiment_history = []
        st.session_state.experiment_step = 0
        st.rerun()
    if b.button("🔄 別の盤面", use_container_width=True):
        reset_experiment(st.session_state.experiment_level, st.session_state.experiment_variant + 1)
        st.rerun()


# -----------------------------
# UI: 適応型チャレンジ
# -----------------------------
elif st.session_state.page == "challenge":
    top1, top2 = st.columns(2)
    if top1.button("← ホームへ", use_container_width=True):
        go("home")
    if top2.button("🧪 テスト盤面へ", use_container_width=True):
        go("experiment")

    level = st.session_state.challenge_level
    st.markdown("### 🎯 どこに打つ？")
    st.caption("正解なら次のレベルへ。不正解なら1段やさしい盤面へ戻って、同じ考え方をもう一度練習します。")
    st.markdown(render_level_meter(level), unsafe_allow_html=True)
    st.markdown(
        f'<div class="challenge-card"><div class="mode-title">{PRACTICE_LEVELS[level]["name"]}</div><div class="mode-text">{PRACTICE_LEVELS[level]["hint"]}</div></div>',
        unsafe_allow_html=True,
    )

    board = st.session_state.challenge_board
    player = st.session_state.challenge_player
    player_name = "くろ" if player == BLACK else "しろ"
    moves = legal_moves(board, player)

    if st.session_state.challenge_feedback is None:
        st.markdown(f'<span class="turn-badge">{player_name} の番</span>', unsafe_allow_html=True)
        show_legal = level <= 3
        st.image(render_board(board, legal=list(moves.keys()) if show_legal else None), use_container_width=True)
        if show_legal:
            st.caption("黄色い点がおける場所。いちばんよいと思う手を選んでください。")
        else:
            st.caption("レベル4・5は黄色いヒントなし。座標ボタンから考えて選んでください。")

        chosen = move_buttons(moves, "challenge_move", st.session_state.challenge_variant)
        if chosen is not None:
            with st.spinner("答えを確かめています…"):
                best, best_flips, meta = best_move_training(board, player)
                top_score = meta["all"][0][0]
                accepted = {m for score, m, _f, _child in meta["all"] if abs(score - top_score) < 1e-9}
                correct = chosen in accepted
                reason = explain_move(board, player, best, best_flips, meta)
                st.session_state.challenge_total += 1
                if correct:
                    st.session_state.challenge_correct += 1
                next_level = min(5, level + 1) if correct else max(1, level - 1)
                st.session_state.challenge_feedback = {
                    "correct": correct,
                    "chosen": coord(chosen),
                    "best": coord(best),
                    "reason": reason,
                    "chosen_summary": move_quality_summary(board, player, chosen),
                    "next_level": next_level,
                }
                st.session_state.challenge_selected = chosen
                st.session_state.challenge_best = best
                st.session_state.challenge_next_level = next_level
            st.rerun()
    else:
        fb = st.session_state.challenge_feedback
        st.image(render_board(board, recommended=st.session_state.challenge_best), use_container_width=True)
        if fb["correct"]:
            st.markdown(
                f'<div class="feedback-good"><b>せいかい！ {fb["chosen"]}</b><br>{fb["reason"]}<br>つぎはレベル {fb["next_level"]}。</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f'<div class="feedback-bad"><b>今回は {fb["chosen"]}。AIのおすすめは {fb["best"]}</b><br>{fb["chosen_summary"]}<br><br><b>おすすめ手の理由：</b>{fb["reason"]}<br>つぎはレベル {fb["next_level"]} のやさしい盤面で確認しよう。</div>',
                unsafe_allow_html=True,
            )
        if st.button(f"つぎの問題へ　レベル {fb['next_level']}", use_container_width=True, type="primary"):
            new_level = fb["next_level"]
            st.session_state.challenge_level = new_level
            st.session_state.challenge_variant += 1
            new_challenge(new_level, st.session_state.challenge_variant)
            st.rerun()

    st.caption(f"これまで：{st.session_state.challenge_correct} / {st.session_state.challenge_total} 正解")
    with st.expander("レベルを選び直す", expanded=False):
        manual_level = st.selectbox(
            "開始レベル",
            list(PRACTICE_LEVELS.keys()),
            index=level - 1,
            format_func=lambda x: PRACTICE_LEVELS[x]["name"],
            key="challenge_level_manual",
        )
        if st.button("このレベルからやり直す", use_container_width=True):
            st.session_state.challenge_variant += 1
            new_challenge(manual_level, st.session_state.challenge_variant)
            st.rerun()


# -----------------------------
# UI: 学習（授業一覧）
# -----------------------------
elif st.session_state.page == "learn":
    if st.button("← ホームへ"):
        go("home")

    st.markdown("### 📘 オセロを学ぶ")
    st.caption("1つずつ授業をえらんで、初歩から上級まで順番に進めます。定石は『丸暗記』ではなく、盤の形を見比べながら学びます。")

    learn_c1, learn_c2 = st.columns(2)
    if learn_c1.button("🧪 テスト盤面", use_container_width=True, key="learn_to_experiment"):
        go("experiment")
    if learn_c2.button("🎯 挑戦", use_container_width=True, key="learn_to_challenge"):
        go("challenge")

    completed = set(st.session_state.completed_lessons)
    total = len(LESSONS)
    done = len(completed)
    st.progress(done / total if total else 0.0)
    st.caption(f"すすみぐあい：{done} / {total} 授業")

    for i, stage in enumerate(LESSON_STAGES):
        stage_done = sum(1 for lesson_id in stage["lesson_ids"] if lesson_id in completed)
        expanded = (i == 0 and done == 0) or (0 < stage_done < len(stage["lesson_ids"]))
        with st.expander(
            f"{stage['title']}　{stage_done}/{len(stage['lesson_ids'])}",
            expanded=expanded,
        ):
            st.caption(f"{stage['level']}｜{stage['short']}")
            for lesson_id in stage["lesson_ids"]:
                lesson = LESSONS[lesson_id]
                mark = "✅" if lesson_id in completed else "▶"
                if st.button(
                    f"{mark} {lesson_id}. {lesson['title']}",
                    use_container_width=True,
                    key=f"open_lesson_{lesson_id}",
                ):
                    st.session_state.current_lesson = lesson_id
                    go("lesson")

    st.markdown("---")
    st.caption("※ 定石名は代表的なオセロのオープニング名称です。『この順なら必ず勝つ』という意味ではありません。")
    if st.button("📷 盤面を撮って実戦で試す", use_container_width=True, type="primary"):
        go("home")


# -----------------------------
# UI: 1つの授業
# -----------------------------
elif st.session_state.page == "lesson":
    lesson_id = st.session_state.get("current_lesson", "01")
    lesson = LESSONS.get(lesson_id, LESSONS["01"])
    stage = stage_for_lesson(lesson_id)

    top1, top2 = st.columns(2)
    if top1.button("← 授業一覧", use_container_width=True):
        go("learn")
    if top2.button("📷 実戦へ", use_container_width=True):
        go("home")

    header_html = (
        '<div class="lesson-head">'
        f'<div class="lesson-stage">{stage["title"]}｜授業 {lesson_id}</div>'
        f'<div class="lesson-title">{lesson["title"]}</div>'
        f'<div class="lesson-goal"><b>きょうのゴール：</b>{lesson["goal"]}</div>'
        '</div>'
    )
    st.markdown(header_html, unsafe_allow_html=True)

    st.markdown("#### せんせいの説明")
    for line in lesson["teach"]:
        st.markdown(f"- {line}")

    if lesson.get("sequence"):
        lesson_board, last = board_from_sequence(lesson["sequence"])
        seq_text = lesson.get("sequence_name", " → ".join(lesson["sequence"]))
        st.markdown(f'<div class="lesson-seq">{seq_text}</div>', unsafe_allow_html=True)
        st.image(render_teaching_board(lesson_board, focus=[last] if last else None), use_container_width=True)
        st.caption("黄色のマスが、この手順で最後に打った場所です。黒から交互に打っています。")
    elif lesson.get("highlight"):
        h = lesson["highlight"]
        st.image(
            render_teaching_board(
                initial_board(),
                focus=h.get("focus"),
                danger=h.get("danger"),
                safe=h.get("safe"),
            ),
            use_container_width=True,
        )
        legend = []
        if h.get("safe"):
            legend.append("緑＝大事・安全の例")
        if h.get("danger"):
            legend.append("赤＝注意する場所")
        if h.get("focus"):
            legend.append("黄＝見てほしい場所")
        if legend:
            st.caption(" / ".join(legend))
    elif lesson.get("board") == "initial":
        st.image(render_board(initial_board(), legal=legal_moves(initial_board(), BLACK).keys()), use_container_width=True)
        st.caption("黄色い点が、黒が最初における4か所です。")

    st.markdown(f'<div class="lesson-point"><b>これだけ覚える：</b><br>{lesson["remember"]}</div>', unsafe_allow_html=True)

    st.markdown("#### やってみよう")
    quiz = lesson["quiz"]
    choice = st.radio(
        quiz["q"],
        quiz["options"],
        index=None,
        key=f"quiz_choice_{lesson_id}",
    )
    show_key = f"quiz_show_{lesson_id}"
    if show_key not in st.session_state:
        st.session_state[show_key] = False

    if st.button("こたえを見る", use_container_width=True, key=f"quiz_btn_{lesson_id}"):
        st.session_state[show_key] = True

    if st.session_state[show_key]:
        if choice is None:
            st.info("まず、こたえを1つえらんでね。")
        elif choice == quiz["answer"]:
            st.success(f"せいかい！　{quiz['why']}")
        else:
            st.warning(f"もう一度見てみよう。こたえは『{quiz['answer']}』。{quiz['why']}")

    st.markdown("---")
    next_id = next_lesson_id(lesson_id)
    finish_label = "✅ この授業をおわる"
    if lesson_id in st.session_state.completed_lessons:
        finish_label = "✅ この授業はクリアずみ"

    if st.button(finish_label, use_container_width=True, type="primary", key=f"finish_{lesson_id}"):
        if lesson_id not in st.session_state.completed_lessons:
            st.session_state.completed_lessons.append(lesson_id)
        if next_id:
            st.session_state.current_lesson = next_id
            st.rerun()
        else:
            st.success("全20授業クリア！　実戦の盤面で、考え方を使ってみよう。")

    if next_id:
        if st.button(f"次の授業へ　→ {next_id}. {LESSONS[next_id]['title']}", use_container_width=True, key=f"next_{lesson_id}"):
            st.session_state.current_lesson = next_id
            st.rerun()

