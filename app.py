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
/* Streamlit の固定ツールバーに本文が潜り込まないよう、上側に余白を確保する。 */
.block-container {
    max-width: 760px;
    padding-top: 4.25rem !important;
    padding-bottom: 4rem;
}
.big-title {
    text-align:center;
    font-size:2rem;
    font-weight:800;
    line-height:1.25;
    margin:.35rem 0 .25rem;
}
.sub {text-align:center; color:#777; margin-bottom:1.35rem; line-height:1.55;}
.result-card {
    border:2px solid rgba(128,128,128,.55);
    border-radius:18px;
    padding:16px;
    margin:10px 0;
    background:rgba(255,255,255,.06);
}
.result-main {font-size:1.45rem; font-weight:800; text-align:center;}
.kid-text {font-size:1.15rem; line-height:1.7; text-align:center;}
.small-note {font-size:.88rem; color:#777;}
div.stButton > button {border-radius:14px; min-height:50px; font-weight:700;}

/* 子どもが使う主要操作は少し大きめにする。 */
[data-testid="stCameraInput"] {margin-top:.35rem;}
[data-testid="stFileUploader"] {margin-top:.25rem;}

@media (max-width: 768px) {
    .block-container {
        padding-top: calc(5.75rem + env(safe-area-inset-top)) !important;
        padding-left: 1rem !important;
        padding-right: 1rem !important;
    }
    .big-title {
        font-size:1.8rem;
        margin-top:.5rem;
    }
    .sub {
        font-size:.98rem;
        margin-bottom:1.15rem;
    }
    div.stButton > button {
        min-height:54px;
        font-size:1.02rem;
    }
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
LESSONS = {
    "レベル1 はじめて": [
        ("① まずは『おける場所』を見つけよう", "じぶんの石で、あいての石を はさめるところに おけるよ。たて・よこ・ななめを見よう。"),
        ("② かどは とても強い", "4つの かどは、いちど取ると ひっくり返されないよ。かどが取れるなら、大きなチャンス。"),
        ("③ たくさん取ればいい、ではない", "はじめのころは、いっぺんに たくさん取るより、つぎに動きやすい手が強いことがあるよ。"),
        ("④ かどのすぐ近くは気をつける", "かどが空いているとき、そのすぐ近くにおくと、あいてに かどをあげてしまうことがあるよ。"),
    ],
    "レベル2 つよくなる": [
        ("① あいての『おける場所』をへらす", "あいてが たくさんの場所におけると、好きな手をえらばれやすいよ。おける場所を少なくすると戦いやすい。"),
        ("② はしは強いけれど、いつでも安全ではない", "はしの石は動きにくいけれど、かどが空いていると危ないこともあるよ。かどとのつながりを見よう。"),
        ("③ 外に出すぎない", "空いているマスのとなりに、自分の石がたくさん出ると、あいてにひっくり返されやすくなるよ。"),
        ("④ 次の次を考える", "『ここにおく→あいてがここ→そのあと自分は？』まで考えると、ぐっと強くなるよ。"),
    ],
    "レベル3 もっと強く": [
        ("① 静かな手を使う", "取る石が少なくても、あいての選べる手をへらせるなら、とても強い手になるよ。"),
        ("② 安定した石を増やす", "かどからつながった石など、もう返されない石を少しずつ増やそう。"),
        ("③ 終盤は石の数が大事になる", "さいごが近くなったら、途中の形だけでなく、最後に何個のこるかを数えることが大切だよ。"),
        ("④ 手番の順番も考える", "空きマスが少なくなると、『どちらが最後におくか』が大きく効くことがあるよ。"),
    ],
}


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


def go(page):
    st.session_state.page = page
    st.rerun()


# -----------------------------
# UI: ホーム
# -----------------------------
st.markdown('<div class="big-title">⚫ オセロせんせい ⚪</div>', unsafe_allow_html=True)
st.markdown('<div class="sub">しゃしんを とったら、おすすめの1手をいっしょに考えるよ</div>', unsafe_allow_html=True)

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
    if st.button("📘 オセロを学ぶ", use_container_width=True, type="secondary"):
        go("learn")

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
    if c2.button("📘 オセロを学ぶ", use_container_width=True):
        go("learn")


# -----------------------------
# UI: 学習
# -----------------------------
elif st.session_state.page == "learn":
    if st.button("← もどる"):
        go("home")

    st.markdown("### 📘 オセロを学ぶ")
    st.write("むずかしさを えらんでね。")
    level = st.radio("レベル", list(LESSONS.keys()), horizontal=False, label_visibility="collapsed")

    for title, body in LESSONS[level]:
        with st.expander(title, expanded=True):
            st.markdown(f"<div class='kid-text' style='text-align:left'>{body}</div>", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### きょうの おぼえかた")
    if level.startswith("レベル1"):
        st.info("① かどをさがす → ② かどの近くに気をつける → ③ たくさん取りすぎない")
    elif level.startswith("レベル2"):
        st.info("① あいてのおける場所を数える → ② 次の次まで考える → ③ はしとかどの関係を見る")
    else:
        st.info("① 静かな手 → ② 安定した石 → ③ 終盤の手順と最後の石数")

    if st.button("📷 盤面を撮ってやってみる", use_container_width=True, type="primary"):
        go("home")
