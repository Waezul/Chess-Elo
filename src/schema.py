"""Shared column names and engine thresholds.

Identifiers and the result are stored for joins, splits, and an ablation.
They are not default model features. Player names, opponent Elo, rating
diffs, and site URLs are never written.
"""

ELO_MIN = 800  # drop ratings below this so the label band stays comparable
ELO_MAX = 2400  # drop ratings above this for the same reason
TERMINATION = "Normal"  # keep finished games; forfeits cut the move record short
DEFAULT_POOL = "classical"  # one rating pool; blitz Elo is not classical Elo
DEFAULT_MIN_GAMES = 10  # legacy player rollup only; the site sample uses a per-bin cap
DEFAULT_TEST_SIZE = 0.2  # fraction of players held out for the final score
DEFAULT_VAL_SIZE = 0.1  # fraction of players used only to early-stop
DEFAULT_SEED = 42  # makes the player split repeatable

# Site sample. Eight equal-width bins, tails filled before the crowded middle.
RATING_EDGES = list(range(ELO_MIN, ELO_MAX + 1, 200))
BIN_LABELS = [f"{edge}-{edge + 200}" for edge in RATING_EDGES[:-1]]
TAIL_BIN_INDEXES = {0, 1, 6, 7}  # 800-1200 and 2000-2400
GAMES_PER_BIN = 4000  # side-rows, not unique games
PER_PLAYER_CAP = 4  # games from one player inside one bin
THIN_BIN_PLAYERS = 500  # a tail bin below this keeps every game and is marked thin
# Player model. Eight games from each of 700 players per bin, then one averaged row.
PLAYERS_PER_BIN = 700
GAMES_PER_PLAYER = 8
PLAYER_MACRO_MAE_MAX = 120.0  # test macro MAE at or under this can ship
PLAYER_TAIL_BIAS_MAX = 80.0  # absolute tail bias at or under this can ship
PLAYER_WIDTH_MAX = 350.0  # typical 10-90% width at or under this can ship
PLAYER_BIN_COVERAGE_MIN = 0.70  # every reported bin must cover at least this share
COVERAGE_TARGET = 0.8  # 10-90% interval should cover about this share of held-out games
QUANTILE_ALPHAS = (0.1, 0.5, 0.9)
MACRO_BIN_MIN_N = 20  # bins smaller than this stay in the report and skip early stopping

MISTAKE_CP = 100  # centipawn loss that counts as a mistake
BLUNDER_CP = 200  # centipawn loss that counts as a blunder
MATE_CP = 1000  # mate scores are clamped here so one mate cannot dwarf the mean
DEFAULT_DEPTH = 10  # Stockfish search depth for the quality features

POOLS = ("classical", "rapid", "blitz", "bullet")

# Per subject, per game. At player level each becomes <name>_mean and <name>_std.
NUMERIC_FEATURES = [
    "game_plies",  # how long the game lasted; style and time control both show up here
    "n_subject_moves",  # how many moves this side actually played
    "rate_pawn",  # share of this side's moves that were pawn moves
    "rate_knight",  # share that were knight moves
    "rate_bishop",  # share that were bishop moves
    "rate_rook",  # share that were rook moves
    "rate_queen",  # share that were queen moves
    "rate_king",  # share that were king moves, including castling
    "rate_capture",  # how often this side captured
    "rate_check",  # how often this side gave check
    "rate_promotion",  # how often this side promoted
    "castled",  # 1 if this side castled in the game
    "castle_kingside",  # 1 kingside, 0 queenside, empty if they never castled
    "castle_ply",  # ply of castling; early castling is a development habit
    "mat_opening_mean",  # average material from this side's view through ply 20
    "mat_middlegame_mean",  # same average after ply 20, ignoring the last 6 plies
    "developed_minors_ply10",  # knights and bishops off their start squares by ply 10
    "avg_legal_moves",  # average choices available on this side's turn
]

# Stockfish, subject moves only. Same mean/std rollup as the behavior columns.
ENGINE_FEATURES = [
    "cpl_mean",  # average centipawns lost per move; lower means more accurate
    "cpl_median",  # typical loss, less pulled by one huge blunder than the mean
    "eval_volatility",  # how much the evaluation swung across this side's moves
    "mistake_rate",  # share of moves that lost at least 100 centipawns
    "blunder_rate",  # share of moves that lost at least 200 centipawns
    "best_move_rate",  # share of moves that matched Stockfish's top move
]

# Game-level only. The player table must not learn "how often they were White".
GAME_EXTRA = ["is_white"]  # 1 when the row is White, so the game model can condition on color

# Stored for joins and the label. The model does not train on these raw columns.
# first_move is one-hot encoded; the raw string stays out of the numeric matrix.
GAME_ID_COLUMNS = [
    "game_id",  # Lichess id, used only to join rows from the same game
    "player_key",  # hash of the username, used only to group and split players
    "color",  # which side this row describes
    "elo",  # label: that side's Elo in this game
    "result_subject",  # 1 / 0.5 / 0 from this side's view; not a feature
    "eco_family",  # not a feature; for Black it is mostly White's choice
    "first_move",  # this side's first move; encoded, not used raw
]

GAME_COLUMNS = [
    "game_id",
    "player_key",
    "color",
    "elo",
    "is_white",
    "result_subject",
    "eco_family",
    "first_move",
    *NUMERIC_FEATURES,
    *ENGINE_FEATURES,
]

# Joined on after the sample. Present on games_sample.parquet, never as inputs.
SAMPLE_COLUMNS = ["split", "elo_bin", "thin_bin"]

# Lower centipawn loss and fewer errors should not raise the predicted Elo.
MONOTONE = {
    "cpl_mean": -1,
    "cpl_median": -1,
    "mistake_rate": -1,
    "blunder_rate": -1,
    "best_move_rate": 1,
}

# rapid8-v3. Per game, then the spread of four quality columns across the eight games.
BALANCED_CP = 150  # |eval before the move| at or under this counts as a balanced position
OPENING_PLY = 20  # ply of the move, 1-based; opening is this ply or earlier
LATE_PLY = 60  # moves after this ply are the late phase
INSTANT_SECONDS = 2.0  # a move after ply 20 thought for less than this is instant
LOW_CLOCK_FRAC = 0.10  # clock left under this share of the estimated game length
CALM_CLOCK_FRAC = 0.20  # a move with at least this share left is a calm move
CLOCK_DROPOUT = 0.2  # share of train rows whose clock block is blanked
TRAIN_SUBSETS = 3  # disjoint 8-game chunks taken from one train player
V3_REVISION = "rapid8-v3"
FEATURE_VERSION = 3

V3_GAME_ENGINE = [
    "wpl_mean",  # average win-percent lost on this side's moves
    "acc_mean",  # average Lichess move accuracy, 0-100
    "cpl_balanced",  # average centipawn loss from balanced positions
    "mistake_rate_balanced",  # share of balanced moves that lost at least 100 centipawns
    "cpl_opening",  # average centipawn loss on plies 1-20
    "cpl_late",  # average centipawn loss after ply 60
]
V3_OPPONENT = [
    "opp_cpl_mean",  # opponent's average centipawn loss; not a monotone feature
    "opp_wpl_mean",
    "opp_acc_mean",
    "opp_blunder_rate",
    "opp_cpl_balanced",
]
V3_CLOCK = [
    "has_clock",  # 1 only when the time control parses and every subject move has a clock
    "clk_think_frac_mean",
    "clk_think_cv",
    "clk_instant_rate",
    "clk_low_rate",
    "clk_end_frac",
    "cpl_calm",
]
V3_TC = ["tc_base", "tc_increment"]  # seconds; blank when TimeControl does not parse
V3_SPREAD_SOURCES = ["acc_mean", "wpl_mean", "cpl_mean", "blunder_rate"]
V3_SPREAD = [f"{name}_{stat}" for name in V3_SPREAD_SOURCES for stat in ("std", "min", "max")]
V3_GAME_FEATURES = [
    *GAME_EXTRA,
    *NUMERIC_FEATURES,
    *ENGINE_FEATURES,
    *V3_GAME_ENGINE,
    *V3_OPPONENT,
    *V3_CLOCK,
    *V3_TC,
]
V3_PLAYER_FEATURES = [*V3_GAME_FEATURES, *V3_SPREAD]
V3_FEATURE_SETS = {
    "v2": [*GAME_EXTRA, *NUMERIC_FEATURES, *ENGINE_FEATURES],
    "+engine": [*GAME_EXTRA, *NUMERIC_FEATURES, *ENGINE_FEATURES, *V3_GAME_ENGINE, *V3_TC],
    "+opponent": [
        *GAME_EXTRA,
        *NUMERIC_FEATURES,
        *ENGINE_FEATURES,
        *V3_GAME_ENGINE,
        *V3_TC,
        *V3_OPPONENT,
    ],
    "+clock": [
        *GAME_EXTRA,
        *NUMERIC_FEATURES,
        *ENGINE_FEATURES,
        *V3_GAME_ENGINE,
        *V3_TC,
        *V3_OPPONENT,
        *V3_CLOCK,
    ],
    "all": V3_PLAYER_FEATURES,
}
# Opponent columns stay free: a strong subject inflates the opponent's loss.
MONOTONE_V3 = {
    **MONOTONE,
    "wpl_mean": -1,
    "acc_mean": 1,
    "cpl_balanced": -1,
    "mistake_rate_balanced": -1,
    "cpl_calm": -1,
}

# Shown with the top features. Keyed by the column the model actually sees.
FEATURE_DESCRIPTIONS = {
    "is_white": "Whether this row is White, so the guess can differ by color.",
    "game_plies": "How long the game lasted, in plies.",
    "n_subject_moves": "How many moves this side played.",
    "rate_pawn": "Share of this side's moves that were pawn moves.",
    "rate_knight": "Share of this side's moves that were knight moves.",
    "rate_bishop": "Share of this side's moves that were bishop moves.",
    "rate_rook": "Share of this side's moves that were rook moves.",
    "rate_queen": "Share of this side's moves that were queen moves.",
    "rate_king": "Share of this side's moves that were king moves, including castling.",
    "rate_capture": "How often this side captured.",
    "rate_check": "How often this side gave check.",
    "rate_promotion": "How often this side promoted.",
    "castled": "1 if this side castled, 0 otherwise.",
    "castle_kingside": "1 for kingside castling, 0 for queenside, empty if they never castled.",
    "castle_ply": "The ply on which this side castled.",
    "mat_opening_mean": "Average material from this side's view through ply 20.",
    "mat_middlegame_mean": "Average material from this side's view after ply 20, ignoring the last 6 plies so the final result does not leak in. Being ahead here tends to mark the higher-rated side.",
    "developed_minors_ply10": "Knights and bishops that had left their start squares by ply 10.",
    "avg_legal_moves": "Average number of legal moves on this side's turns. Open positions have more choices; cramped or lost positions have fewer.",
    "cpl_mean": "Average centipawns given away on this side's moves. Lower means more accurate play and usually a higher rating.",
    "cpl_median": "Typical centipawn loss on this side's moves, less pulled by one huge blunder than the mean.",
    "eval_volatility": "How much the engine evaluation swung across this side's moves.",
    "mistake_rate": "Share of this side's moves that lost at least 100 centipawns (one pawn).",
    "blunder_rate": "Share of this side's moves that lost at least 200 centipawns. This was the strongest signal when many games were averaged.",
    "best_move_rate": "Share of this side's moves that matched Stockfish's top move at the training depth.",
    "wpl_mean": "Average win-percent given away on this side's moves. Equal positions lose more from the same centipawn error than positions that are already decided.",
    "acc_mean": "Average move accuracy on this side's moves, from 0 to 100. 100 would mean every move matched the engine's win-percent change.",
    "cpl_balanced": "Average centipawns given away on this side's moves that started within 150 centipawns of equal.",
    "mistake_rate_balanced": "Share of this side's balanced-position moves that lost at least 100 centipawns.",
    "cpl_opening": "Average centipawns given away on this side's moves at ply 20 or earlier.",
    "cpl_late": "Average centipawns given away on this side's moves after ply 60.",
    "opp_cpl_mean": "Average centipawns the opponent gave away. A strong subject tends to raise this number, so it is not forced up or down.",
    "opp_wpl_mean": "Average win-percent the opponent gave away on their moves.",
    "opp_acc_mean": "Average move accuracy of the opponent, from 0 to 100. This is how cleanly they played, not their rating.",
    "opp_blunder_rate": "Share of the opponent's moves that lost at least 200 centipawns.",
    "opp_cpl_balanced": "Average centipawns the opponent gave away from positions within 150 centipawns of equal.",
    "has_clock": "1 when the time control parsed and every move by this side has a clock, otherwise 0.",
    "clk_think_frac_mean": "Average think time as a fraction of the estimated game length. This side's first move is left out.",
    "clk_think_cv": "How uneven this side's think times were: the spread divided by the mean. The first move is left out.",
    "clk_instant_rate": "Share of this side's moves after ply 20 that were played in under 2 seconds.",
    "clk_low_rate": "Share of this side's moves played with under a tenth of the estimated game time left.",
    "clk_end_frac": "Clock left after this side's last move, as a fraction of the estimated game length.",
    "cpl_calm": "Average centipawns given away on moves played with at least a fifth of the estimated time still on the clock.",
    "tc_base": "Base time on the clock at the start, in seconds.",
    "tc_increment": "Seconds added after each of this side's moves.",
    "acc_mean_std": "Spread of acc_mean across this player's games in the subset.",
    "acc_mean_min": "Lowest acc_mean across this player's games in the subset.",
    "acc_mean_max": "Highest acc_mean across this player's games in the subset.",
    "wpl_mean_std": "Spread of wpl_mean across this player's games in the subset.",
    "wpl_mean_min": "Lowest wpl_mean across this player's games in the subset.",
    "wpl_mean_max": "Highest wpl_mean across this player's games in the subset.",
    "cpl_mean_std": "Spread of cpl_mean across this player's games in the subset.",
    "cpl_mean_min": "Lowest cpl_mean across this player's games in the subset.",
    "cpl_mean_max": "Highest cpl_mean across this player's games in the subset.",
    "blunder_rate_std": "Spread of blunder_rate across this player's games in the subset.",
    "blunder_rate_min": "Lowest blunder_rate across this player's games in the subset.",
    "blunder_rate_max": "Highest blunder_rate across this player's games in the subset.",
}

# Never train on these directly. n_games remains for the legacy player rollup.
NON_FEATURES = {
    "game_id",
    "player_key",
    "color",
    "elo",
    "split",
    "elo_bin",
    "thin_bin",
    "n_games",
    "result_subject",
    "result_mean",
    "eco_family",
    "first_move",
    "offset",
}
