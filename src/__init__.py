"""Estimate classical Elo from chess games.

One game, then train in Colab:

    python -m src.pipeline --input games.pgn --output-dir data
    notebooks/train_site.ipynb

Eight games from one player, using the header index already on disk:

    python -m src.player_pipeline --input games.pgn --output-dir data --workers 4
    notebooks/train_player8.ipynb
"""
