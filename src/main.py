"""
Interactive song recommendations, with the original scoring examples via --demo.
"""

import argparse
import textwrap
from pathlib import Path

if __package__:
    from .recommender import MODES, load_songs, match_label, recommend_songs
else:
    from recommender import MODES, load_songs, match_label, recommend_songs


PROFILES = {
    "High-Energy Pop": {
        "genre": "pop",
        "mood": "happy",
        "energy": 0.80,
        "valence": 0.82,
        "acousticness": 0.20,
    },
    "Chill Lofi": {
        "genre": "lofi",
        "mood": "chill",
        "energy": 0.38,
        "valence": 0.58,
        "acousticness": 0.80,
    },
    "Throwback Acoustic": {
        "genre": "folk",
        "mood": "peaceful",
        "energy": 0.35,
        "acousticness": 0.90,
        "mood_tags": ["nostalgic", "warm"],
        "release_decade": 1990,
        "popularity": 45,
        "instrumentalness": 0.2,
        "language": "en",
    },
    "Deep Intense Rock": {
        "genre": "rock",
        "mood": "intense",
        "energy": 0.92,
        "valence": 0.40,
        "acousticness": 0.10,
    },
}

# Edge cases built to trick the scorer: conflicting, impossible, unknown or out-of-range prefs.
ADVERSARIAL_PROFILES = {
    "Conflict: high energy + sad": {
        "genre": "pop", "mood": "sad", "energy": 0.9, "valence": 0.1, "acousticness": 0.5,
    },
    "Impossible: loud + fully acoustic": {
        "genre": "metal", "mood": "peaceful", "energy": 1.0, "valence": 0.0, "acousticness": 1.0,
    },
    "Unknown genre and mood": {
        "genre": "polka", "mood": "nostalgic", "energy": 0.5, "valence": 0.5, "acousticness": 0.5,
    },
    "Wrong case and whitespace": {
        "genre": "Pop ", "mood": "Happy", "energy": 0.8, "valence": 0.8, "acousticness": 0.2,
    },
    "Out of range energy (5)": {
        "genre": "pop", "mood": "happy", "energy": 5, "valence": 0.8, "acousticness": 0.2,
    },
    "Only genre given": {"genre": "jazz"},
    "Dead center (0.5 everywhere)": {"energy": 0.5, "valence": 0.5, "acousticness": 0.5},
}


REASON_WIDTH = 44


def format_table(recommendations: list) -> str:
    """Builds an ASCII table: rank, song, artist, overall match, and the reasons."""
    rows = []
    for rank, (song, score, explanation) in enumerate(recommendations, start=1):
        reasons = [
            line
            for reason in explanation.split("; ")
            for line in textwrap.wrap(reason, REASON_WIDTH) or [""]
        ]
        rows.append([str(rank), song["title"], song["artist"], match_label(score), reasons])
    headers = ["#", "Title", "Artist", "Match", "Reasons"]
    widths = [
        max(len(headers[c]), *(len(r[c]) for r in rows)) if rows else len(headers[c])
        for c in range(4)
    ] + [REASON_WIDTH]
    line = "+" + "+".join("-" * (w + 2) for w in widths) + "+"

    def cells(values):
        return "| " + " | ".join(v.ljust(w) for v, w in zip(values, widths)) + " |"

    out = [line, cells(headers), line]
    for rank, title, artist, label, reasons in rows:
        for n, reason in enumerate(reasons):
            out.append(cells([rank, title, artist, label, reason] if n == 0 else ["", "", "", "", reason]))
        out.append(line)
    return "\n".join(out)


def print_recommendations(name: str, recommendations: list) -> None:
    """Prints one profile's ranked results as a table."""
    print(f"\n{name}")
    print(format_table(recommendations))
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Music Recommender Simulation")
    parser.add_argument("--demo", action="store_true", help="run the original scoring examples")
    parser.add_argument("--evaluate", action="store_true", help="compare retrieval quality without calling Gemini")
    parser.add_argument("--mode", choices=MODES, help="run scoring examples with this mode (default: balanced)")
    parser.add_argument("--adversarial", action="store_true", help="run the edge-case profiles")
    parser.add_argument("--no-diversity", action="store_true", help="turn off the diversity penalty")
    args = parser.parse_args()

    if args.evaluate:
        if args.demo or args.adversarial or args.no_diversity or args.mode:
            parser.error("--evaluate cannot be combined with scoring demo options")
        if __package__:
            from .evaluate import main as evaluation_main
        else:
            # Allow the documented direct-file entry point as well as -m.
            import sys
            sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
            from src.evaluate import main as evaluation_main
        return evaluation_main()

    if not (args.demo or args.adversarial or args.no_diversity or args.mode):
        if __package__:
            from .embeddings import main as rag_main
        else:
            from embeddings import main as rag_main
        return rag_main()

    songs = load_songs(str(Path(__file__).resolve().parent.parent / "data" / "songs.csv"))
    profiles = ADVERSARIAL_PROFILES if args.adversarial else PROFILES
    mode = args.mode or "balanced"
    print(f"Scoring mode: {mode}")
    for name, user_prefs in profiles.items():
        recommendations = recommend_songs(
            user_prefs, songs, k=5, mode=mode, diversity=not args.no_diversity
        )
        print_recommendations(name, recommendations)


if __name__ == "__main__":
    raise SystemExit(main())
