"""
exercise_aliases.py
-------------------
Canonical exercise name mapping for workout RAG pipeline.

Structure:
  CANONICAL_NAME -> [list of aliases / alternate spellings]

At ingest:  alias → canonical (improves embedding clustering & MRR)
At query:   expand user term → all known aliases (improves recall)
"""
from __future__ import annotations
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "core"))

import re

# ---------------------------------------------------------------------------
# Alias map  {canonical_name: [alias, alias, ...]}
# Keep canonical names Title Cased and human-readable.
# Aliases are matched case-insensitively and with flexible whitespace.
# ---------------------------------------------------------------------------

EXERCISE_ALIASES: dict[str, list[str]] = {

    # ── Hinge / Posterior Chain ─────────────────────────────────────────────
    "Romanian Deadlift": [
        "RDL", "Romanian Dead Lift", "Romanian DL", "Stiff Leg Deadlift",
        "Stiff-Leg Deadlift", "SLDL",
    ],
    "Deadlift": [
        "Conventional Deadlift", "Dead Lift", "DL", "Barbell Deadlift",
        "Conv Deadlift",
    ],
    "Sumo Deadlift": [
        "Sumo DL", "Sumo Dead Lift",
    ],
    "Good Morning": [
        "Good Mornings", "GM",
    ],
    "Hip Hinge": [
        "Hip Hinge Drill",
    ],

    # ── Squat Pattern ───────────────────────────────────────────────────────
    "Back Squat": [
        "Squat", "Barbell Squat", "BB Squat", "High Bar Squat",
        "Low Bar Squat", "High-Bar Squat", "Low-Bar Squat",
    ],
    "Front Squat": [
        "Front Squats", "FS",
    ],
    "Goblet Squat": [
        "Goblet Squats", "KB Goblet Squat", "DB Goblet Squat",
    ],
    "Bulgarian Split Squat": [
        "BSS", "Bulgarian SS", "RFESS", "Rear Foot Elevated Split Squat",
        "Split Squat",
    ],
    "Hack Squat": [
        "Hack Squats", "Machine Hack Squat",
    ],
    "Leg Press": [
        "45 Degree Leg Press", "Horizontal Leg Press",
    ],
    "Pendulum Squat": [
        "Pendulum Squats",
    ],

    # ── Knee Extension / Isolation ──────────────────────────────────────────
    "Leg Extension": [
        "Leg Extensions", "Knee Extension", "Knee Extensions",
        "Machine Leg Extension",
    ],
    "Leg Curl": [
        "Leg Curls", "Lying Leg Curl", "Seated Leg Curl",
        "Hamstring Curl", "Ham Curl",
    ],
    "Nordic Hamstring Curl": [
        "Nordic Curl", "Nordic Curls", "Nordic Ham Curl",
    ],

    # ── Hip Abduction / Adduction ────────────────────────────────────────────
    "Hip Abduction": [
        "Hip Abductor", "Abductor Machine", "Machine Hip Abduction",
        "Side Lying Abduction",
    ],
    "Hip Adduction": [
        "Hip Adductor", "Adductor Machine", "Machine Hip Adduction",
        "Inner Thigh Machine",
    ],

    # ── Glute / Hip Extension ────────────────────────────────────────────────
    "Hip Thrust": [
        "Hip Thrusts", "Barbell Hip Thrust", "BB Hip Thrust",
        "Glute Bridge", "Barbell Glute Bridge",
    ],
    "Cable Pull Through": [
        "Pull Through", "Cable Pullthrough",
    ],
    "Glute Kickback": [
        "Cable Kickback", "Donkey Kick",
    ],

    # ── Calf ────────────────────────────────────────────────────────────────
    "Calf Raise": [
        "Calf Raises", "Standing Calf Raise", "Seated Calf Raise",
        "Machine Calf Raise", "Smith Machine Calf Raise",
        "Leg Press Calf Raise",
    ],

    # ── Horizontal Push ─────────────────────────────────────────────────────
    "Bench Press": [
        "Chest Press", "Barbell Bench Press", "BB Bench Press",
        "Flat Bench Press", "Flat Bench", "Barbell Bench",
    ],
    "Incline Bench Press": [
        "Incline Press", "Incline Barbell Press", "Incline BB Press",
        "Incline Chest Press",
    ],
    "Decline Bench Press": [
        "Decline Press", "Decline Barbell Press",
    ],
    "Dumbbell Bench Press": [
        "DB Bench Press", "DB Chest Press", "Dumbbell Press",
        "DB Flat Press",
    ],
    "Incline Dumbbell Press": [
        "Incline DB Press", "Incline Dumbbell Bench Press",
        "Incline DB Chest Press",
    ],
    "Machine Chest Press": [
        "Chest Press Machine", "Hammer Strength Chest Press",
        "Machine Press",
    ],
    "Push Up": [
        "Push-Up", "Pushup", "Push Ups", "Push-Ups",
    ],

    # ── Vertical Push ───────────────────────────────────────────────────────
    "Overhead Press": [
        "OHP", "Shoulder Press", "Military Press", "Barbell Overhead Press",
        "Barbell OHP", "BB OHP", "Strict Press",
    ],
    "Dumbbell Overhead Press": [
        "DB Overhead Press", "DB Shoulder Press", "DB OHP",
        "Arnold Press", "Seated DB Press",
    ],
    "Machine Shoulder Press": [
        "Machine OHP", "Shoulder Press Machine",
    ],
    "Lateral Raise": [
        "Lateral Raises", "Side Lateral Raise", "Side Raise",
        "DB Lateral Raise", "Cable Lateral Raise",
    ],
    "Front Raise": [
        "Front Raises", "DB Front Raise",
    ],
    "Face Pull": [
        "Face Pulls", "Cable Face Pull",
    ],
    "Rear Delt Fly": [
        "Rear Delt Flye", "Reverse Fly", "Reverse Flye", "Rear Fly",
        "Bent Over Reverse Fly", "Pec Deck Reverse Fly",
    ],

    # ── Chest Fly ───────────────────────────────────────────────────────────
    "Cable Fly": [
        "Cable Flye", "Cable Chest Fly", "Cable Crossover",
        "Low Cable Fly", "High Cable Fly",
    ],
    "Dumbbell Fly": [
        "DB Fly", "DB Flye", "Dumbbell Flye", "Chest Fly",
        "Flat DB Fly",
    ],
    "Pec Deck": [
        "Pec Deck Fly", "Machine Fly", "Machine Chest Fly",
    ],

    # ── Vertical Pull ───────────────────────────────────────────────────────
    "Pull Up": [
        "Pull-Up", "Pullup", "Pull Ups", "Pull-Ups",
        "Weighted Pull Up", "Assisted Pull Up",
    ],
    "Chin Up": [
        "Chin-Up", "Chinup", "Chin Ups", "Chin-Ups",
        "Assisted Chin Up",
    ],
    "Lat Pulldown": [
        "Lat Pull Down", "Pulldown", "Cable Pulldown",
        "Wide Grip Pulldown", "Close Grip Pulldown",
    ],
    "Straight Arm Pulldown": [
        "Straight Arm Pull Down", "Stiff Arm Pulldown",
        "Cable Straight Arm Pulldown",
    ],

    # ── Horizontal Pull ─────────────────────────────────────────────────────
    "Barbell Row": [
        "BB Row", "Bent Over Row", "Bent-Over Row",
        "Barbell Bent Over Row", "Pendlay Row",
    ],
    "Dumbbell Row": [
        "DB Row", "One Arm DB Row", "Single Arm DB Row",
        "Single Arm Row", "Kroc Row",
    ],
    "Cable Row": [
        "Seated Cable Row", "Low Cable Row", "Machine Row",
        "Close Grip Cable Row",
    ],
    "Machine Row": [
        "Hammer Strength Row", "Chest Supported Row",
        "Chest Supported Machine Row",
    ],
    "T-Bar Row": [
        "T Bar Row", "Landmine Row",
    ],
    "Inverted Row": [
        "Bodyweight Row", "TRX Row",
    ],

    # ── Biceps ──────────────────────────────────────────────────────────────
    "Barbell Curl": [
        "BB Curl", "EZ Bar Curl", "EZ Curl",
    ],
    "Dumbbell Curl": [
        "DB Curl", "Alternating Curl", "Hammer Curl",
        "Incline DB Curl",
    ],
    "Cable Curl": [
        "Cable Bicep Curl", "Rope Curl",
    ],
    "Preacher Curl": [
        "Preacher Curls", "Machine Preacher Curl",
    ],

    # ── Triceps ─────────────────────────────────────────────────────────────
    "Tricep Pushdown": [
        "Triceps Pushdown", "Cable Pushdown", "Rope Pushdown",
        "Tricep Press Down", "Rope Tricep Pushdown",
    ],
    "Tricep Overhead Extension": [
        "Overhead Tricep Extension", "French Press",
        "EZ Bar Overhead Extension", "DB Overhead Tricep Extension",
        "Cable Overhead Tricep Extension",
    ],
    "Skull Crusher": [
        "Skull Crushers", "Lying Tricep Extension",
        "EZ Bar Skull Crusher",
    ],
    "Tricep Dip": [
        "Dip", "Dips", "Parallel Bar Dip", "Weighted Dip",
        "Bench Dip",
    ],
    "Close Grip Bench Press": [
        "Close-Grip Bench Press", "CGBP",
    ],

    # ── Core ────────────────────────────────────────────────────────────────
    "Plank": [
        "Planks", "Front Plank",
    ],
    "Ab Wheel Rollout": [
        "Ab Rollout", "Ab Wheel", "Wheel Rollout",
    ],
    "Cable Crunch": [
        "Rope Crunch", "Kneeling Cable Crunch",
    ],
    "Hanging Leg Raise": [
        "Hanging Knee Raise", "Leg Raise",
    ],
    "Sit Up": [
        "Sit-Up", "Situp", "Sit Ups", "Crunch",
    ],
    "Pallof Press": [
        "Pallof Presses", "Cable Pallof Press",
    ],

    # ── Carries / Loaded Conditioning ───────────────────────────────────────
    "Farmer's Carry": [
        "Farmers Carry", "Farmer Walk", "Farmers Walk",
        "DB Carry",
    ],

    # ── Olympic / Power ─────────────────────────────────────────────────────
    "Power Clean": [
        "Clean", "Barbell Clean",
    ],
    "Hang Clean": [
        "Hang Power Clean",
    ],
    "Snatch": [
        "Barbell Snatch", "Power Snatch",
    ],
}


# ---------------------------------------------------------------------------
# Build reverse lookup: alias (lower) → canonical
# ---------------------------------------------------------------------------

def _build_reverse_map(alias_map: dict[str, list[str]]) -> dict[str, str]:
    reverse: dict[str, str] = {}
    for canonical, aliases in alias_map.items():
        # canonical maps to itself
        reverse[_normalize_key(canonical)] = canonical
        for alias in aliases:
            key = _normalize_key(alias)
            if key in reverse and reverse[key] != canonical:
                raise ValueError(
                    f"Alias conflict: '{alias}' maps to both "
                    f"'{reverse[key]}' and '{canonical}'"
                )
            reverse[key] = canonical
    return reverse


def _normalize_key(name: str) -> str:
    """Lower-case, collapse whitespace, strip punctuation for matching."""
    name = name.lower().strip()
    name = re.sub(r"[\-_/]", " ", name)   # hyphens / slashes → space
    name = re.sub(r"[^a-z0-9 ]", "", name) # drop remaining punctuation
    name = re.sub(r"\s+", " ", name).strip()
    return name


# Eagerly built at import time — fast O(1) lookups at runtime
_REVERSE_MAP: dict[str, str] = _build_reverse_map(EXERCISE_ALIASES)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def normalize_exercise_name(name: str) -> str:
    """
    Map an exercise name to its canonical form.
    Returns the canonical name if found, otherwise returns the original
    name (title-cased) so unknown exercises are still clean.

    Examples
    --------
    >>> normalize_exercise_name("RDL")
    'Romanian Deadlift'
    >>> normalize_exercise_name("bench press")
    'Bench Press'
    >>> normalize_exercise_name("some new machine")
    'Some New Machine'
    """
    canonical = _REVERSE_MAP.get(_normalize_key(name))
    return canonical if canonical is not None else name.strip().title()


def expand_query_aliases(query: str) -> list[str]:
    """
    Given a user query string, return a list of all synonyms for any
    recognized exercise name found within it.  Used at retrieval time
    to broaden search coverage.

    Returns a list of alternative phrasings (may be empty if no alias
    matches are found).  The caller can append these to the original
    query or use them as additional search terms.

    Example
    -------
    >>> expand_query_aliases("How much did I bench press last week?")
    ['Bench Press', 'Chest Press', 'Barbell Bench Press', ...]
    """
    query_key = _normalize_key(query)
    expansions: list[str] = []

    for alias_key, canonical in _REVERSE_MAP.items():
        # Check if any known alias appears as a substring of the query
        if alias_key in query_key:
            all_terms = [canonical] + EXERCISE_ALIASES.get(canonical, [])
            for term in all_terms:
                if term not in expansions:
                    expansions.append(term)

    return expansions


def get_all_aliases(canonical_or_alias: str) -> list[str]:
    """
    Return every known name for an exercise (canonical + all aliases).
    Useful for building metadata filter sets.

    >>> get_all_aliases("RDL")
    ['Romanian Deadlift', 'RDL', 'Romanian Dead Lift', ...]
    """
    canonical = normalize_exercise_name(canonical_or_alias)
    aliases = EXERCISE_ALIASES.get(canonical, [])
    return [canonical] + aliases