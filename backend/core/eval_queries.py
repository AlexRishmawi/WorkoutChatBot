from backend.core.ingest import normalize_exercise_name
from backend.core.ingest import get_all_aliases
from langchain_core.documents import Document

def _exercise_match(doc: Document, exercise_query: str) -> bool:
    canonical = normalize_exercise_name(exercise_query)
    all_aliases = set(a.lower() for a in get_all_aliases(canonical))

    raw = doc.metadata.get("exercise_names", [])
    if isinstance(raw, str):
        exercise_names = [ex.strip() for ex in raw.split(",") if ex.strip()]
    else:
        exercise_names = list(raw)

    doc_exercises = set(ex.lower() for ex in exercise_names)

    if all_aliases & doc_exercises:
        return True

    return any(exercise_query.lower() in ex for ex in doc_exercises)


EVAL_DATASET = [
    {
        "query": "What did I squat in Week 1?",
        "condition": lambda doc: (
            doc.metadata.get("week") == "Week 1"
            and _exercise_match(doc, "squat")
        ),
    },
    {
        "query": "Show me my primary bench logs for week 2",
        "condition": lambda doc: (
            doc.metadata.get("week") == "Week 2"
            and _exercise_match(doc, "bench press")
        ),
    },
    {
        "query": "Did I leave any notes on deadlifts?",
        "condition": lambda doc: _exercise_match(doc, "deadlift"),
    },
    {
        "query": "What exercises did I do on Day 1 of Week 3?",
        "condition": lambda doc: (
            doc.metadata.get("week") == "Week 3"
            and "day 1" in str(doc.metadata.get("day", "")).lower()
        ),
    },
    {
        "query": "Check my barbell row weights on Week 1",
        "condition": lambda doc: (
            doc.metadata.get("week") == "Week 1"
            and _exercise_match(doc, "row")
        ),
    },
    {
        "query": "What was my leg extension weight in Week 2?",
        "condition": lambda doc: (
            doc.metadata.get("week") == "Week 2"
            and _exercise_match(doc, "extension")
        ),
    },
    {
        "query": "Show my shoulder or dumbbell press performance in Week 1",
        "condition": lambda doc: (
            doc.metadata.get("week") == "Week 1"
            and _exercise_match(doc, "press")
        ),
    },
    {
        "query": "Check my calf raises logs across the program",
        "condition": lambda doc: _exercise_match(doc, "calf"),
    },
    {
        "query": "What did I perform on Wednesday of Week 2?",
        "condition": lambda doc: (
            doc.metadata.get("week") == "Week 2"
            and "day 3" in str(doc.metadata.get("day", "")).lower()
        ),
    },
    {
        "query": "Did I do any chest supported rows in Week 3?",
        "condition": lambda doc: (
            doc.metadata.get("week") == "Week 3"
            and _exercise_match(doc, "supported row")
        ),
    },
    {
        "query": "Find my hip thrust data",
        "condition": lambda doc: _exercise_match(doc, "thrust"),
    },
    {
        "query": "Show my lat pulldown notes on neutral grip exercises",
        "condition": lambda doc: _exercise_match(doc, "pulldown"),
    },
    {
        "query": "What did I perform for dips in Week 2?",
        "condition": lambda doc: (
            doc.metadata.get("week") == "Week 2"
            and _exercise_match(doc, "dips")
        ),
    },
    {
        "query": "Show my dumbbell curl logs from Week 1",
        "condition": lambda doc: (
            doc.metadata.get("week") == "Week 1"
            and _exercise_match(doc, "curl")
        ),
    },
    {
        "query": "What exercises did I perform on Week 3 Day 2?",
        "condition": lambda doc: (
            doc.metadata.get("week") == "Week 3"
            and "day 2" in str(doc.metadata.get("day", "")).lower()
        ),
    },
    {
        "query": "Are there any rest days in Week 1?",
        "condition": lambda doc: (
            doc.metadata.get("week") == "Week 1"
            and doc.metadata.get("is_rest_day") is True
        ),
    },
    {
        "query": "Look up my tricep pushdown weight logs",
        "condition": lambda doc: _exercise_match(doc, "pushdown"),
    },
    {
        "query": "What was my bench variation on week 2?",
        "condition": lambda doc: (
            doc.metadata.get("week") == "Week 2"
            and _exercise_match(doc, "bench variation")
        ),
    },
    {
        "query": "Tell me what weights I used for lunges or split squats",
        "condition": lambda doc: (
            _exercise_match(doc, "lunge") or _exercise_match(doc, "bulgarian")
        ),
    },
    {
        "query": "Show me Week 3 rest days",
        "condition": lambda doc: (
            doc.metadata.get("week") == "Week 3"
            and doc.metadata.get("is_rest_day") is True
        ),
    }
]