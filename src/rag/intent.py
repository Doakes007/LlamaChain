def classify_query_intent(query):
    """Classify query intent for adaptive logic"""
    q = query.lower()

    if any(k in q for k in ["worst", "best", "highest", "lowest", "perform"]):
        return "performance"

    if any(k in q for k in ["how many", "number", "count", "total"]):
        return "factual"

    if any(k in q for k in ["explain", "how", "why", "steps", "process"]):
        return "explanation"

    if any(k in q for k in ["compare", "difference", "similar"]):
        return "comparison"

    if any(k in q for k in ["show", "diagram", "figure", "visualize"]):
        return "visual"

    return "general"


def detect_query_domain(query):
    q = query.lower()
    if any(k in q for k in ["cnn", "figure", "diagram", "visual"]):
        return "vision"
    if any(k in q for k in ["nlp", "text", "bert", "roberta", "language"]):
        return "nlp"
    return "general"


def detect_query_source(query):
    q = query.lower()
    if any(k in q for k in ["nlp", "text", "bert", "roberta", "language"]):
        return "nlp"
    elif any(k in q for k in ["cnn", "vision", "classification"]):
        return "cnn"
    return None