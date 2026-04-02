from .rerank import get_ce_score
from .intent import classify_query_intent


def prioritize_conclusion_chunks(query, docs):
    """
    Use CE-based prioritization instead of keywords.
    Works on ANY dataset.
    """
    intent = classify_query_intent(query)

    if intent == "performance":
        strong = []

        for d in docs:
            content = d.page_content.lower()

            # Use CE score instead of keywords
            ce_score = get_ce_score(query, d.page_content)
            
            if ce_score > 0.7:  # Strong match
                strong.append(d)
            elif any(k in content for k in ["confusion", "misclassified"]):
                continue  # Skip noise
            else:
                pass

        if strong:
            return strong

    return docs