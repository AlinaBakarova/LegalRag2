"""Search-time evidence selection; does not change stored chunks or index identity."""

import re

from legal_workspace.retrieval.chunking import normalize

STOP_WORDS = set(
    normalize(
        "что такое такой как какой какие каков какова каковы чем почему когда где "
        "сколько за это этот эти тот "
        "расскажи пожалуйста объясни укажи дай определение по о об в во на и или а с со из "
        "для при к ко ли же РФ Российской Федерации КоАП УК ГК кодекс статья пункт"
    ).split()
)
ARTICLE = re.compile(r"^\s*статья\s+(\d+(?:\.\d+)*)(?:\.|\s|$)", re.I)
ARTICLE_REFERENCE = re.compile(r"\b(?:статья|статье|статьи|ст\.)\s*(\d+(?:\.\d+)*)", re.I)
STRUCTURE = re.compile(
    r"^(?:(?:статья|глава|раздел|часть)\s+[IVXLC\d]+(?:\.\d+)*[.\s].*|"
    r"(?:уголовный|гражданский|налоговый|трудовой|семейный)\s+кодекс.*|"
    r"кодекс\s+российской\s+федерации.*|российская\s+федерация|"
    r"содержание|оглавление|общая\s+часть|особенная\s+часть|"
    r"принят\s+.*|одобрен\s+.*|федеральный\s+закон.*)$",
    re.I,
)
# Protect short operative clauses even when PDF extraction puts them on the heading line.
OPERATIVE = re.compile(
    r"\b(?:является|представляет|означает|призна[её]тся|влеч[её]т|устанавливается|"
    r"составляет|назначается|налагается|обязан(?:а|о|ы)?|вправе|запрещ\w*|не\s+допускается|"
    r"должен|должна|должны|подлеж\w*|если|кроме|исключени\w*|не\s+может)\b",
    re.I,
)
ACTS = {
    "koap": re.compile(
        r"\bкоап\b|кодекс\s+российской\s+федерации\s+об\s+административных\s+правонарушениях", re.I
    ),
    "uk": re.compile(r"\bук\s*(?:рф|россии)\b|уголовн\w*\s+кодекс", re.I),
    "gk": re.compile(r"\bгк\s*(?:рф|россии)\b|гражданск\w*\s+кодекс", re.I),
}


def query_terms(query):
    return list(dict.fromkeys(t for t in normalize(query).split() if t not in STOP_WORDS))[:50]


def article_key(heading):
    match = ARTICLE.match(heading)
    return match.group(1).rstrip(".") if match else None


def structural_only(text, heading=""):
    text = text.strip()
    if not text or OPERATIVE.search(text):
        return not text
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return bool(lines) and all(STRUCTURE.fullmatch(line) for line in lines)


def evidence_priority(item, terms, query):
    # A complete topical match in BODY rescues a strong one-channel hit from RRF bias.
    body = set(normalize(item["text"]).split())
    topical = bool(terms) and set(terms).issubset(body)
    reference = ARTICLE_REFERENCE.search(query)
    reference_match = bool(
        reference and article_key(item["heading"]) == reference.group(1).rstrip(".")
    )
    definition_query = bool(re.search(r"что\s+так(?:ое|ой)|определени", query, re.I))
    definition_body = topical and bool(
        re.search(r"\b(?:является|представляет|означает|призна[её]тся)\b", item["text"], re.I)
    )
    return (reference_match, definition_query and definition_body, topical, item["score"])


def retrieval_query(question, history):
    # Only genuinely dependent follow-ups inherit the previous question.
    dependent = re.match(r"^\s*(?:а\b|и\s+если\b|тогда\b|а\s+если\b)", question, re.I)
    dependent = dependent or re.search(
        r"\b(?:это|этот|эта|эти|этого|этому|этом|этой|указанн\w*|тот\s+же)\b", question, re.I
    )
    if dependent and not any(pattern.search(question) for pattern in ACTS.values()):
        previous = [m["content"] for m in history if m["role"] == "user"][-1:]
        return (" ".join(previous)[:500] + " " + question).strip()[-1500:]
    return question
