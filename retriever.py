"""Private guest-list RAG for Alfred.

Loads the `agents-course/unit3-invitees` dataset, turns each guest into a single
text document, and exposes a BM25 keyword retriever as a smolagents Tool.
BM25 needs no embeddings or GPU, which suits a small private guest list.
"""

import re

from datasets import load_dataset
from rank_bm25 import BM25Okapi
from smolagents import Tool

DATASET_ID = "agents-course/unit3-invitees"


def load_guest_documents() -> list[str]:
    """Return one text document per guest, combining all of their fields."""
    guests = load_dataset(DATASET_ID, split="train")
    return [
        "\n".join(
            [
                f"Name: {guest['name']}",
                f"Relation: {guest['relation']}",
                f"Description: {guest['description']}",
                f"Email: {guest['email']}",
            ]
        )
        for guest in guests
    ]


def _tokenize(text: str) -> list[str]:
    """Lowercase word tokens; good enough for BM25 over short bios."""
    return re.findall(r"\w+", text.lower())


class GuestInfoRetrieverTool(Tool):
    name = "guest_info_retriever"
    description = (
        "Retrieves detailed information about gala guests (name, relation to the host, "
        "biography and interests, email) from the private guest list. "
        "Search by name, relation, or a topic associated with the guest. "
        "Returns plain text (not a dict): one block per matching guest with "
        "'Name:', 'Relation:', 'Description:' and 'Email:' lines."
    )
    inputs = {
        "query": {
            "type": "string",
            "description": "The name, relation, or topic of the guest you want information about.",
        }
    }
    output_type = "string"

    def __init__(self, docs: list[str] | None = None, top_k: int = 3, **kwargs):
        super().__init__(**kwargs)
        self.docs = docs if docs is not None else load_guest_documents()
        self.bm25 = BM25Okapi([_tokenize(doc) for doc in self.docs])
        self.top_k = top_k

    def forward(self, query: str) -> str:
        scores = self.bm25.get_scores(_tokenize(query))
        ranked = sorted(range(len(self.docs)), key=lambda i: scores[i], reverse=True)
        # Only keep documents that actually share terms with the query.
        hits = [self.docs[i] for i in ranked[: self.top_k] if scores[i] > 0]
        if not hits:
            return "No matching guest information found."
        return "\n\n".join(hits)


if __name__ == "__main__":
    # Quick manual check: python retriever.py
    tool = GuestInfoRetrieverTool()
    print(tool("Lady Ada Lovelace"))
