from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


@dataclass(frozen=True)
class DocumentChunk:
    source: str
    text: str


@dataclass(frozen=True)
class RetrievedChunk:
    source: str
    text: str
    score: float


class TransportationRAG:
    def __init__(self, docs_dir: str | Path, chunk_size: int = 900, chunk_overlap: int = 150):
        self.docs_dir = Path(docs_dir)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.chunks = self._load_chunks()

        if not self.chunks:
            raise ValueError(f"No .txt documents found in {self.docs_dir}")

        self.vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
        self.matrix = self.vectorizer.fit_transform(chunk.text for chunk in self.chunks)

    def retrieve(self, query: str, top_k: int = 3) -> list[RetrievedChunk]:
        query_vector = self.vectorizer.transform([query])
        scores = cosine_similarity(query_vector, self.matrix).flatten()
        ranked_indices = scores.argsort()[::-1][:top_k]

        return [
            RetrievedChunk(
                source=self.chunks[index].source,
                text=self.chunks[index].text,
                score=float(scores[index]),
            )
            for index in ranked_indices
            if scores[index] > 0
        ]

    def build_prompt(self, question: str, retrieved: Iterable[RetrievedChunk]) -> str:
        context = "\n\n".join(
            f"Source: {chunk.source}\nContent: {chunk.text}" for chunk in retrieved
        )

        return f"""You are a helpful transportation and logistics assistant.
Use only the provided context to answer the user's question about routes, schedules, or vehicle information.
Do not provide real-time traffic updates or safety-critical navigation unless specified in the context.
If the context is insufficient, say so clearly.

Context:
{context}

Question:
{question}

Answer:"""

    def retrieval_only_answer(self, question: str, retrieved: list[RetrievedChunk]) -> str:
        if not retrieved:
            return "No specific information found."

        best = retrieved[0]
        # Aggressive extraction for FAQ style
        lines = [line.strip() for line in best.text.splitlines() if line.strip()]
        query_words = set(question.lower().split())
        
        # Look for the absolute best matching Q: line
        for i, line in enumerate(lines):
            if line.startswith("Q:"):
                # Check for word overlap
                overlap = len(query_words.intersection(set(line.lower().split())))
                if overlap >= 3: # Require at least 3 matching words
                    if i + 1 < len(lines):
                        answer = lines[i+1]
                        if answer.startswith("A:"):
                            return answer[2:].strip()
                        return answer

        # Fallback to the first few lines if no Q: match found
        return lines[0] if lines else "Information unavailable."

    def _load_chunks(self) -> list[DocumentChunk]:
        chunks: list[DocumentChunk] = []
        for path in sorted(self.docs_dir.glob("*.txt")):
            text = path.read_text(encoding="utf-8").strip()
            for chunk in self._chunk_text(text):
                chunks.append(DocumentChunk(source=path.name, text=chunk))
        return chunks

    def _chunk_text(self, text: str) -> list[str]:
        paragraphs = [" ".join(part.split()) for part in text.splitlines() if part.strip()]
        chunks: list[str] = []
        current = ""

        for paragraph in paragraphs:
            if not current:
                current = paragraph
                continue

            candidate = f"{current}\n\n{paragraph}"
            if len(candidate) <= self.chunk_size:
                current = candidate
            else:
                chunks.append(current)
                current = paragraph

        if current:
            chunks.append(current)

        return chunks
