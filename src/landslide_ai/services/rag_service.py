from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from landslide_ai.config import load_config
from landslide_ai.utils.optional_dependencies import openai_available


@dataclass(slots=True)
class RetrievalChunk:
    source_id: str
    title: str
    content: str


@dataclass(slots=True)
class RetrievalHit:
    source_id: str
    title: str
    content: str
    score: float


class LandslideRAGService:
    def __init__(self, project_root: str | Path = ".") -> None:
        self.project_root = Path(project_root)
        self.config = load_config(self.project_root)
        self.knowledge_dir = self.project_root / "data" / "knowledge"
        self._chunks = self._load_chunks()
        self._vectorizer = TfidfVectorizer(stop_words="english")
        corpus = [chunk.content for chunk in self._chunks] or [""]
        self._matrix = self._vectorizer.fit_transform(corpus)

    def retrieve(self, query: str, top_k: int = 3) -> list[RetrievalHit]:
        if not query.strip() or not self._chunks:
            return []
        query_vector = self._vectorizer.transform([query])
        similarities = cosine_similarity(query_vector, self._matrix).flatten()
        ranked_indices = similarities.argsort()[::-1]
        hits: list[RetrievalHit] = []
        for index in ranked_indices[:top_k]:
            score = float(similarities[index])
            if score <= 0.0:
                continue
            chunk = self._chunks[index]
            hits.append(
                RetrievalHit(
                    source_id=chunk.source_id,
                    title=chunk.title,
                    content=chunk.content,
                    score=score,
                )
            )
        return hits

    def summarize_for_advisory(self, query: str, top_k: int = 2) -> str:
        hits = self.retrieve(query, top_k=top_k)
        if not hits:
            return ""
        llm_answer = self._llm_answer(
            query=query,
            hits=hits,
            system_instruction=(
                "You write concise landslide operational guidance. Use only the supplied context. "
                "Keep the answer to one or two short sentences."
            ),
            max_tokens=140,
        )
        if llm_answer:
            return llm_answer
        snippets = [self._shorten(hit.content) for hit in hits]
        return " ".join(snippets)

    def answer_query(self, query: str, top_k: int = 3) -> dict[str, object]:
        hits = self.retrieve(query, top_k=top_k)
        answer = self._llm_answer(
            query=query,
            hits=hits,
            system_instruction=(
                "You answer questions about a landslide intelligence project. "
                "Use only the supplied retrieval context. If the context is insufficient, say so briefly."
            ),
            max_tokens=220,
        )
        if not answer and hits:
            answer = " ".join(self._shorten(hit.content, max_chars=220) for hit in hits[:2])
        return {
            "query": query,
            "row_count": len(hits),
            "rows": [
                {
                    "source_id": hit.source_id,
                    "title": hit.title,
                    "content": hit.content,
                    "score": round(hit.score, 4),
                }
                for hit in hits
            ],
            "answer": answer,
            "llm_enabled": self._llm_ready(),
        }

    def _load_chunks(self) -> list[RetrievalChunk]:
        if not self.knowledge_dir.exists():
            return []
        chunks: list[RetrievalChunk] = []
        for path in sorted(self.knowledge_dir.glob("*.md")):
            text = path.read_text(encoding="utf-8")
            chunks.extend(self._chunk_markdown(path.stem, text))
        return chunks

    def _llm_ready(self) -> bool:
        return (
            self.config.enable_llm
            and self.config.llm_provider.lower() == "openai"
            and bool(self.config.openai_api_key)
            and openai_available()
        )

    def _llm_answer(
        self,
        query: str,
        hits: list[RetrievalHit],
        system_instruction: str,
        max_tokens: int,
    ) -> str:
        if not hits or not self._llm_ready():
            return ""
        try:
            from openai import OpenAI

            context = "\n\n".join(
                f"[{hit.title}] {hit.content}"
                for hit in hits
            )
            client = OpenAI(api_key=self.config.openai_api_key)
            response = client.responses.create(
                model=self.config.llm_model,
                input=[
                    {"role": "system", "content": system_instruction},
                    {
                        "role": "user",
                        "content": (
                            f"Question: {query}\n\n"
                            f"Context:\n{context}\n\n"
                            "Answer using only the context."
                        ),
                    },
                ],
                max_output_tokens=max_tokens,
            )
            return getattr(response, "output_text", "").strip()
        except Exception:
            return ""

    @staticmethod
    def _chunk_markdown(source_id: str, text: str) -> list[RetrievalChunk]:
        sections = re.split(r"^##\s+", text, flags=re.MULTILINE)
        chunks: list[RetrievalChunk] = []
        for section in sections:
            cleaned = section.strip()
            if not cleaned:
                continue
            lines = cleaned.splitlines()
            title = lines[0].lstrip("# ").strip()
            content = " ".join(line.strip() for line in lines[1:] if line.strip())
            if content:
                chunks.append(RetrievalChunk(source_id=source_id, title=title, content=content))
        return chunks

    @staticmethod
    def _shorten(text: str, max_chars: int = 180) -> str:
        if len(text) <= max_chars:
            return text
        shortened = text[: max_chars - 3].rsplit(" ", 1)[0].strip()
        return f"{shortened}..."


@lru_cache(maxsize=4)
def get_rag_service(project_root: str | Path = ".") -> LandslideRAGService:
    return LandslideRAGService(project_root)
