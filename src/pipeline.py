"""End-to-end RAG pipeline: retrieve → abstain gate → generate."""

from __future__ import annotations

from dataclasses import dataclass, field

from src.abstain import GateResult, decide_abstain
from src.config import CORPUS_PATH, TOP_N
from src.generate import Generator, generate_answer, load_generator
from src.retrieve import RetrievalHit, RetrievalIndex, build_index, retrieve


@dataclass
class AnswerResult:
    query: str
    answer: str
    abstained: bool
    gate: GateResult
    hits: list[RetrievalHit] = field(default_factory=list)
    cited_doc_ids: list[str] = field(default_factory=list)


@dataclass
class RagPipeline:
    index: RetrievalIndex
    generator: Generator
    use_llm: bool = False

    @classmethod
    def create(
        cls,
        corpus_path: str | None = None,
        use_llm: bool = False,
        load_llm: bool = False,
    ) -> "RagPipeline":
        index = build_index(str(corpus_path or CORPUS_PATH))
        generator = load_generator(prefer_llm=load_llm or use_llm)
        return cls(index=index, generator=generator, use_llm=use_llm)

    def answer(self, query: str, top_n: int = TOP_N) -> AnswerResult:
        hits = retrieve(query, self.index, top_n=top_n)
        gate = decide_abstain(query, hits)
        if gate.abstain:
            return AnswerResult(
                query=query,
                answer=gate.message,
                abstained=True,
                gate=gate,
                hits=hits,
                cited_doc_ids=[],
            )
        docs = [h.doc for h in hits]
        text = generate_answer(query, docs, generator=self.generator, use_llm=self.use_llm)
        return AnswerResult(
            query=query,
            answer=text,
            abstained=False,
            gate=gate,
            hits=hits,
            cited_doc_ids=[h.doc.id for h in hits],
        )
