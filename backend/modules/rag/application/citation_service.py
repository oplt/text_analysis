from backend.modules.rag.domain.models import Citation, RetrievedChunk


class CitationService:
    def build_citations(self, chunks: list[RetrievedChunk]) -> list[Citation]:
        return [
            Citation(
                document_id=chunk.document_id,
                chunk_id=chunk.chunk_id,
                filename=chunk.filename,
                score=chunk.score,
                snippet=chunk.content[:400],
                page_number=chunk.page_number,
                chunk_index=chunk.chunk_index,
            )
            for chunk in chunks
        ]
