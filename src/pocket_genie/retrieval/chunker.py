from __future__ import annotations

import dataclasses
import re

_IMAGE_TAG_RE = re.compile(r'<image\s+id=["\']([^"\']+)["\']\s*/?>')
_MD_IMAGE_RE = re.compile(r"!\[([^\]]*)\]\(([^)]+)\)")


@dataclasses.dataclass(slots=True)
class DocumentChunk:
    doc_id: str
    source_path: str
    chunk_index: int
    text: str
    image_markers: list[str] = dataclasses.field(default_factory=list)

    @property
    def citation(self) -> str:
        return f"{self.source_path}#chunk={self.chunk_index}"


def extract_image_markers(text: str) -> list[str]:
    markers: list[str] = []
    for m in _IMAGE_TAG_RE.finditer(text):
        markers.append(m.group(1))
    for m in _MD_IMAGE_RE.finditer(text):
        target = m.group(2).strip()
        if not target.startswith(("http://", "https://", "/api/media/")):
            markers.append(target)
    return markers


def reconstruct_chunk_with_media(text: str, media_prefix: str = "/api/media/kb-images") -> str:
    """Transform XML/Markdown image placeholders into stable media endpoints."""
    prefix = media_prefix.rstrip("/")
    step1 = _IMAGE_TAG_RE.sub(lambda m: f"![Image]({prefix}/{m.group(1)})", text)

    def _replace_md(match: re.Match[str]) -> str:
        alt, ref = match.group(1), match.group(2).strip()
        if ref.startswith(("http://", "https://", "/api/media/", "file://")):
            return match.group(0)
        return f"![{alt or 'Image'}]({prefix}/{ref})"

    return _MD_IMAGE_RE.sub(_replace_md, step1)


def slice_document_into_chunks(
    doc_id: str,
    source_path: str,
    content: str,
    chunk_chars: int = 800,
    overlap_chars: int = 120,
) -> list[DocumentChunk]:
    """Slice text into sequential chunks identified by chunk_index while preserving paragraph/line boundaries."""
    cleaned = content.strip()
    if not cleaned:
        return []

    paragraphs = re.split(r"\n\s*\n", cleaned)
    chunks: list[DocumentChunk] = []
    buffer = ""
    chunk_idx = 0

    for para in paragraphs:
        candidate = f"{buffer}\n\n{para}".strip() if buffer else para.strip()
        if len(candidate) <= chunk_chars:
            buffer = candidate
            continue

        if buffer:
            chunks.append(
                DocumentChunk(
                    doc_id=doc_id,
                    source_path=source_path,
                    chunk_index=chunk_idx,
                    text=buffer,
                    image_markers=extract_image_markers(buffer),
                )
            )
            chunk_idx += 1
            tail = buffer[-overlap_chars:] if overlap_chars > 0 else ""
            buffer = f"{tail}\n\n{para}".strip() if tail else para.strip()
        else:
            buffer = para.strip()

        while len(buffer) > chunk_chars:
            piece = buffer[:chunk_chars]
            chunks.append(
                DocumentChunk(
                    doc_id=doc_id,
                    source_path=source_path,
                    chunk_index=chunk_idx,
                    text=piece,
                    image_markers=extract_image_markers(piece),
                )
            )
            chunk_idx += 1
            step = max(1, chunk_chars - overlap_chars)
            buffer = buffer[step:].strip()

    if buffer:
        chunks.append(
            DocumentChunk(
                doc_id=doc_id,
                source_path=source_path,
                chunk_index=chunk_idx,
                text=buffer,
                image_markers=extract_image_markers(buffer),
            )
        )

    return chunks
