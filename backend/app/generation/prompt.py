"""
Prompt Construction for Baseline RAG.

Builds deterministic, grounded prompts that instruct Gemini to answer
strictly from supplied context and to cite sources.

No chain-of-thought, no reasoning traces — output only.
"""

from app.retrieval.retriever import RetrievedChunk


# ──────────────────────────────────────────────────────────────────────────────
# System-level instruction (prepended once per request)
# ──────────────────────────────────────────────────────────────────────────────
SYSTEM_INSTRUCTION = """\
You are a precise question-answering assistant.

Rules:
1. Answer ONLY using information found in the provided context passages.
2. If the context does not contain enough information to answer the question, \
say: "I cannot answer this question based on the provided context."
3. Do NOT invent, assume, or extrapolate facts not present in the context.
4. After your answer, list the sources you used in the format:
   Sources: [<source>, page <page>], ...
5. Be concise and direct. Do not repeat the question.\
"""


def format_context_block(chunks: list[RetrievedChunk]) -> str:
    """Convert a list of retrieved chunks into a numbered context block.

    Each passage is labelled with its sequence number, source filename,
    and page number so the model can reference them precisely.

    Args:
        chunks: Ranked list of RetrievedChunk objects.

    Returns:
        Multi-line string ready to be embedded in the prompt.
        Returns an empty string when the chunk list is empty.
    """
    if not chunks:
        return ""

    lines: list[str] = []
    for i, chunk in enumerate(chunks, start=1):
        lines.append(
            f"[{i}] Source: {chunk.source} | Page: {chunk.page_number}"
        )
        lines.append(chunk.text.strip())
        lines.append("")          # blank separator between passages

    return "\n".join(lines).rstrip()


def build_rag_prompt(
    question: str,
    chunks: list[RetrievedChunk],
) -> str:
    """Assemble the full RAG prompt for a single question.

    Args:
        question: The user's question string.
        chunks:   Retrieved context passages (may be empty).

    Returns:
        Complete prompt string to send to the Gemini API.
    """
    question = question.strip()
    context_block = format_context_block(chunks)

    if not context_block:
        # No retrieved context — instruct the model to say so explicitly
        context_section = "No relevant context was retrieved for this question."
    else:
        context_section = context_block

    prompt = (
        f"{SYSTEM_INSTRUCTION}\n\n"
        f"---\n\n"
        f"CONTEXT PASSAGES:\n\n"
        f"{context_section}\n\n"
        f"---\n\n"
        f"QUESTION: {question}\n\n"
        f"ANSWER:"
    )
    return prompt
