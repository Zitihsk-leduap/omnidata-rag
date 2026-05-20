import argparse
import os
import shutil
import re
import unicodedata
import hashlib
from typing import List, Tuple

from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_core.documents import Document
from langchain_chroma import Chroma

from generate_embeddings import get_embeddings


# PATHS
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.abspath(os.path.join(BASE_DIR, "..", "Data"))
CHROMA_PATH = os.path.join(BASE_DIR, "chroma")

print("CHROMA PATH:", CHROMA_PATH)
print("DATA PATH:", DATA_PATH)


# CLEAN
def clean_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


# LOAD TXT FILES
def load_txt_files() -> List[Document]:
    loader = DirectoryLoader(
        DATA_PATH,
        glob="**/*.txt",
        loader_cls=TextLoader,
        show_progress=True
    )

    docs = loader.load()

    for d in docs:
        d.metadata["source_type"] = "txt"

    return docs


# ================================================================
# PARENT-CHILD HIERARCHICAL CHUNKING
# ================================================================

# Structural regex patterns for Nepali legal documents
# Matches: परिच्छेद-१, परिच्छेद–२, परिच्छेद—३, etc.
RE_PARICHHED = re.compile(r'(परिच्छेद[-–—]?\s*[०-९0-9]+)')

# Matches दफा/section start: Devanagari number + dot at line start
# e.g. "१.", "२.", "१८.", "१५३."
RE_DAFA_START = re.compile(r'(?:^|\n)\s*([०-९]+)\.')

# Clause markers: (क), (ख), (ग), (घ), (ङ), (च), (छ), (ज), (झ),
# (ञ), (ट), (ठ), (ड), (ढ), (ण), (त), (थ), (द), (ध), (न),
# (प), (प१), (फ), (ब), (भ), etc.
RE_CLAUSE_SPLIT = re.compile(r'(?=\([क-ह][क-ह0-9१-९]*\))')


def _detect_parichhed(text: str, pos: int, parichhed_map: list) -> str:
    """Find which परिच्छेद a given position belongs to."""
    current = ""
    for pp in parichhed_map:
        if pp["pos"] <= pos:
            current = pp["name"]
    return current


def _extract_title(section_text: str) -> str:
    """Extract a short title from the first line of a section."""
    # Match pattern: "N. Title:" or "N. Title।"
    m = re.match(r'[०-९]+\.\s*(.+?)(?:\s*[:।(])', section_text)
    if m:
        return clean_text(m.group(1))
    # Fallback: first 80 characters
    first_line = section_text.split('\n')[0] if '\n' in section_text else section_text
    return clean_text(first_line[:80])


def _split_text_into_sections(full_text: str) -> List[dict]:
    """
    Split full document text into legal sections (दफा) using
    Devanagari numbered markers (१., २., ३., etc.).

    Also detects परिच्छेद (chapter) boundaries for metadata.

    Returns list of dicts with keys: text, title, parichhed
    """
    # Build a map of परिच्छेद positions and names
    parichhed_map = []
    for m in re.finditer(
        r'(परिच्छेद[-–—]?\s*[०-९0-9]+)\s*\n\s*(.+?)(?:\n|$)', full_text
    ):
        parichhed_map.append({
            "pos": m.start(),
            "name": clean_text(m.group(1) + " " + m.group(2))
        })

    # Find all दफा start positions
    dafa_starts = list(RE_DAFA_START.finditer(full_text))

    sections = []

    if not dafa_starts:
        # No section markers found — treat entire text as one section
        cleaned = clean_text(full_text)
        if len(cleaned) > 20:
            sections.append({
                "text": cleaned,
                "title": _extract_title(full_text),
                "parichhed": parichhed_map[0]["name"] if parichhed_map else "",
            })
        return sections

    # Capture preamble (text before first दफा)
    if dafa_starts[0].start() > 0:
        preamble = full_text[:dafa_starts[0].start()].strip()
        cleaned_pre = clean_text(preamble)
        if len(cleaned_pre) > 20:
            sections.append({
                "text": cleaned_pre,
                "title": _extract_title(preamble),
                "parichhed": _detect_parichhed(
                    full_text, 0, parichhed_map
                ),
            })

    # Extract each दफा section
    for i, match in enumerate(dafa_starts):
        start = match.start()
        end = (
            dafa_starts[i + 1].start()
            if i + 1 < len(dafa_starts)
            else len(full_text)
        )

        raw = full_text[start:end].strip()
        cleaned = clean_text(raw)

        if len(cleaned) < 20:
            continue

        sections.append({
            "text": cleaned,
            "title": _extract_title(raw),
            "parichhed": _detect_parichhed(
                full_text, start, parichhed_map
            ),
        })

    return sections


def _split_section_into_clauses(section_text: str) -> List[dict]:
    """
    Split a section into clause groups (NOT individual clauses).
    
    Production strategy:
    1. Group related clauses (e.g., (क) + (ख) together, (ग) + (घ) together)
    2. Keep preamble separate
    3. Create overlapping chunks for context continuity
    
    Returns list of dicts with keys: text, clause_markers
    """
    # Split by clause markers but keep them
    parts = RE_CLAUSE_SPLIT.split(section_text)
    parts = [p.strip() for p in parts if p.strip()]
    
    if not parts or len(parts) < 2:
        # No clauses found, treat entire section as one chunk
        return [{"text": section_text, "clause_markers": ""}]
    
    # Extract preamble (text before first clause)
    preamble = parts[0]
    clauses_raw = parts[1:]
    
    clause_groups = []
    
    # Group consecutive clauses (2 per group) for richer context
    group_size = 2
    for i in range(0, len(clauses_raw), group_size):
        group = clauses_raw[i:i+group_size]
        grouped_text = " ".join(group)
        
        # Extract clause markers for metadata
        markers = re.findall(r'\([क-ह][क-ह0-9१-९]*\)', grouped_text)
        markers_str = ", ".join(markers) if markers else ""
        
        clause_groups.append({
            "text": grouped_text,
            "clause_markers": markers_str
        })
    
    # Build overlapping chunks: preamble + first group, then sliding window
    chunks = []
    
    # Always include preamble + first clause group
    if clause_groups:
        preamble_chunk = preamble + " " + clause_groups[0]["text"]
        chunks.append({
            "text": preamble_chunk,
            "clause_markers": clause_groups[0]["clause_markers"],
            "has_preamble": True
        })
        
        # Overlapping chunks: group[i] + group[i+1] for context continuity
        for i in range(len(clause_groups) - 1):
            overlap_chunk = clause_groups[i]["text"] + " " + clause_groups[i+1]["text"]
            chunks.append({
                "text": overlap_chunk,
                "clause_markers": clause_groups[i]["clause_markers"] + ", " + clause_groups[i+1]["clause_markers"],
                "has_preamble": False
            })
        
        # Last group standalone (if not already in overlap)
        if len(clause_groups) > 1:
            chunks.append({
                "text": clause_groups[-1]["text"],
                "clause_markers": clause_groups[-1]["clause_markers"],
                "has_preamble": False
            })
    else:
        # Only preamble
        chunks.append({
            "text": preamble,
            "clause_markers": "",
            "has_preamble": True
        })
    
    return chunks


def build_parent_child_chunks(
    docs: List[Document],
) -> Tuple[List[Document], List[Document]]:
    """
    Build hierarchical parent-child chunks with context windows.

    PARENT CHUNKS:
        Complete legal sections (दफा). Each parent contains the full
        cleaned text of a section. Metadata includes type="parent",
        a unique parent_id, source, title, and parichhed (chapter).

    CHILD CHUNKS:
        Clause-GROUP level units (2+ related clauses), NOT individual clauses.
        Each child contains grouped clauses with overlapping context.
        Metadata includes type="child", a unique child_id, parent_id,
        clause_markers (for filtering/context).

    STRATEGY:
        - Group related clauses together for richer context
        - Overlap consecutive groups for continuity
        - Preserve full section as parent for contextual expansion
        - All children link back to parent for retrieval augmentation

    Returns:
        Tuple of (parent_documents, child_documents)
    """
    parents = []
    children = []

    for doc in docs:
        source = doc.metadata.get("source", "unknown")
        full_text = doc.page_content

        # Split document into parent sections (दफा)
        sections = _split_text_into_sections(full_text)
        print(f"  Source: {source} -> {len(sections)} parent sections")

        for section in sections:
            # Deterministic parent ID
            parent_hash = hashlib.md5(
                section["text"].encode()
            ).hexdigest()
            parent_id = f"parent_{parent_hash}"

            parent_doc = Document(
                page_content=section["text"],
                metadata={
                    "type": "parent",
                    "parent_id": parent_id,
                    "id": parent_id,
                    "source": source,
                    "source_type": "txt",
                    "title": section["title"],
                    "parichhed": section["parichhed"],
                },
            )
            parents.append(parent_doc)

            # Split section into clause GROUPS with overlap (not individual clauses)
            clause_groups = _split_section_into_clauses(section["text"])

            for group_idx, group in enumerate(clause_groups):
                if len(group["text"]) < 20:
                    continue

                child_hash = hashlib.md5(
                    group["text"].encode()
                ).hexdigest()
                child_id = f"child_{child_hash}"

                child_doc = Document(
                    page_content=group["text"],
                    metadata={
                        "type": "child",
                        "child_id": child_id,
                        "id": child_id,
                        "parent_id": parent_id,
                        "source": source,
                        "source_type": "txt",
                        "title": section["title"],
                        "parichhed": section["parichhed"],
                        "clause_markers": group.get("clause_markers", ""),
                        "group_index": group_idx,
                        "has_preamble": group.get("has_preamble", False),
                    },
                )
                children.append(child_doc)

    print(f"\nHierarchical chunking (production) complete:")
    print(f"  Parents (full sections): {len(parents)}")
    print(f"  Children (clause groups with overlap): {len(children)}")

    return parents, children


# VECTOR STORE
def add_to_vectorstore(chunks: List[Document]):
    db = Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=get_embeddings()
    )

    print("BEFORE:", len(db.get().get("ids", [])))

    for c in chunks:
        if "id" not in c.metadata:
            c.metadata["id"] = hashlib.md5(
                c.page_content.encode()
            ).hexdigest()

    # Avoid duplicates in same run
    seen = set()
    unique = []

    for c in chunks:
        if c.metadata["id"] not in seen:
            unique.append(c)
            seen.add(c.metadata["id"])

    # Also skip IDs already in the database
    existing_ids = set(db.get().get("ids", []))
    new_docs = [c for c in unique if c.metadata["id"] not in existing_ids]

    if new_docs:
        db.add_documents(
            new_docs,
            ids=[c.metadata["id"] for c in new_docs]
        )
        print(f"Added {len(new_docs)} new documents")
    else:
        print("No new documents to add")

    print("AFTER:", len(db.get().get("ids", [])))


# ---------------- CLEAR DB ----------------
def clear_db():
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)
        print("DB CLEARED")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--clear_db", action="store_true")
    args = parser.parse_args()

    if args.clear_db:
        clear_db()

    print("Loading TXT files...")
    docs = load_txt_files()

    print("Building parent-child hierarchical chunks...")
    parents, children = build_parent_child_chunks(docs)

    # Combine parents + children for insertion into vector store
    # Children are the primary retrieval units (fine-grained clauses)
    # Parents serve as contextual expansion units (full sections)
    all_chunks = parents + children

    print(f"\nTotal documents to insert: {len(all_chunks)}")
    print(f"  Parents (contextual expansion): {len(parents)}")
    print(f"  Children (primary retrieval): {len(children)}")

    print("\nSaving to DB...")
    add_to_vectorstore(all_chunks)

    print("DONE")


if __name__ == "__main__":
    main()