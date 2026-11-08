"""Load hand-authored profile content from ``llm/content/profile.md``.

Each ``## `` section becomes one ``SourceDoc`` (skills, bio, career story, contact,
site/bot FAQ). HTML comments are stripped first so the file can carry notes and
TODOs that never reach the index. Career sections cite ``/experience``; the rest
cite the Projects page (``/``).
"""

from __future__ import annotations

import re
from pathlib import Path

from ..documents import SourceDoc
from .content_json import _slugify

_COMMENT_RE = re.compile(r"<!--.*?-->", re.DOTALL)
_SECTION_RE = re.compile(r"^## +(.+?)\s*$", re.MULTILINE)

# Sections about Haruka's career point at the Experience page.
_EXPERIENCE_SECTIONS = {"about-haruka", "background-and-how-haruka-got-into-tech", "career-direction"}


def load_profile_docs(profile_md_path: str | Path) -> list[SourceDoc]:
    text = _COMMENT_RE.sub("", Path(profile_md_path).read_text(encoding="utf-8"))
    matches = list(_SECTION_RE.finditer(text))

    docs: list[SourceDoc] = []
    for i, m in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[m.end() : end].strip()
        if not body:
            continue
        title = m.group(1)
        slug = _slugify(title)
        docs.append(
            SourceDoc(
                doc_id=f"profile-{slug}",
                title=title,
                text=body,
                source_type="profile",
                url="/experience" if slug in _EXPERIENCE_SECTIONS else "/",
            )
        )
    return docs
