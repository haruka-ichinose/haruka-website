"""Pull public GitHub repository READMEs for the projects Haruka links to.

The repo list is every ``github`` link in ``frontend/lib/content.json`` plus the
``extra_repos`` setting, so it stays in sync as projects are added. Reuses the
request helpers from ``journals.py``. Missing repos (renamed, private) are skipped
with a warning so the build never fails on one bad link.
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
from pathlib import Path
from urllib.parse import urlparse

import aiohttp

from ..config import settings
from ..documents import SourceDoc
from .content_json import _slugify
from .journals import _BASE_URL, _get_json

logger = logging.getLogger(__name__)

# Cap README text so one huge README can't dominate retrieval.
README_MAX_CHARS = 8000


def _repo_slug(url: str) -> str | None:
    """``https://github.com/owner/name`` -> ``owner/name``."""
    parsed = urlparse(url)
    if parsed.netloc.lower() != "github.com":
        return None
    parts = [p for p in parsed.path.split("/") if p]
    if len(parts) < 2:
        return None
    return f"{parts[0]}/{parts[1].removesuffix('.git')}"


def linked_repos(content_json_path: str | Path) -> list[str]:
    """Every ``owner/name`` linked as a github link from content.json."""
    data = json.loads(Path(content_json_path).read_text(encoding="utf-8"))
    repos: list[str] = []
    for project in data.get("projects", []):
        for link in project.get("links", []):
            if link.get("type") == "github" and (slug := _repo_slug(link.get("url", ""))):
                repos.append(slug)
    return repos


def _repo_doc(repo: dict, readme: str | None, *, linked: bool) -> SourceDoc:
    name = repo["name"]
    meta = [f"Language: {repo['language']}"] if repo.get("language") else []
    if repo.get("topics"):
        meta.append(f"Topics: {', '.join(repo['topics'])}")
    parts = [
        f"{name} (GitHub repository, {repo['html_url']})",
        repo.get("description") or "",
        " · ".join(meta),
        (readme or "")[:README_MAX_CHARS],
    ]
    return SourceDoc(
        doc_id=f"repo-{_slugify(repo['full_name'])}",
        title=f"{name} (GitHub repository)",
        text="\n\n".join(p for p in parts if p),
        source_type="repo",
        # Linked repos cite the Projects page, where they're showcased.
        url="/" if linked else repo["html_url"],
        repo_url=repo["html_url"],
    )


async def _fetch_repo(session: aiohttp.ClientSession, slug: str) -> tuple[dict, str | None] | None:
    try:
        repo = await _get_json(session, f"{_BASE_URL}/repos/{slug}")
    except aiohttp.ClientResponseError as e:
        logger.warning("Skipping repo %s: HTTP %s", slug, e.status)
        return None
    try:
        data = await _get_json(session, f"{_BASE_URL}/repos/{slug}/readme")
        readme = base64.b64decode(data["content"].replace("\n", "")).decode("utf-8")
    except aiohttp.ClientResponseError as e:
        logger.warning("No README for %s: HTTP %s", slug, e.status)
        readme = None
    return repo, readme


async def fetch_repo_docs() -> list[SourceDoc]:
    """Fetch every linked + extra repo as a ``SourceDoc``. Requires ``github_token``."""
    settings.require("github_token")

    linked = linked_repos(settings.content_json_path)
    slugs = list(dict.fromkeys([*linked, *settings.extra_repos]))

    async with aiohttp.ClientSession() as session:
        results = await asyncio.gather(*(_fetch_repo(session, s) for s in slugs))

    return [
        _repo_doc(repo, readme, linked=slug in linked)
        for slug, result in zip(slugs, results)
        if result
        for repo, readme in [result]
    ]
