import json

from mecha_llm.sources.github_repos import README_MAX_CHARS, _repo_doc, linked_repos

_REPO = {
    "name": "apple-thinning",
    "full_name": "uc-vision/apple-thinning",
    "html_url": "https://github.com/uc-vision/apple-thinning",
    "description": "VR apple thinning game",
    "language": "GDScript",
    "topics": ["vr", "godot"],
}


def test_linked_repos_reads_github_links(tmp_path):
    path = tmp_path / "content.json"
    path.write_text(
        json.dumps(
            {
                "projects": [
                    {"links": [{"type": "github", "url": "https://github.com/uc-vision/apple-thinning/"}]},
                    {"links": [{"type": "demo", "url": "https://example.com"}]},
                ]
            }
        ),
        encoding="utf-8",
    )
    assert linked_repos(path) == ["uc-vision/apple-thinning"]


def test_linked_repo_cites_projects_page():
    doc = _repo_doc(_REPO, "# Apple Thinning\n\nA game.", linked=True)
    assert doc.source_type == "repo"
    assert doc.url == "/"
    assert doc.repo_url == _REPO["html_url"]
    assert "Language: GDScript · Topics: vr, godot" in doc.text
    assert "A game." in doc.text


def test_extra_repo_cites_github_and_caps_readme():
    doc = _repo_doc(_REPO, "x" * (README_MAX_CHARS + 500), linked=False)
    assert doc.url == _REPO["html_url"]
    assert doc.text.count("x") <= README_MAX_CHARS + 10
