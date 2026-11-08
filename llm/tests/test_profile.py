from mecha_llm.sources.profile import load_profile_docs


def _write_profile(tmp_path):
    path = tmp_path / "profile.md"
    path.write_text(
        "<!-- file header note -->\n\n"
        "## Skills and tech stack\n\nPython and FastAPI.\n\n<!-- TODO: trim -->\n\n"
        "## Career direction\n\nAI engineering.\n\n"
        "## Empty section\n\n<!-- only a comment -->\n",
        encoding="utf-8",
    )
    return path


def test_one_doc_per_section_skipping_empty(tmp_path):
    docs = load_profile_docs(_write_profile(tmp_path))
    assert [d.title for d in docs] == ["Skills and tech stack", "Career direction"]
    assert all(d.source_type == "profile" for d in docs)


def test_comments_are_stripped(tmp_path):
    docs = load_profile_docs(_write_profile(tmp_path))
    assert all("TODO" not in d.text and "<!--" not in d.text for d in docs)


def test_career_sections_route_to_experience(tmp_path):
    docs = {d.title: d for d in load_profile_docs(_write_profile(tmp_path))}
    assert docs["Career direction"].url == "/experience"
    assert docs["Skills and tech stack"].url == "/"
    assert docs["Skills and tech stack"].doc_id == "profile-skills-and-tech-stack"


def test_real_profile_file_loads():
    docs = load_profile_docs("content/profile.md")
    assert len(docs) == 8
