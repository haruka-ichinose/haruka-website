"""Golden Q/A set grounded in the real portfolio content (projects, experience,
profile, repo READMEs), plus two unanswerable questions whose reference is the
bot's guiding behaviour rather than a fact.

Reference answers are the ground truth for RAGAS (context recall, correctness)
and the expected output for DeepEval's G-Eval. Kept factual and content-derived
so the gate is meaningful; adversarial/off-topic cases live in ``redteam.py``.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class GoldenItem:
    question: str
    reference: str  # ground-truth answer
    expected_url: str | None = None  # a route/link a good answer should surface


GOLDEN: list[GoldenItem] = [
    GoldenItem(
        question="Does Haruka have a project building a VR game?",
        reference=(
            "Yes — Apple Thinning, an agriculture VR training game for thinning apple "
            "fruitlets, built with the Godot engine to run on the Oculus Quest 2, as part "
            "of University of Canterbury's computer vision lab. The public repo is "
            "https://github.com/uc-vision/apple-thinning."
        ),
        expected_url="https://github.com/uc-vision/apple-thinning",
    ),
    GoldenItem(
        question="What database does the B2B Sales Recommendation System use?",
        reference=(
            "The B2B Sales Recommendation System uses a Snowflake database. It is an AI "
            "sales recommendation system using RAG, built with Next.js, FastAPI, and Azure "
            "OpenAI."
        ),
    ),
    GoldenItem(
        question="What is Spectral Detect and what was it built with?",
        reference=(
            "Spectral Detect is an invasive plant detection and location-tracking full-stack "
            "web app built with Vue.js, Spring Boot, and MariaDB, hosted on AWS. It was a "
            "University of Canterbury master's project to help the Department of Conservation "
            "automatically detect euphorbia paralias from UAV imagery."
        ),
    ),
    GoldenItem(
        question="Where can I see Haruka's work history?",
        reference=(
            "On the Experience page (/experience), which lists Haruka's career timeline — "
            "work history and education."
        ),
        expected_url="/experience",
    ),
    GoldenItem(
        question="What is Haruka's current job?",
        reference=(
            "Haruka is an AI Engineer at Spark New Zealand, from 2023 to the present, "
            "developing applications and systems integrated with AI, including an "
            "AI-decisioning marketing system and a B2B sales recommendation system."
        ),
    ),
    GoldenItem(
        question="Where did Haruka intern before, and what did they work on?",
        reference=(
            "Haruka was a Software Developer Intern at Umajin (2022–2023), working on a React "
            "dashboard app for an automotive oil company client and a WebXR application using "
            "THREE.js with Jest testing."
        ),
    ),
    GoldenItem(
        question="What did Haruka study at university?",
        reference=(
            "At the University of Canterbury, Haruka completed a Professional Master of "
            "Computer Science (a project-based degree building a full-stack web app to "
            "automate invasive plant detection) and a Bachelor of Science with a double major "
            "in Computer Science and Physics."
        ),
    ),
    GoldenItem(
        question="What tech stack is the Fruit Maturity Clearance project built with?",
        reference=(
            "The Fruit Maturity Clearance Project is built with Next.js, .NET, and SQL Server "
            "on Azure. It is a maturity clearance system for fruit growers, packhouses, and "
            "labs to digitally track fruit quality and maturity."
        ),
    ),
    GoldenItem(
        question="What is this portfolio website built with?",
        reference=(
            "The portfolio website is built with Next.js, Tailwind CSS, and FastAPI, hosted "
            "on Vercel. It features a project showcase, a career timeline, and a dev journal "
            "scraped from a private Obsidian repository."
        ),
        expected_url="/",
    ),
    GoldenItem(
        question="What's Haruka's tech stack?",
        reference=(
            "Haruka works mainly in Python and TypeScript, with Next.js, React, and Vue.js on "
            "the frontend, FastAPI, .NET, and Spring Boot on the backend, and AI tooling such "
            "as RAG pipelines, Azure OpenAI, pgvector, and RAGAS/DeepEval evaluation, on "
            "Azure, AWS, and Vercel."
        ),
        expected_url="/",
    ),
    GoldenItem(
        question="Tell me about Haruka.",
        reference=(
            "Haruka is a software and AI engineer based in New Zealand, working as an AI "
            "Engineer at Spark New Zealand since 2023. They studied Computer Science and "
            "Physics at the University of Canterbury, followed by a Professional Master of "
            "Computer Science."
        ),
        expected_url="/experience",
    ),
    GoldenItem(
        question="What kind of work is Haruka focused on?",
        reference=(
            "Haruka is focused on AI engineering: building practical, well-evaluated LLM and "
            "RAG applications and the software and infrastructure that make them reliable "
            "in production."
        ),
        expected_url="/experience",
    ),
    GoldenItem(
        question="How can I contact Haruka?",
        reference=(
            "Use the contact form in the footer of any page on the site, which sends a "
            "message to Haruka's inbox. Haruka's GitHub profile is also linked."
        ),
        expected_url="/",
    ),
    GoldenItem(
        question="What can you help me with?",
        reference=(
            "Mecha Haruka is an AI persona of Haruka that answers questions about Haruka's "
            "projects, work and education history, skills, background, public GitHub "
            "repositories, dev journal, and this website, using only that material."
        ),
        expected_url="/",
    ),
    GoldenItem(
        question="Which version of Godot was the Apple Thinning VR game built with?",
        reference=(
            "According to its README, the Apple Thinning project is built on the Godot game "
            "engine v3.4."
        ),
        expected_url="https://github.com/uc-vision/apple-thinning",
    ),
    GoldenItem(
        question="What did Haruka do in 2015?",
        reference=(
            "The bot says it doesn't have information about 2015, describes the closest "
            "thing it knows (Haruka's studies at the University of Canterbury from 2018 and "
            "later work), suggests questions it can answer, and links the Experience page "
            "(/experience)."
        ),
        expected_url="/experience",
    ),
    GoldenItem(
        question="What is Haruka's favourite food?",
        reference=(
            "The bot says it doesn't have that information, offers what it can talk about "
            "instead (Haruka's projects, experience, skills, and dev journal) with suggested "
            "questions, and links a relevant page such as the dev journal (/journal)."
        ),
    ),
]
