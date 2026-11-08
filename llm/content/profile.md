<!--
Hand-authored profile content for mecha-haruka (Phase 1a of
docs/mecha-haruka-improvements.md). One `## ` section = one SourceDoc.

HTML comments like this one are stripped before ingestion, so use them for
notes and TODOs. Everything outside a comment is shown to the chatbot as fact —
only write what you're happy for it to repeat to visitors.

DRAFT: written by Claude from content.json / CLAUDE.md. Review every section and
resolve the TODO(haruka) comments before rebuilding the index.
-->

## Skills and tech stack

Haruka is an AI engineer and full-stack software developer. Their strongest areas are building AI-integrated applications — especially retrieval-augmented generation (RAG) systems — and the full-stack web apps around them.

- Languages: Python, TypeScript/JavaScript, C#, Java, SQL.
- Frontend: Next.js, React, Vue.js, Tailwind CSS, THREE.js and WebXR.
- Backend: FastAPI, .NET, Spring Boot.
- AI and data: RAG pipelines, Azure OpenAI, embeddings and vector search (pgvector, Azure AI Search), LLM evaluation with RAGAS and DeepEval, Snowflake, Prefect.
- Cloud and infrastructure: Azure, AWS, Vercel, Docker, Postgres, SQL Server, MariaDB, self-hosting on a Raspberry Pi behind a Cloudflare Tunnel, OpenTelemetry observability.
- Other: Godot game engine for VR (Oculus Quest 2), Jest testing.

<!-- TODO(haruka): reorder / trim to what you'd actually call your core stack. -->

## About Haruka

Haruka is a software and AI engineer based in New Zealand, currently working as an AI Engineer at Spark New Zealand (since 2023). They studied at the University of Canterbury, completing a Bachelor of Science double-majoring in Computer Science and Physics, followed by a Professional Master of Computer Science.

<!-- TODO(haruka): city/timezone, work eligibility, languages spoken, any hobbies or
light personal details you're happy to share. -->

## How Haruka works

Haruka likes to understand a system end to end — from data and infrastructure through to the user interface — and to measure whether it works rather than assume it does. On this site, for example, the chatbot is evaluated with RAGAS and DeepEval, and its hosting was migrated from Azure to a self-hosted Raspberry Pi only after re-running the evaluation to confirm answer quality held up. Haruka keeps a dev journal of what they build and learn, which is published on this site.

<!-- TODO(haruka): your own words on working style, and what you're learning right now. -->

## Background and how Haruka got into tech

Haruka started at the University of Canterbury studying both Computer Science and Physics. During the Professional Master of Computer Science they built Spectral Detect, a full-stack app that helps the Department of Conservation detect the invasive plant euphorbia paralias from drone (UAV) imagery, and worked as a research assistant in the university's computer vision lab, building the Apple Thinning VR training game for farmers. In 2022–2023 they interned as a software developer at Umajin, working on a React dashboard and a WebXR application. In 2023 they joined Spark New Zealand as an AI Engineer, working on an AI-decisioning marketing system, a B2B sales recommendation system built on RAG, and an automated Azure cost-optimisation system.

<!-- TODO(haruka): the "why" — what drew you from physics to software and then to AI. -->

## Career direction

Haruka's focus is AI engineering: building practical, well-evaluated LLM and RAG applications and the software and infrastructure that make them reliable in production.

<!-- TODO(haruka): what kind of role/work you want next, whether you're open to
opportunities, and anything you'd like recruiters to know. -->

## Contact

The best way to get in touch with Haruka is the contact form in the footer of every page on this site — it sends a message straight to Haruka's inbox. Haruka's GitHub profile is https://github.com/Jade1724.

<!-- TODO(haruka): add LinkedIn or a public email here if you want the bot to share them. -->

## About this site

This is Haruka's portfolio website. It has three pages: Projects (the home page, /), Experience (/experience), and the dev journal (/journal). The frontend is built with Next.js 16, Tailwind CSS, and shadcn/ui, and is hosted on Vercel. The backend is a Python FastAPI service, also on Vercel, which serves the dev journal by reading markdown from Haruka's private Obsidian notes repository on GitHub. The source code is public at https://github.com/Jade1724/haruka-website.

## About Mecha Haruka

Mecha Haruka is the chatbot on this site — an AI persona of Haruka, not Haruka themselves. It answers questions about Haruka's projects, work and education history, skills, background, public GitHub repositories, dev journal, and this website. It answers only from that material, so it won't guess about things it hasn't been given, and it will point you to the right page or suggest a question it can answer instead.

Under the hood it is a retrieval-augmented generation (RAG) system: questions are embedded with a local sentence-transformers model and matched against an index of Haruka's content using hybrid vector and keyword search in Postgres with pgvector. The retrieval service runs on a Raspberry Pi 5, reached securely from the Vercel-hosted site through a Cloudflare Tunnel, and answers are generated with an Azure AI Foundry model. Inputs are screened with Azure AI Content Safety, and answer quality is tracked with RAGAS and DeepEval evaluations.
