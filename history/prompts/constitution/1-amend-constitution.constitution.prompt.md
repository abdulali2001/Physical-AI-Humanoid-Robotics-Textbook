---
id: 1
title: Amend Constitution
stage: constitution
date: 2025-12-16
surface: agent
model: Claude Sonnet 4.5
feature: none
branch: master
user: user
command: /sp.constitution
labels: [constitution,book,rag-chatbot]
links:
  spec: null
  ticket: null
  adr: null
  pr: null
files:
 - .specify/memory/constitution.md
tests:
 - none
---

## Prompt

```text
/sp.constitution

Project: AI-Authored Technical Book with Embedded RAG Chatbot

Purpose:
Create a unified technical book using Spec-Kit Plus and Claude Code, published with Docusaurus on GitHub Pages, with an embedded RAG chatbot that answers questions strictly from the book content or user-selected text.

Core principles:
- Spec-driven execution (no deviation from generated specs/tasks)
- Grounded accuracy (no hallucinations)
- Developer-focused clarity
- Reproducibility
- Controlled AI behavior

Key standards:

Book:
- Written in Markdown/MDX using Docusaurus
- Clear chapter structure and navigation
- Accurate, implementation-focused content
- Correct, runnable code examples
- Readability: Flesch-Kincaid grade 9–11

Deployment:
- GitHub Pages deployment
- Reproducible build and deploy instructions
- Clean repository structure with README

RAG Chatbot:
- Embedded in the book UI
- Stack:
  - OpenAI Agents / ChatKit SDKs
  - FastAPI
  - Neon Serverless Postgres
  - Qdrant Cloud (free tier)
- Retrieval based
```

## Response snapshot

Updated the constitution file to reflect the AI-authored technical book project with embedded RAG chatbot. The new constitution includes:

Core Principles:
1. Spec-Driven Execution: All development follows generated specifications and tasks without deviation
2. Grounded Accuracy: No hallucinations - all content must be factually accurate and verifiable
3. Developer-Focused Clarity: Content prioritizes developer understanding with clear, runnable examples
4. Reproducibility: All build and deployment processes must be reproducible
5. Controlled AI Behavior: AI tools operate within defined boundaries
6. Book Content Quality: High standards for technical accuracy and readability

Additional sections:
- RAG Chatbot Standards: Technical stack and operational requirements
- Development Workflow: Change review and testing processes
- Governance: Compliance and conflict resolution

## Outcome

- ✅ Impact: Created comprehensive project constitution with 6 core principles and supporting sections
- 🧪 Tests: none
- 📁 Files: Updated .specify/memory/constitution.md with project-specific principles
- 🔁 Next prompts: Continue with spec creation, plan development, and implementation
- 🧠 Reflection: Constitution now provides clear governance framework for the AI-authored book project

## Evaluation notes (flywheel)

- Failure modes observed: None
- Graders run and results (PASS/FAIL): Not applicable
- Prompt variant (if applicable): Not applicable
- Next experiment (smallest change to try): Begin spec creation based on this constitution