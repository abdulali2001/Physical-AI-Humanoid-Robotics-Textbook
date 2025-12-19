<!--
Sync Impact Report:
- Version change: 0.1.0 → 1.0.0
- Modified principles: [PRINCIPLE_1_NAME] → Spec-Driven Execution, [PRINCIPLE_2_NAME] → Grounded Accuracy, [PRINCIPLE_3_NAME] → Developer-Focused Clarity, [PRINCIPLE_4_NAME] → Reproducibility, [PRINCIPLE_5_NAME] → Controlled AI Behavior, [PRINCIPLE_6_NAME] → Book Content Quality
- Added sections: Additional Constraints (RAG Chatbot Standards), Development Workflow
- Removed sections: None
- Templates requiring updates: ⚠ pending - .specify/templates/plan-template.md, .specify/templates/spec-template.md, .specify/templates/tasks-template.md
- Follow-up TODOs: None
-->

# AI-Authored Technical Book with Embedded RAG Chatbot Constitution

## Core Principles

### Spec-Driven Execution
All development follows generated specifications and tasks without deviation. Implementation work must strictly adhere to the generated specs and task lists. Every feature and functionality originates from documented requirements before implementation begins.

### Grounded Accuracy
No hallucinations or fabricated information is allowed in the book content or RAG chatbot responses. All content must be factually accurate, verifiable, and grounded in the actual book material or user-selected text. Citations and source attribution are mandatory for all claims.

### Developer-Focused Clarity
Content must prioritize developer understanding and practical implementation. Technical explanations should be clear, concise, and accompanied by correct, runnable code examples. Content should be accessible to readers with Flesch-Kincaid grade 9-11 reading level.

### Reproducibility
All build and deployment processes must be reproducible. Instructions should work consistently across different environments. GitHub Actions workflows and documentation should ensure consistent deployments to GitHub Pages.

### Controlled AI Behavior
AI tools (Claude Code, OpenAI Agents) must operate within defined boundaries and constraints. AI-generated content requires verification and validation. RAG chatbot responses must be strictly limited to book content or user-selected text.

### Book Content Quality
All book content must meet high standards of technical accuracy, proper structure, and readability. Markdown/MDX content must be properly formatted, with clear navigation and accurate code examples. Docusaurus configuration must support optimal reading experience.

## RAG Chatbot Standards
- Chatbot stack: OpenAI Agents / ChatKit SDKs, FastAPI, Neon Serverless Postgres, Qdrant Cloud
- Retrieval-based responses only - no generative responses outside of book context
- Responses must cite source material from book content
- Privacy: no personal data collection without consent
- Performance: responses under 3 seconds for typical queries

## Development Workflow
- All changes must be reviewed and tested before merging
- Book content changes require accuracy verification
- Code examples must be tested and validated as runnable
- RAG chatbot changes require accuracy testing against book content
- GitHub Pages deployment pipeline must pass all checks
- Prompt History Records (PHRs) must be created for all significant changes

## Governance

This constitution governs all aspects of the AI-authored technical book and RAG chatbot project. All team members must comply with these principles. Amendments to this constitution require explicit approval and documentation of changes. All pull requests and code reviews must verify compliance with these principles. Any conflicts between this constitution and other project practices are resolved in favor of these principles.

**Version**: 1.0.0 | **Ratified**: 2025-12-16 | **Last Amended**: 2025-12-16
