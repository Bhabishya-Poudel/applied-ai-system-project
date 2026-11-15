# AI-Assisted Development Notes

I developed this project with AI assistance for implementation options, code
drafts, test cases, and documentation. The summaries below record project goals,
design decisions, and verification rather than verbatim prompts.

## Extended song attributes

I expanded the original catalog with popularity, release decade, detailed mood
tags, instrumentalness, and language. AI assistance supported the data-model and
scoring changes. The design skips preferences absent from a profile, preserving
existing behavior. Automated checks cover numeric bounds, mood-tag overlap, and
loading the added columns. The catalog is synthetic.

## Scoring modes

I selected a dictionary of weight tables over separate classes for each mode.
This keeps the original scorer readable and allows balanced, genre-first,
mood-first, and energy-focused rankings. AI assistance helped compare those
options. Tests check that changing the mode can change the ranking and that
unknown modes are rejected.

## Variety and terminal presentation

The original scoring demo includes penalties for repeated artists and genres,
plus a terminal table of recommendations and reasons. AI assistance supported
the implementation and test cases. The penalties are adjustable ranking rules;
they do not establish cultural fairness or equal representation.

## RAG and custom documents

I extended the application to accept natural-language requests, retrieve song
records with local embeddings and preference checks, and generate explanations
with Gemini. AI-assisted custom listening notes add activity context to selected
songs. After reviewing the Gym Hero mismatch, I challenged embedding-only
ranking and adopted explicit mood and energy checks.

## Reliability and reflection

The recorded suite has 53 passing tests covering ranking, validation, source
context, API failures, and log privacy. Separate evaluation compares retrieval
methods on eight labeled requests. Code correctness and recommendation quality
remain separate concerns. The [model card](model_card.md#responsible-ai-reflection)
records accepted and revised suggestions, limitations, and unnecessary complexity.
