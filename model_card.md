# Model Card: Music Recommender with RAG

## Purpose and models

An educational extension of **Music Recommender Simulation**, CodePath Module 3,
Week 6. Local MiniLM embeddings and preference rules search 18 synthetic song
records, enriched with three custom listening notes. Gemini explains the
selected records and notes. Neither model was fine-tuned;
the app does not listen to or play music. See the [system diagram](diagrams/system.md).

## Evaluation and safeguards

**53 automated tests passed.** On eight labeled requests, hybrid retrieval
returned a relevant first choice **8/8 times**, but only **11/24 returned songs**
were labeled relevant. The [testing summary](README.md#testing-summary) records
checks and results. Mocked tests measure code behavior; retrieval evaluation
does not measure explanation factuality.

The app validates inputs, catalog values, response format, and cited IDs.
Invalid explanations are withheld, and API failures leave retrieved songs
visible. Application logs exclude request text and credentials.

### RAG enhancement: a second source

`data/listening_notes.md` contains AI-assisted editorial activity suggestions.
Notes are joined to song records before embedding and supplied to Gemini only
for selected songs. Outputs identify the source files. For a Pomodoro lecture-note
request, Focus Flow moved from outside the top three to first place. The
[before/after example](README.md#rag-enhancement-custom-documents) records this
case; it is not a broad quality benchmark. The notes are suggestions, not
listening studies or evidence of cognitive benefits.

## Responsible-AI reflection

### Limitations and biases

The small, uneven, synthetic catalog cannot represent different listeners or
musical cultures. Always returning three songs introduces weak suggestions
when only one fits. Equating calm with low energy reflects a design assumption,
not everyone's experience. Hand-chosen weights and English keyword rules can
misread nuanced requests. Valid IDs do not guarantee factual explanations, and
eight labeled requests cannot establish broad reliability.

### Possible misuse and prevention

Someone could present fictional catalog records or generated explanations as
verified music knowledge, or ask the model to ignore its sources. Labeling the
catalog synthetic, restricting the prompt to retrieved records, validating IDs,
and reviewing claims help reduce those risks. These checks cannot establish
the truth of every sentence.

Requests and selected records go to Google, so users should avoid sensitive
information. Keys stay in an ignored `.env` file, which is not encrypted.
A public version would also need usage limits to reduce API abuse and costs;
those limits are not implemented in this local application.

### What surprised me during reliability testing

Gym Hero ranked above a calm song because its embedding similarity was slightly
higher. Gemini then correctly explained why its own supplied recommendation was
a poor fit. Retrieval quality and explanation quality were separate problems.
Even when all automated tests passed, many second and third choices remained weak.
A passing test suite was useful evidence, but not proof of a good experience.

### How I collaborated with AI

I used Codex to help refactor, test, explain, and document the application. I
questioned the Gym Hero result, asked what embeddings contributed, and requested
simpler explanations and documentation. I used outputs to challenge design
choices rather than treating generated code as automatically correct.

### One suggestion I accepted

I accepted loading the embedding model and building the catalog index once per
session. Reusing them across requests improves responsiveness without changing
which songs are available.

### Suggestions I rejected or revised

I challenged the embedding-only ranking after the calm-song failure. We added
mood and energy checks while keeping semantic retrieval for flexible wording.
I also asked for a shorter README and a simpler diagram because the original
documentation obscured the main flow with too much detail.

### Where AI overengineered, overexplained, or overcoded

- **Ranking:** fixing one mismatch grew into synonym groups, negation rules,
  energy targets, and several weights. The rules help, but duplicate parts of
  the original scorer and add assumptions that need more evidence.
- **Evaluation:** three retrieval strategies and four metrics support comparison,
  but are a lot to explain for eight requests. One clearly defined quality
  measure would have been a simpler starting point.
- **Infrastructure:** structured JSON, strict ordering checks, rotating logs,
  timing, and many test fixtures expanded a small command-line app. Some protect
  important boundaries; they should not be presented as proof of factual accuracy.
- **Setup:** the full dependency snapshot supports repeatability, but retaining
  unused starter dependencies such as pandas and Streamlit makes installation
  heavier than the current interface needs.
- **Communication:** repeated scoring explanations, long tables, duplicated
  setup instructions, and a busy diagram made a beginner project harder to
  follow. We shortened the documentation and used plain-language diagram labels.

The documentation was simplified; the additional code and dependencies remain.
My next revision would simplify shared scoring logic, review dependencies, and
allow fewer results when matches are weak. I would keep changes supported by
clear failures or measured improvements instead of adding features to look advanced.
