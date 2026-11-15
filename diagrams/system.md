# How the system works

```mermaid
flowchart TD
    A["You describe the music"] --> B["Find matching songs"]
    S["Song catalog + custom listening notes"] --> B
    B --> C["Gemini explains the choices"]
    C --> D["Check the answer and show results"]
    D --> H["You review the recommendations"]
    E["Evaluation checks song matches"] -.-> B
    T["Tests check code and errors"] -.-> D
```

- **Find songs:** compare the request with the catalog using embeddings and
  mood, genre, and energy preferences. Custom listening notes enrich the indexed
  song descriptions.
- **Explain and check:** send the request and three selected records to Gemini;
  include their custom notes and validate the answer's format and cited song
  IDs before showing explanations with source filenames.
  Retrieved songs remain visible if generation fails.
- **Automated checks:** `pytest` checks code behavior; `--evaluate` compares
  retrieval results with manually labeled song IDs.
- **Human review:** check whether recommendations fit and explanations agree
  with the catalog. This is a manual review, not an approval step in the app.
