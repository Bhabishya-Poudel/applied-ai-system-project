# How the system works

```mermaid
flowchart TD
    A["Music request"] --> B["Song retrieval"]
    S["Song catalog + custom listening notes"] --> B
    B --> C["Gemini explanation"]
    C --> D["Answer checks and results"]
    D --> H["Human review"]
    E["Song-match evaluation"] -.-> B
    T["Code and error checks"] -.-> D
```

- **Song retrieval:** the request is compared with the catalog using embeddings and
  mood, genre, and energy preferences. Custom listening notes enrich the indexed
  song descriptions.
- **Explanation and validation:** Gemini receives the request, three selected
  records, and their custom notes. Answer format and cited song IDs are checked
  before explanations appear with source filenames.
  Retrieved songs remain visible if generation fails.
- **Automated checks:** `pytest` checks code behavior; `--evaluate` compares
  retrieval results with manually labeled song IDs.
- **Human review:** relevance and agreement with the catalog can be assessed
  manually. The app does not include an approval step.
