"""Check retrieval, grounded prompts, API failures, and the interactive loop."""

import json
import logging
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest

from src import embeddings


@pytest.fixture
def rag(monkeypatch):
    class FakeModel:
        def encode(self, texts, **kwargs):
            if isinstance(texts, str):
                return np.array([1.0, 0.0])
            vectors = np.tile([0.0, 1.0], (len(texts), 1))
            vectors[0] = [1.0, 0.0]
            vectors[1] = [0.8, 0.6]
            vectors[2] = [0.6, 0.8]
            return vectors

    monkeypatch.setattr(embeddings, "SentenceTransformer", lambda *a, **kw: FakeModel())
    return embeddings.SongRAG()


def test_retrieve_returns_top_k_in_similarity_order(rag):
    matches = rag.retrieve("something to listen to", k=2)
    assert [song["id"] for song, _ in matches] == ["1", "2"]
    assert [score for _, score in matches] == pytest.approx([1.0, 0.8])


@pytest.mark.parametrize(
    "query", ["calm", "calm songs", "CALM music!", "relaxing songs", "low-energy music"]
)
def test_calm_request_overrides_misleading_embedding_scores(rag, query):
    # Reproduce the failure: the embedding model prefers Gym Hero over calm songs.
    rag.song_embeddings[:] = [0.0, 1.0]
    gym_index = next(i for i, song in enumerate(rag.songs) if song["id"] == "5")
    rag.song_embeddings[gym_index] = [1.0, 0.0]

    matches = rag.retrieve(query)
    assert all(float(song["energy"]) < 0.5 for song, _ in matches)
    assert all(song["id"] != "5" for song, _ in matches)
    all_ids = [song["id"] for song, _ in rag.retrieve(query, k=len(rag.songs))]
    assert all_ids.index("2") < all_ids.index("5")


def test_explicit_genre_outweighs_unrelated_embedding_match(rag):
    assert rag.retrieve("rock", k=1)[0][0]["id"] == "3"


def test_high_energy_request_does_not_apply_calm_preference(rag):
    matches = rag.retrieve("energetic music for a workout")
    assert all(float(song["energy"]) >= 0.75 for song, _ in matches)


def test_negated_calm_does_not_boost_calm_songs(rag):
    moods, _, energy = embeddings.query_preferences("not calm", rag.songs)
    assert energy is None
    assert embeddings.preference_fit({"calm"}, moods) == 0
    assert embeddings.preference_fit({"intense"}, moods) == 1


def test_retrieve_rejects_empty_request(rag):
    with pytest.raises(ValueError, match="music request"):
        rag.retrieve("   ")


def test_prompt_contains_only_retrieved_songs(rag, monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    client = MagicMock()
    client.models.generate_content.return_value = SimpleNamespace(
        text=json.dumps({"recommendations": [{"song_id": "1", "reason": "Its happy mood fits your request."}]}),
        candidates=[SimpleNamespace(finish_reason=embeddings.types.FinishReason.STOP)],
    )
    factory = MagicMock()
    factory.return_value.__enter__.return_value = client
    monkeypatch.setattr(embeddings.genai, "Client", factory)

    answer = rag.explain("happy pop", rag.retrieve("happy pop", k=1))

    assert "Sunrise City by Neon Echo (ID: 1): Its happy mood fits your request." in answer
    prompt = client.models.generate_content.call_args.kwargs["contents"]
    assert "happy pop" in prompt
    assert "Song ID: 1\nSource: data/songs.csv" in prompt
    assert "Sunrise City" in prompt
    assert "Midnight Coding" not in prompt
    assert "Storm Runner" not in prompt
    config = client.models.generate_content.call_args.kwargs["config"]
    assert config.response_mime_type == "application/json"
    assert config.response_schema


def test_missing_key_preserves_retrieval(rag, monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    matches = rag.retrieve("happy pop")
    assert len(matches) == 3
    assert "Set GEMINI_API_KEY" in rag.explain("happy pop", matches)


def test_quota_failure_preserves_retrieval(rag, monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    client = MagicMock()
    client.models.generate_content.side_effect = embeddings.errors.ClientError(
        429, {"error": {"message": "quota reached", "status": "RESOURCE_EXHAUSTED"}}
    )
    factory = MagicMock()
    factory.return_value.__enter__.return_value = client
    monkeypatch.setattr(embeddings.genai, "Client", factory)
    assert "quota or rate limit" in rag.explain("happy pop", rag.retrieve("happy pop"))


def test_interactive_loop_reuses_index_and_handles_blank_input(monkeypatch, capsys):
    requests = iter(["", "calm music", "energetic rock", "quit"])
    monkeypatch.setattr("builtins.input", lambda _: next(requests))
    fake = MagicMock()
    fake.retrieve.return_value = [({"id": "1", "title": "Song", "artist": "Artist"}, 0.8)]
    fake.explain.return_value = "Explanation"
    factory = MagicMock(return_value=fake)
    monkeypatch.setattr(embeddings, "SongRAG", factory)

    embeddings.main()

    factory.assert_called_once()
    assert fake.retrieve.call_count == 2
    output = capsys.readouterr().out
    assert "Please enter a music request" in output
    assert output.count("Explanation") == 2
    assert "Goodbye!" in output


@pytest.mark.parametrize("query", ["", " " * 4, "x" * 501])
def test_invalid_requests_never_reach_embedding_model(rag, query):
    rag.model = MagicMock()
    with pytest.raises(ValueError):
        rag.retrieve(query)
    rag.model.encode.assert_not_called()


@pytest.mark.parametrize(
    "content",
    [
        "id,title\n1,Song\n",
        "id,title,artist,genre,mood,mood_tags,energy,acousticness\n",
        "id,title,artist,genre,mood,mood_tags,energy,acousticness\n1,T,A,pop,happy,warm,NaN,0.5\n",
        "id,title,artist,genre,mood,mood_tags,energy,acousticness\n1,T,A,pop,happy,warm,1.5,0.5\n",
        "id,title,artist,genre,mood,mood_tags,energy,acousticness\n1,T,A,pop,happy,warm,0.5,0.5\n1,T,A,pop,happy,warm,0.5,0.5\n",
    ],
)
def test_invalid_catalog_is_rejected(tmp_path, content):
    path = tmp_path / "songs.csv"
    path.write_text(content)
    with pytest.raises(ValueError):
        embeddings.load_catalog(path)


@pytest.mark.parametrize(
    "response_text",
    [
        "not json",
        '{"recommendations": []}',
        '{"recommendations": [{"song_id": "999", "reason": "Invented recommendation"}]}',
        '{"recommendations": [{"song_id": "1", "reason": ""}]}',
        '{"recommendations": [{"song_id": "1", "reason": 42}]}',
    ],
)
def test_invalid_generation_is_withheld(rag, monkeypatch, response_text):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    factory = MagicMock()
    factory.return_value.__enter__.return_value.models.generate_content.return_value = SimpleNamespace(
        text=response_text,
        candidates=[SimpleNamespace(finish_reason=embeddings.types.FinishReason.STOP)],
    )
    monkeypatch.setattr(embeddings.genai, "Client", factory)
    answer = rag.explain("happy pop", rag.retrieve("happy pop", k=1))
    assert "invalid explanation" in answer
    assert "Invented recommendation" not in answer


def test_duplicate_and_reordered_citations_are_rejected(rag):
    matches = rag.retrieve("anything", k=2)
    ids = [song["id"] for song, _ in matches]
    for wrong_ids in ([ids[0], ids[0]], ids[::-1]):
        text = json.dumps({"recommendations": [{"song_id": song_id, "reason": "Reason"} for song_id in wrong_ids]})
        with pytest.raises(ValueError):
            embeddings.render_explanation(text, matches)


def test_incomplete_generation_is_withheld(rag, monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    factory = MagicMock()
    factory.return_value.__enter__.return_value.models.generate_content.return_value = SimpleNamespace(
        text='{"recommendations":',
        candidates=[SimpleNamespace(finish_reason=embeddings.types.FinishReason.MAX_TOKENS)],
    )
    monkeypatch.setattr(embeddings.genai, "Client", factory)
    assert "No complete explanation" in rag.explain("happy pop", rag.retrieve("happy pop"))


def test_logs_do_not_record_requests_keys_or_api_error_bodies(rag, monkeypatch, caplog):
    monkeypatch.setenv("GEMINI_API_KEY", "secret-api-key-example")
    factory = MagicMock()
    factory.return_value.__enter__.return_value.models.generate_content.side_effect = embeddings.errors.ClientError(
        403, {"error": {"message": "secret-api-key-example", "status": "PERMISSION_DENIED"}}
    )
    monkeypatch.setattr(embeddings.genai, "Client", factory)
    with caplog.at_level(logging.INFO, logger="music_rag"):
        query = "private-request-example"
        answer = rag.explain(query, rag.retrieve(query))
    assert "retrieval_complete" in caplog.text
    assert "generation_failed" in caplog.text
    assert "secret-api-key-example" not in caplog.text + answer
    assert "private-request-example" not in caplog.text


def test_model_startup_failure_is_reported_without_traceback(monkeypatch, capsys):
    monkeypatch.setattr("builtins.input", lambda _: "calm")
    monkeypatch.setattr(embeddings, "SongRAG", MagicMock(side_effect=OSError("sensitive error detail")))
    assert embeddings.main() == 1
    output = capsys.readouterr().out
    assert "Unable to load" in output
    assert "sensitive error detail" not in output


def test_main_entry_point_invokes_rag(monkeypatch):
    from src import main
    monkeypatch.setattr("sys.argv", ["music-recommender"])
    runner = MagicMock(return_value=0)
    monkeypatch.setattr(embeddings, "main", runner)
    assert main.main() == 0
    runner.assert_called_once()


def test_main_entry_point_invokes_evaluation(monkeypatch):
    from src import evaluate, main
    monkeypatch.setattr("sys.argv", ["music-recommender", "--evaluate"])
    runner = MagicMock(return_value=0)
    monkeypatch.setattr(evaluate, "main", runner)
    assert main.main() == 0
    runner.assert_called_once()


def test_custom_document_changes_retrieval(monkeypatch):
    class NoteAwareModel:
        def encode(self, texts, **kwargs):
            if isinstance(texts, str):
                return np.array([1.0, 0.0])
            return np.array([[1.0, 0.0] if "Pomodoro" in text else [0.0, 1.0] for text in texts])

    monkeypatch.setattr(embeddings, "SentenceTransformer", lambda *a, **kw: NoteAwareModel())
    before = embeddings.SongRAG(use_notes=False)
    after = embeddings.SongRAG()
    query = "a pomodoro session revising lecture notes"
    assert before.retrieve(query, k=1)[0][0]["id"] != "9"
    assert after.retrieve(query, k=1)[0][0]["id"] == "9"


def test_retrieved_custom_notes_reach_gemini_and_sources_are_displayed(rag, monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    song = next(song for song in rag.songs if song["id"] == "9")
    factory = MagicMock()
    factory.return_value.__enter__.return_value.models.generate_content.return_value = SimpleNamespace(
        text=json.dumps({"recommendations": [{"song_id": "9", "reason": "The custom notes suggest it for a Pomodoro session."}]}),
        candidates=[SimpleNamespace(finish_reason=embeddings.types.FinishReason.STOP)],
    )
    monkeypatch.setattr(embeddings.genai, "Client", factory)
    answer = rag.explain("music for a Pomodoro session", [(song, 0.8)])
    call = factory.return_value.__enter__.return_value.models.generate_content.call_args
    assert "data/listening_notes.md" in call.kwargs["contents"]
    assert "revising\nlecture notes" in call.kwargs["contents"]
    assert "Midnight Coding" not in call.kwargs["contents"]
    assert "Sources: data/songs.csv, data/listening_notes.md" in answer


@pytest.mark.parametrize(
    "content",
    ["No sections", "## Song ID: 999\nUnknown song", "## Song ID: 1\n", "## Song ID: 1\nOne\n## Song ID: 1\nTwo"],
)
def test_invalid_custom_documents_are_rejected(rag, tmp_path, content):
    path = tmp_path / "notes.md"
    path.write_text(content)
    with pytest.raises(ValueError):
        embeddings.add_listening_notes(rag.songs, path)
