"""
Tests for data loaders (src/data/loaders.py) and
training orchestrator (src/training/trainer.py).
"""
import sys, os, json, textwrap, tempfile
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from pathlib import Path
import torch
import pytest

from src.data.loaders import load_tsv, load_ntriples, load_qald, load_lcquad, QALDDataset
from src.training.trainer import TLAQTrainer, TrainConfig


# ═══════════════════════════════════════════════════════════════════════════
# Helpers — write temp files
# ═══════════════════════════════════════════════════════════════════════════

def write_temp(content: str, suffix: str) -> Path:
    f = tempfile.NamedTemporaryFile(
        mode="w", suffix=suffix, delete=False, encoding="utf-8"
    )
    f.write(textwrap.dedent(content))
    f.close()
    return Path(f.name)


# ═══════════════════════════════════════════════════════════════════════════
# ① TSV loader
# ═══════════════════════════════════════════════════════════════════════════

class TestLoadTSV:

    TSV_3COL = """\
        Messi\tplayFor\tFCBarcelona
        Neymar\tplayFor\tPSG
    """
    TSV_4COL = """\
        Messi\tplayFor\tFCBarcelona\t2004
        Neymar\tplayFor\tPSG\t2017
    """
    TSV_5COL = """\
        Messi\tplayFor\tFCBarcelona\t2004\t2021
        Suarez\tserves\tFCBarcelona\t2014\t2020
        Neymar\tplayFor\tPSG\t2017\t-
    """
    TSV_COMMENTS = """\
        # this is a header comment
        Messi\tplayFor\tFCBarcelona\t2004\t2021
        # another comment
        Neymar\tplayFor\tPSG\t2017\t-
    """

    def test_3col_entity_count(self):
        path = write_temp(self.TSV_3COL, ".tsv")
        ds = load_tsv(path)
        assert ds.num_entities >= 3   # Messi, FCBarcelona, Neymar, PSG

    def test_4col_time_start(self):
        path = write_temp(self.TSV_4COL, ".tsv")
        ds = load_tsv(path)
        messi_triples = [t for t in ds.relation_triples
                         if t.head == ds.entity2id["Messi"]]
        assert messi_triples[0].time_start == 2004

    def test_5col_time_end(self):
        path = write_temp(self.TSV_5COL, ".tsv")
        ds = load_tsv(path)
        messi_triples = [t for t in ds.relation_triples
                         if t.head == ds.entity2id["Messi"]]
        assert messi_triples[0].time_end == 2021

    def test_5col_none_time_end(self):
        """'-' time_end should be parsed as None."""
        path = write_temp(self.TSV_5COL, ".tsv")
        ds = load_tsv(path)
        neymar_triples = [t for t in ds.relation_triples
                          if t.head == ds.entity2id["Neymar"]]
        assert neymar_triples[0].time_end is None

    def test_comments_ignored(self):
        path = write_temp(self.TSV_COMMENTS, ".tsv")
        ds = load_tsv(path)
        assert len(ds.relation_triples) == 2

    def test_indices_built(self):
        path = write_temp(self.TSV_3COL, ".tsv")
        ds = load_tsv(path)
        assert ds.num_entities > 0
        assert ds.num_relations > 0

    def test_build_indices_called(self):
        """head_triples index must be populated."""
        path = write_temp(self.TSV_4COL, ".tsv")
        ds = load_tsv(path)
        messi_id = ds.entity2id["Messi"]
        assert messi_id in ds.head_triples


# ═══════════════════════════════════════════════════════════════════════════
# ② N-Triples loader
# ═══════════════════════════════════════════════════════════════════════════

class TestLoadNTriples:

    NT_BASIC = """\
        <http://example.org/Messi> <http://example.org/playFor> <http://example.org/FCBarcelona> .
        <http://example.org/Neymar> <http://example.org/playFor> <http://example.org/PSG> .
    """
    NT_LITERAL = """\
        <http://example.org/Messi> <http://example.org/birthYear> "1987"^^<http://www.w3.org/2001/XMLSchema#gYear> .
    """
    NT_ATTR = """\
        <http://example.org/Messi> <http://example.org/nationality> "Argentine" .
    """

    def test_uri_triples_loaded(self):
        path = write_temp(self.NT_BASIC, ".nt")
        ds = load_ntriples(path)
        assert "Messi" in ds.entity2id
        assert "FCBarcelona" in ds.entity2id

    def test_relation_extracted(self):
        path = write_temp(self.NT_BASIC, ".nt")
        ds = load_ntriples(path)
        assert "playFor" in ds.relation2id

    def test_literal_year_stored(self):
        path = write_temp(self.NT_LITERAL, ".nt")
        ds = load_ntriples(path)
        messi_triples = [t for t in ds.relation_triples
                         if t.head == ds.entity2id.get("Messi", -1)]
        assert any(t.time_start == 1987 for t in messi_triples)

    def test_attr_triple_loaded(self):
        path = write_temp(self.NT_ATTR, ".nt")
        ds = load_ntriples(path)
        assert len(ds.attribute_triples) >= 1

    def test_indices_built(self):
        path = write_temp(self.NT_BASIC, ".nt")
        ds = load_ntriples(path)
        assert ds.num_entities >= 2


# ═══════════════════════════════════════════════════════════════════════════
# ③ QALD loader  (real QALD-6/7 format)
# ═══════════════════════════════════════════════════════════════════════════

# Mirrors the actual ag-sc/QALD multilingual JSON structure
QALD_JSON = {
    "dataset": {"id": "qald-7-train-multilingual"},
    "questions": [
        {
            "id": "1",
            "answertype": "resource",
            "question": [{"language": "en", "string": "Who plays for FCBarcelona?",
                          "keywords": "FCBarcelona, plays for"}],
            "query": {"sparql": "SELECT DISTINCT ?uri WHERE { ?uri dbo:team dbr:FC_Barcelona }"},
            "answers": [{"head": {"vars": ["uri"]}, "results": {"bindings": [
                {"uri": {"type": "uri", "value": "http://dbpedia.org/resource/Messi"}},
                {"uri": {"type": "uri", "value": "http://dbpedia.org/resource/Suarez"}},
            ]}}],
        },
        {
            "id": "2",
            "answertype": "resource",
            "question": [{"language": "en", "string": "Where was Messi born?",
                          "keywords": "Messi, born"}],
            "query": {"sparql": "SELECT DISTINCT ?uri WHERE { dbr:Messi dbo:birthPlace ?uri }"},
            "answers": [{"head": {"vars": ["uri"]}, "results": {"bindings": [
                {"uri": {"type": "uri", "value": "http://dbpedia.org/resource/Argentina"}}
            ]}}],
        },
    ]
}


class TestLoadQALD:

    @pytest.fixture
    def qald_path(self, tmp_path):
        p = tmp_path / "qald.json"
        p.write_text(json.dumps(QALD_JSON), encoding="utf-8")
        return p

    def test_returns_qald_dataset(self, qald_path):
        qs = load_qald(qald_path)
        assert isinstance(qs, QALDDataset)

    def test_question_count(self, qald_path):
        qs = load_qald(qald_path)
        assert len(qs) == 2

    def test_answer_entities_in_vocab(self, qald_path):
        qs = load_qald(qald_path)
        for ans in qs.questions[0]["answers"]:
            assert ans in qs.tkg.entity2id, f"{ans} not in entity vocab"

    def test_answer_entity_ids(self, qald_path):
        qs = load_qald(qald_path)
        ids = qs.answer_entity_ids(0)
        assert isinstance(ids, set)
        assert len(ids) >= 1

    def test_sparql_stored(self, qald_path):
        qs = load_qald(qald_path)
        assert "SELECT" in qs.questions[0]["sparql"]

    def test_local_name_resolution(self, qald_path):
        """DBpedia URIs should be shortened to local names (e.g. 'Messi')."""
        qs = load_qald(qald_path)
        assert "Messi" in qs.tkg.entity2id

    def test_multi_answer_question(self, qald_path):
        """Question 1 has two answers."""
        qs = load_qald(qald_path)
        assert len(qs.questions[0]["answers"]) == 2


# ═══════════════════════════════════════════════════════════════════════════
# ④ LC-QuAD 1.0 loader
# ═══════════════════════════════════════════════════════════════════════════

# Mirrors the actual AskNowQA/LC-QuAD train.json / test.json structure
LCQUAD_JSON = [
    {
        "_id": "1",
        "corrected_question": "What is the birth place of Messi?",
        "sparql_query": "SELECT DISTINCT ?uri WHERE { dbr:Messi dbo:birthPlace ?uri }",
        "sparql_template_id": 1,
        "subgraph": "dbo:birthPlace",
        "entities":  ["http://dbpedia.org/resource/Messi"],
        "relations": ["http://dbpedia.org/ontology/birthPlace"],
        "answer": [
            {"type": "uri", "value": "http://dbpedia.org/resource/Argentina"}
        ]
    },
    {
        "_id": "2",
        "corrected_question": "Which team does Neymar play for?",
        "sparql_query": "SELECT DISTINCT ?uri WHERE { dbr:Neymar dbo:team ?uri }",
        "sparql_template_id": 2,
        "subgraph": "dbo:team",
        "entities":  ["http://dbpedia.org/resource/Neymar"],
        "relations": ["http://dbpedia.org/ontology/team"],
        "answer": [
            {"type": "uri", "value": "http://dbpedia.org/resource/PSG"}
        ]
    },
]


class TestLoadLCQuAD:

    @pytest.fixture
    def lcquad_path(self, tmp_path):
        p = tmp_path / "train.json"
        p.write_text(json.dumps(LCQUAD_JSON), encoding="utf-8")
        return p

    def test_returns_qald_dataset(self, lcquad_path):
        qs = load_lcquad(lcquad_path)
        assert isinstance(qs, QALDDataset)

    def test_question_count(self, lcquad_path):
        qs = load_lcquad(lcquad_path)
        assert len(qs) == 2

    def test_answer_entities_in_vocab(self, lcquad_path):
        qs = load_lcquad(lcquad_path)
        for ans in qs.questions[0]["answers"]:
            assert ans in qs.tkg.entity2id, f"{ans} not in entity vocab"

    def test_answer_entity_ids(self, lcquad_path):
        qs = load_lcquad(lcquad_path)
        ids = qs.answer_entity_ids(0)
        assert isinstance(ids, set)
        assert len(ids) == 1   # Argentina

    def test_entity_annotations_registered(self, lcquad_path):
        """Entities from 'entities' list should be in vocab."""
        qs = load_lcquad(lcquad_path)
        assert "Messi" in qs.tkg.entity2id

    def test_relation_annotations_registered(self, lcquad_path):
        """Relations from 'relations' list should be in vocab."""
        qs = load_lcquad(lcquad_path)
        assert "birthPlace" in qs.tkg.relation2id

    def test_sparql_stored(self, lcquad_path):
        qs = load_lcquad(lcquad_path)
        assert "SELECT" in qs.questions[0]["sparql"]

    def test_local_name_resolution(self, lcquad_path):
        """DBpedia URIs should resolve to local names."""
        qs = load_lcquad(lcquad_path)
        assert "Argentina" in qs.tkg.entity2id


# ═══════════════════════════════════════════════════════════════════════════
# ④ TLAQTrainer
# ═══════════════════════════════════════════════════════════════════════════

@pytest.fixture
def tiny_dataset():
    from src.data.tkg_dataset import TKGDataset
    ds = TKGDataset()
    ds.add_relation_triple("Messi",  "playFor", "FCBarcelona", 2004, 2021)
    ds.add_relation_triple("Suarez", "serves",  "FCBarcelona", 2014, 2020)
    ds.add_relation_triple("Neymar", "playFor", "PSG",         2017, None)
    ds.add_attribute_triple("Messi",  "bornIn", "Argentina")
    ds.add_attribute_triple("Suarez", "bornIn", "Uruguay")
    ds.build_indices()
    return ds


class TestTLAQTrainer:

    @pytest.fixture
    def fast_config(self):
        return TrainConfig(
            embed_dim   = 16,
            gcn_epochs  = 5,
            hyte_epochs = 5,
            gcn_log_every  = 5,
            hyte_log_every = 5,
            patience    = 99,   # disable early stop for these short tests
        )

    def test_train_returns_pipeline(self, tiny_dataset, fast_config):
        from src.pipeline.tlaq_pipeline import TLAQPipeline
        trainer = TLAQTrainer(tiny_dataset, fast_config)
        pipeline, history = trainer.train()
        assert isinstance(pipeline, TLAQPipeline)

    def test_history_gcn_losses(self, tiny_dataset, fast_config):
        trainer = TLAQTrainer(tiny_dataset, fast_config)
        _, history = trainer.train()
        assert len(history.gcn_losses) > 0
        assert all(isinstance(l, float) for l in history.gcn_losses)

    def test_history_hyte_losses(self, tiny_dataset, fast_config):
        trainer = TLAQTrainer(tiny_dataset, fast_config)
        _, history = trainer.train()
        assert len(history.hyte_losses) > 0

    def test_gcn_loss_finite(self, tiny_dataset, fast_config):
        import math
        trainer = TLAQTrainer(tiny_dataset, fast_config)
        _, history = trainer.train()
        assert all(math.isfinite(l) for l in history.gcn_losses)

    def test_pipeline_entity_query_works(self, tiny_dataset, fast_config):
        trainer  = TLAQTrainer(tiny_dataset, fast_config)
        pipeline, _ = trainer.train()
        messi_id = tiny_dataset.entity2id["Messi"]
        play_id  = tiny_dataset.relation2id["playFor"]
        results  = pipeline.entity_query([(messi_id, play_id, -1, 2004, 2021)])
        assert isinstance(results, list)

    def test_early_stopping(self, tiny_dataset):
        """With patience=1, training should stop after the second log chunk."""
        cfg = TrainConfig(
            embed_dim      = 16,
            gcn_epochs     = 100,
            hyte_epochs    = 100,
            gcn_log_every  = 5,
            hyte_log_every = 5,
            patience       = 1,
            min_delta      = 1e10,  # impossible to beat → triggers immediately
        )
        trainer = TLAQTrainer(tiny_dataset, cfg)
        _, history = trainer.train()
        # patience=1 means stop after 2 chunks: losses ≤ 2 * log_every = 10
        assert len(history.gcn_losses) <= 10

    def test_evaluate_returns_scores(self, tiny_dataset, fast_config):
        trainer  = TLAQTrainer(tiny_dataset, fast_config)
        pipeline, _ = trainer.train()

        play_id  = tiny_dataset.relation2id["playFor"]
        messi_id = tiny_dataset.entity2id["Messi"]
        fcb_id   = tiny_dataset.entity2id["FCBarcelona"]

        scores = trainer.evaluate(
            pipeline,
            entity_queries=[(messi_id, play_id, -1, 2004, 2021)],
            entity_gt=[{fcb_id}],
            top_k=5,
        )
        assert "entity" in scores
        for v in scores["entity"].values():
            assert 0.0 <= v <= 1.0

    def test_no_attr_triples_stage2_skipped(self):
        """Dataset with no attribute triples should skip Stage 2 gracefully."""
        from src.data.tkg_dataset import TKGDataset
        ds = TKGDataset()
        ds.add_relation_triple("A", "rel", "B", 2000, 2010)
        ds.build_indices()
        cfg = TrainConfig(embed_dim=8, gcn_epochs=3, hyte_epochs=3,
                          gcn_log_every=3, hyte_log_every=3, patience=99)
        trainer = TLAQTrainer(ds, cfg)
        pipeline, history = trainer.train()
        assert history.hyte_losses == []   # skipped
