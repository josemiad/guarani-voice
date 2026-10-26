from collections import Counter

from guanani.train.corpus import (
    KUATIA_SUBSETS,
    build_examples,
    clean_pairs,
    clean_sentences,
    decont_key,
    passes_text_rules,
    reservoir,
    split_sentences,
)


def _p(gn, es, source="s"):
    return {"gn": gn, "es": es, "source": source}


def test_clean_pairs_filters_each_reason():
    pairs = [
        _p("", "algo"),
        _p("hola", "Hola"),
        _p(" ".join(["ñe'ẽ"] * 201), " ".join(["palabra"] * 201)),
        _p("peteĩ", "uno dos tres cuatro"),
        _p("che róga", "mi casa"),
        _p("Che  róga", "mi casa"),
        _p("ko ñe'ẽ", "esta frase"),
    ]
    kept, stats = clean_pairs(pairs, eval_keys={decont_key("esta frase")})
    assert kept == [{"gn": "che róga", "es": "mi casa", "source": "s"}]
    assert stats == {
        "in": 7, "empty": 1, "identical": 1, "too_long": 1, "ratio": 1, "duplicate": 1, "contaminated": 1,
    }


def test_decont_key_ignores_case_punctuation_and_apostrophe_variants():
    assert decont_key("¡Mba’éichapa, REIKO!") == decont_key("mba'éichapa reiko")


def test_contaminated_gn_side_removed_even_with_variant_spelling():
    keys = {decont_key("Omopotĩ hikuái.")}
    kept, stats = clean_pairs([_p("omopotî hikuái", "limpiaron")], keys)
    assert kept == [] and stats["contaminated"] == 1


def test_split_sentences():
    assert split_sentences("Peteĩ. Mokõi! Mbohapy? Irundy") == ["Peteĩ.", "Mokõi!", "Mbohapy?", "Irundy"]


def test_kuatia_whitelist_excludes_eval_and_informal_sources():
    for bad in ("flores-200", "belele", "jojajovai", "commonvoice", "americasnlp2024", "bible", "twitter_covid_py"):
        assert bad not in KUATIA_SUBSETS


def test_build_examples_real_pairs_both_directions_bt_only_es_gn():
    ex = build_examples([_p("che", "yo")], bt=[_p("ore", "nosotros", "bt:madlad400")])
    assert ex == [
        {"src": "che", "tgt": "yo", "src_lang": "grn_Latn", "tgt_lang": "spa_Latn"},
        {"src": "yo", "tgt": "che", "src_lang": "spa_Latn", "tgt_lang": "grn_Latn"},
        {"src": "nosotros", "tgt": "ore", "src_lang": "spa_Latn", "tgt_lang": "grn_Latn"},
    ]


def test_split_sentences_breaks_on_newlines_and_literal_escapes():
    text = "Peteĩ ñe'ẽ\\nMokõi ha mbohapy. Irundy\nPo\\tpoteĩ"
    assert split_sentences(text) == ["Peteĩ ñe'ẽ", "Mokõi ha mbohapy.", "Irundy", "Po poteĩ"]


def test_passes_text_rules_rejects_spanish_and_numbers_keeps_loans():
    assert passes_text_rules("Tetãvore Tarija ha'e peteĩ tetãvore ñemby gotyo Volíviape.")
    assert passes_text_rules("Ojeporu web ha whatsapp mbo'ehaópe opa ára.")
    assert not passes_text_rules("El gobierno de la ciudad anunció que las obras terminan en marzo.")
    assert not passes_text_rules("- 00:3000:30 10 jasyteĩ 2012 difs.")


def test_clean_sentences_counts_each_reason_per_source():
    docs = [(
        "wiki",
        "Mbykymi. Ko ñe'ẽ ipuku porã ha iporã. Ko ñe'ẽ ipuku porã ha iporã. "
        "El gobierno de la ciudad anunció que las obras terminan. "
        "Ambue ñe'ẽ ipuku porã avei ko'ápe. Ha'e ramo nhuã rarõa omboaxa pova'e kuery.",
    )]
    keys = {decont_key("Ambue ñe'ẽ ipuku porã avei ko'ápe.")}
    stats, seen = Counter(), set()
    got = list(clean_sentences(docs, keys, stats, seen, lambda s: "gun_Latn" if "nhuã" in s else None))
    assert got == [("wiki", "Ko ñe'ẽ ipuku porã ha iporã.")]
    assert stats == Counter({
        "wiki:longitud": 1, "wiki:aceptada": 1, "wiki:duplicada": 1,
        "wiki:reglas": 1, "wiki:contaminada": 1, "wiki:lid:gun_Latn": 1,
    })


def test_clean_sentences_dedups_across_calls_sharing_seen():
    stats, seen = Counter(), set()
    doc = "Ko ñe'ẽ ipuku porã ha iporã."
    assert len(list(clean_sentences([("wikipedia", doc)], set(), stats, seen))) == 1
    assert list(clean_sentences([("madlad400", doc)], set(), stats, seen)) == []
    assert stats["madlad400:duplicada"] == 1


def test_reservoir_deterministic_and_bounded():
    a = reservoir(range(100), 10, seed=3)
    assert len(a) == 10 and a == reservoir(range(100), 10, seed=3)
    assert reservoir(range(5), 10) == [0, 1, 2, 3, 4]
    assert reservoir(range(5), 0) == []


def test_build_examples_bt_tag_only_on_synthetic():
    ex = build_examples([_p("che", "yo")], bt=[_p("ore", "nosotros", "bt:x")], bt_tag=True)
    assert [e["src"] for e in ex] == ["che", "yo", "[BT] nosotros"]
