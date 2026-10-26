from guanani.normalize import normalize


def test_apostrophe_variants_become_puso():
    for v in "’‘´`ʼꞌ′‚":
        assert normalize(f"ka{v}i") == "ka'i"


def test_nfd_becomes_nfc():
    assert normalize("ãra") == "ãra"


def test_legacy_nasal_marks_from_jojajovai():
    assert normalize("omokyre'ÿ omopotî") == "omokyre'ỹ omopotĩ"
    assert normalize("Âva ära") == "Ãva ãra"


def test_g_tilde_variants_and_preserved():
    assert normalize("ĝuahẽ") == "g̃uahẽ"
    assert normalize("ḡuahẽ") == "g̃uahẽ"
    assert normalize("g̃uahẽ") == "g̃uahẽ"


def test_u_diaeresis_kept_for_spanish_loans():
    assert normalize("itaugüeño") == "itaugüeño"


def test_whitespace_and_zero_width():
    assert normalize("  che  ha​ʼe  \n") == "che ha'e"
