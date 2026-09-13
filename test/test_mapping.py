"""Tests for pipeline.mapping — the single source of truth (spec §4)."""
import re

from pipeline.mapping import CELLS, SAMPLE_TEXTS, cell, ligature_rules

GID_RE = re.compile(r"^g_u[0-9A-F]{4}(_u[0-9A-F]{4})*$")

EXPECTED_KIND_COUNTS = {
    "uyir": 12, "ayutham": 1, "consonant": 18, "sign": 13,
    "pulli_form": 18, "uyirmei": 198, "numeral": 10, "grantha": 4,
    "digit": 10, "punct": 12,
}


def test_total_and_kind_counts():
    assert len(CELLS) == 296
    got = {}
    for c in CELLS:
        got[c.kind] = got.get(c.kind, 0) + 1
    assert got == EXPECTED_KIND_COUNTS


def test_gid_format_and_uniqueness():
    gids = [c.gid for c in CELLS]
    assert len(set(gids)) == len(gids)
    assert all(GID_RE.match(g) for g in gids)


def test_sheet_bookkeeping():
    placements = {(c.sheet, c.row, c.col) for c in CELLS}
    assert len(placements) == len(CELLS)
    used_sheets = {c.sheet for c in CELLS}
    assert used_sheets == {f"S{i}" for i in range(1, 13)}
    uyirmei_sizes = {}
    for c in CELLS:
        if c.kind == "uyirmei":
            key = c.sheet
            uyirmei_sizes[key] = max(uyirmei_sizes.get(key, (0, 0)), (c.row, c.col))
    # uyirmei sheets: 6 cols; group A (6 vowels) fills 6 rows, group B (5) fills 5
    for sheet, (row, col) in uyirmei_sizes.items():
        assert row <= 5 and col <= 5, sheet


def test_cmap_fields():
    for c in CELLS:
        if c.kind in ("uyirmei", "pulli_form"):
            assert c.cmap_cp is None
            assert len(c.cps) >= 2
        else:
            assert c.cmap_cp == c.cps[0]


def test_ligature_rules_count_and_targets():
    rules = ligature_rules()
    assert len(rules) == 270
    gids = {c.gid for c in CELLS}
    seqs = set()
    for seq, target in rules:
        assert target in gids
        assert seq
        assert seq not in seqs, seq
        seqs.add(seq)


def test_ligature_rules_longest_first():
    rules = ligature_rules()
    lengths = [len(s) for s, _ in rules]
    assert lengths == sorted(lengths, reverse=True)


def test_every_sign_combo_has_exactly_one_rule():
    from pipeline.mapping import CONSONANTS, VOWEL_SIGNS

    rules = {seq: gid for seq, gid in ligature_rules()}
    for c in CONSONANTS:
        for options in VOWEL_SIGNS.values():
            for seq in options:
                assert (c,) + tuple(seq) in rules
        assert (c, 0x0BCD) in rules


def test_sample_texts_resolvable():
    cmap_cps = {c.cmap_cp for c in CELLS if c.cmap_cp}
    rule_cps = {cp for seq, _ in ligature_rules() for cp in seq}
    for text in SAMPLE_TEXTS:
        for ch in text:
            if ch.isspace():
                continue
            assert ord(ch) in cmap_cps or ord(ch) in rule_cps, hex(ord(ch))


def test_cell_lookup():
    c = cell("g_u0B95_u0BBF")
    assert c.kind == "uyirmei"
    assert c.cps == (0x0B95, 0x0BBF)
