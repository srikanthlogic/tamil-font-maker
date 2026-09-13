"""Single source of truth for the Tamil glyph set (spec §4).

Every other stage derives from this table; nothing else hardcodes Tamil.
Counts (locked by test_mapping.py): 296 cells, 306 GSUB ligature rules.
"""
from dataclasses import dataclass

UPM = 2048

# --- codepoint tables -------------------------------------------------------

UYIR = (0x0B85, 0x0B86, 0x0B87, 0x0B88, 0x0B89, 0x0B8A,
        0x0B8E, 0x0B8F, 0x0B90, 0x0B92, 0x0B93, 0x0B94)
AYUTHAM = 0x0B83
CONSONANTS = (0x0B95, 0x0B99, 0x0B9A, 0x0B9E, 0x0B9F, 0x0BA3,
              0x0BA4, 0x0BA8, 0x0BAA, 0x0BAE, 0x0BAF, 0x0BB0,
              0x0BB2, 0x0BB5, 0x0BB4, 0x0BB3, 0x0BB1, 0x0BA9)
PULLI = 0x0BCD
# vowel codepoint -> sign options; first = composed codepoint, others =
# NFC-decomposed sequences that must ligate to the same precomposed cell.
VOWEL_SIGNS = {
    0x0B86: ((0x0BBE,),),
    0x0B87: ((0x0BBF,),),
    0x0B88: ((0x0BC0,),),
    0x0B89: ((0x0BC1,),),
    0x0B8A: ((0x0BC2,),),
    0x0B8E: ((0x0BC6,),),
    0x0B8F: ((0x0BC7,),),
    0x0B90: ((0x0BC8,),),
    0x0B92: ((0x0BCA,), (0x0BC6, 0x0BBE)),
    0x0B93: ((0x0BCB,), (0x0BC7, 0x0BBE)),
    # ௌ U+0BCC = U+0BC6 + U+0BD7 (canonical decomposition; keep both forms).
    # The independent vowel ஔ itself never follows a consonant — only its sign.
    0x0B94: ((0x0BCC,), (0x0BC6, 0x0BD7)),
}
SIGN_CPS = (0x0BBE, 0x0BBF, 0x0BC0, 0x0BC1, 0x0BC2, 0x0BC6,
            0x0BC7, 0x0BC8, 0x0BCA, 0x0BCB, 0x0BCC, 0x0BD7, 0x0BCD)
NUMERALS = tuple(range(0x0BE6, 0x0BF0))
GRANTHA = (0x0B9C, 0x0BB7, 0x0BB8, 0x0BB9)
DIGITS = tuple(range(0x30, 0x3A))
PUNCT = tuple(ord(ch) for ch in ".,;:!?-'\"()/")

# --- cell model -------------------------------------------------------------

@dataclass(frozen=True)
class Cell:
    gid: str            # "g_u0B95_u0BBF"
    cps: tuple          # codepoint sequence (ligature cells: consonant+signs)
    kind: str           # uyir|ayutham|consonant|sign|pulli_form|uyirmei|numeral|grantha|digit|punct
    sheet: str          # "S1".."S12"
    row: int
    col: int
    cmap_cp: int | None # direct cmap mapping (None for pulli_form/uyirmei)


def _gid(cps) -> str:
    return "g_" + "_".join(f"u{cp:04X}" for cp in cps)


def _g(*cps: int) -> tuple:
    return tuple(cps)


# Build the complete ordered cell list with deterministic sheet placement.
def _cells() -> list[Cell]:
    result: list[Cell] = []
    cursor: dict[str, int] = {}  # next free slot per sheet

    def add(kind: str, items: list[tuple], cols: int, sheet: str):
        i = cursor.get(sheet, 0)
        for cps in items:
            result.append(Cell(
                gid=_gid(cps) if len(cps) > 1 else f"g_u{cps[0]:04X}",
                cps=cps,
                kind=kind,
                sheet=sheet,
                row=i // cols,
                col=i % cols,
                cmap_cp=cps[0] if kind != "uyirmei" and kind != "pulli_form" else None,
            ))
            i += 1
        cursor[sheet] = i

    add("uyir", [_g(cp) for cp in UYIR], 5, "S1")
    add("ayutham", [_g(AYUTHAM)], 5, "S1")
    add("consonant", [_g(cp) for cp in CONSONANTS], 5, "S2")
    add("sign", [_g(cp) for cp in SIGN_CPS], 5, "S3")
    add("pulli_form", [_g(cp, PULLI) for cp in CONSONANTS], 5, "S4")

    # uyirmei: canonical order (consonant-major, vowel-minor), 25 cells per
    # 5x5 portrait sheet -> S5..S12 (A4 300dpi fits 5 columns of 400px, not 6)
    uyirmei_items = []
    for cv in CONSONANTS:
        for vv, options in VOWEL_SIGNS.items():
            uyirmei_items.append(_g(cv, *options[0]))
    for offset in range(0, len(uyirmei_items), 25):
        add("uyirmei", uyirmei_items[offset:offset + 25], 5, f"S{5 + offset // 25}")

    add("numeral", [_g(cp) for cp in NUMERALS], 5, "S13")
    add("grantha", [_g(cp) for cp in GRANTHA], 5, "S13")
    add("digit", [_g(cp) for cp in DIGITS], 5, "S14")
    add("punct", [_g(cp) for cp in PUNCT], 5, "S14")
    return result


CELLS: tuple[Cell, ...] = tuple(_cells())
_BY_GID = {c.gid: c for c in CELLS}


def cell(gid: str) -> Cell:
    return _BY_GID[gid]


def ligature_rules() -> list[tuple[tuple[int, ...], str]]:
    """(codepoint sequence) -> target gid, longest sequences first. 270 rules.

    15 per consonant: 14 uyirmei (8 simple matras; ொ ோ ௌ each in composed and
    NFC-decomposed form) + 1 pulli. C + independent-vowel sequences (e.g.
    க+ஔ) are intentionally not ligated — real fonts render them as two glyphs.
    """
    # uyirmei cells store (consonant, composed sign); recover the vowel via the
    # composed option's first codepoint.
    vowel_by_sign = {opts[0][0]: vowel for vowel, opts in VOWEL_SIGNS.items()}
    rules: list[tuple[tuple[int, ...], str]] = []
    for c in CELLS:
        if c.kind == "pulli_form":
            rules.append((c.cps, c.gid))
        elif c.kind == "uyirmei":
            base, vowel = c.cps[0], vowel_by_sign[c.cps[1]]
            for option in VOWEL_SIGNS[vowel]:
                rules.append(((base,) + tuple(option), c.gid))
    rules.sort(key=lambda r: (-len(r[0]), r[0]))
    return rules


SAMPLE_TEXTS = (
    "தமிழ் எழுத்துக் கலை.",
    "ஆண்டு 2026 ஆகும்.",
    "ஃஜஷஸஹ (க-ன்) ௦௧௨௩",
)
