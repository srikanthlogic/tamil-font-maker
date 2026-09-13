"""Builds a tiny two-glyph TTF for rasterizer tests: 'a' filled square,
'b' square with a square hole (even-odd check)."""
from fontTools.fontBuilder import FontBuilder
from fontTools.pens.ttGlyphPen import TTGlyphPen


def _square(pen, x0, y0, x1, y1):
    pen.moveTo((x0, y0))
    pen.lineTo((x0, y1))
    pen.lineTo((x1, y1))
    pen.lineTo((x1, y0))
    pen.closePath()


def make_square_font(path):
    fb = FontBuilder(1000, isTTF=True)
    fb.setupGlyphOrder([".notdef", "a", "b", "space"])

    square = TTGlyphPen(None)
    _square(square, 100, 100, 700, 700)

    ring = TTGlyphPen(None)
    _square(ring, 100, 100, 700, 700)   # outer (winding will be XORed at raster)
    _square(ring, 250, 250, 550, 550)   # hole

    blank = TTGlyphPen(None)

    fb.setupCharacterMap({ord("a"): "a", ord("b"): "b", ord(" "): "space"})
    fb.setupGlyf({".notdef": blank.glyph(), "a": square.glyph(),
                  "b": ring.glyph(), "space": blank.glyph()})
    fb.setupHorizontalMetrics({".notdef": (400, 0), "a": (800, 100),
                               "b": (800, 100), "space": (400, 0)})
    fb.setupHorizontalHeader(ascent=800, descent=-200)
    fb.setupNameTable({"familyName": "Sq", "styleName": "Regular"})
    fb.setupOS2()
    fb.save(str(path))
