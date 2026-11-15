"""Fill the A1 poster template with the final-report content (python-pptx, native shapes/charts/table)."""
import copy, sys
from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE, MSO_CONNECTOR
from pptx.chart.data import CategoryChartData
from pptx.enum.chart import XL_CHART_TYPE, XL_LEGEND_POSITION, XL_LABEL_POSITION
from pptx.oxml.ns import qn
from lxml import etree

SRC, OUT = sys.argv[1], sys.argv[2]
prs = Presentation(SRC)
s = prs.slides[0]

PURPLE = RGBColor(0x47, 0x24, 0x7B)
INK = RGBColor(0x26, 0x26, 0x26)
GREY = RGBColor(0x59, 0x59, 0x59)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
BLUE = RGBColor(0x2A, 0x78, 0xD6)      # validated pair: correct
RED = RGBColor(0xE3, 0x49, 0x48)       #                 wrong accepted
VIOLET = RGBColor(0x62, 0x50, 0xD6)    # validated pair: recall
ORANGE = RGBColor(0xEB, 0x68, 0x34)    #                 false rejection
NEUTRAL = RGBColor(0x8A, 0x89, 0x85)


ORDER = ['lnSpc','spcBef','spcAft','buClrTx','buClr','buSzTx','buSzPct','buSzPts','buFontTx','buFont',
         'buNone','buAutoNum','buChar','buBlip','tabLst','defRPr','extLst']
def put(ppr, tag, **attrs):
    """Insert/replace child `tag` of a:pPr at its schema position."""
    for e in ppr.findall(qn('a:'+tag)): ppr.remove(e)
    el = etree.Element(qn('a:'+tag))
    for k, v in attrs.items(): el.set(k, v)
    idx = ORDER.index(tag); pos = len(ppr)
    for i, ch in enumerate(ppr):
        name = etree.QName(ch).localname
        if name in ORDER and ORDER.index(name) > idx: pos = i; break
    ppr.insert(pos, el); return el

shapes = {sh.shape_id: sh for sh in s.shapes}

def place(sh, x, y, w, h):
    sh.left, sh.top, sh.width, sh.height = Inches(x), Inches(y), Inches(w), Inches(h)

def set_text(sh, paras, size=None, align=None, color=None):
    """paras: list of paragraphs; each is a str or a list of (text, bold) runs; dict {'bullet':True,'runs':...}.
    Keeps the template's first-run formatting (font, color) and first paragraph's properties."""
    tf = sh.text_frame
    p0 = tf.paragraphs[0]._p
    ppr0 = p0.find(qn('a:pPr'))
    r0 = p0.find(qn('a:r'))
    rpr0 = r0.find(qn('a:rPr')) if r0 is not None else None
    # find a bulleted pPr in the template, if any
    bullet_ppr = None
    for p in tf.paragraphs:
        pp = p._p.find(qn('a:pPr'))
        if pp is not None and pp.find(qn('a:buChar')) is not None:
            bullet_ppr = pp; break
    txBody = tf._txBody
    for p in list(txBody.findall(qn('a:p'))):
        txBody.remove(p)
    for para in paras:
        bullet = isinstance(para, dict) and para.get('bullet')
        runs = para['runs'] if isinstance(para, dict) else para
        if isinstance(runs, str):
            runs = [(runs, False)]
        p = etree.SubElement(txBody, qn('a:p'))
        src_ppr = bullet_ppr if (bullet and bullet_ppr is not None) else ppr0
        if src_ppr is not None:
            ppr = copy.deepcopy(src_ppr)
            if not bullet:
                for tag in ('a:buChar', 'a:buAutoNum', 'a:buFont', 'a:buSzPts', 'a:buClr'):
                    for e in ppr.findall(qn(tag)): ppr.remove(e)
                ppr.set('indent', '0'); ppr.set('marL', '0')
                put(ppr, 'buNone')
            if align: ppr.set('algn', align)
            p.append(ppr)
        for text, bold in runs:
            r = etree.SubElement(p, qn('a:r'))
            rpr = copy.deepcopy(rpr0) if rpr0 is not None else etree.Element(qn('a:rPr'))
            rpr.set('lang', 'en-US')
            if size: rpr.set('sz', str(int(size * 100)))
            rpr.set('b', '1' if bold else '0')
            if color is not None:
                for e in rpr.findall(qn('a:solidFill')): rpr.remove(e)
                sf = etree.Element(qn('a:solidFill')); c = etree.SubElement(sf, qn('a:srgbClr')); c.set('val', str(color))
                rpr.insert(0, sf)
            r.append(rpr)
            t = etree.SubElement(r, qn('a:t')); t.text = text

def textbox(x, y, w, h, paras, size, color=INK, bold=False, align=PP_ALIGN.LEFT, name=None, anchor=MSO_ANCHOR.TOP):
    tb = s.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    if name: tb.name = name
    tf = tb.text_frame; tf.word_wrap = True; tf.vertical_anchor = anchor
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    for i, ptxt in enumerate(paras):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = align
        r = p.add_run(); r.text = ptxt
        r.font.size = Pt(size); r.font.bold = bold; r.font.color.rgb = color; r.font.name = 'Arial'
    return tb

def remove(sh):
    sh._element.getparent().remove(sh._element)

# ---------------------------------------------------------------- title
title = shapes[108]
place(title, 0.5, 0.25, 22.3, 2.5)
set_text(title, [[('Intent-Gated Data Transformation Agent', True)]])
for r in title.text_frame.paragraphs[0].runs: r.font.size = Pt(72)
title.text_frame.vertical_anchor = MSO_ANCHOR.MIDDLE

# ---------------------------------------------------------------- abstract
set_text(shapes[86], [
    [('Problem: ', True), ('Small language models can generate code for data-transformation tasks, but may misinterpret the request '
      'or unintentionally alter unrelated data. We explore how explicit intent contracts, verification and guided '
      'repair can improve their correctness and reliability.', False)],
    [('Idea: ', True), ('Turn each request into an ', False), ('intent contract', True),
     (' stating what must change and what must be preserved, verify the data before and after against it, '
      'and let a bounded tool-using agent repair failing code.', False)],
    [('Research question: ', True), ('Can intent contracts detect silent errors in LLM-generated transformations, '
      'and does an agent built on them deliver more correct transformations?', False)],
], size=24, align='l')

# ---------------------------------------------------------------- methodology
from pptx.enum.dml import MSO_PATTERN, MSO_LINE_DASH_STYLE
place(shapes[91], 0.8, 11.85, 8.8, 3.85)
set_text(shapes[91], [
    {'bullet': True, 'runs': [('Intent parser: ', True), ('a QLoRA fine-tuned Qwen2.5-1.5B turns the request and data schema '
                                                          'into an intent contract (selective-facts or tool-use variant).', False)]},
    {'bullet': True, 'runs': [('Code generator: ', True), ('Qwen2.5-3B writes a pandas transform(df); an AST safety checker '
                                                           'blocks unsafe code and a sandbox runs the rest with time and memory limits.', False)]},
    {'bullet': True, 'runs': [('Verifier: ', True), ('deterministic; checks every contract predicate on the data before and after '
                                                     'and returns PASS, VIOLATION or UNCERTAIN.', False)]},
    {'bullet': True, 'runs': [('Repair agent: ', True), ('Qwen2.5-3B gets the failure report and read-only data tools; '
                                                         'at most 3 attempts, then the task is rejected or escalated.', False)]},
    {'bullet': True, 'runs': [('Benchmark: ', True), ('120 tasks (4 public datasets × 5 transformation types) and '
                                                      '600 LLM-written programs labelled correct or wrong.', False)]},
], size=18, align='l')
for p in shapes[91].text_frame.paragraphs:
    pPr = p._p.get_or_add_pPr()
    pPr.set('marL', str(Inches(0.35))); pPr.set('indent', str(-Inches(0.35)))
    for e in pPr.findall(qn('a:buNone')): pPr.remove(e)
    if pPr.find(qn('a:buChar')) is None:
        put(pPr, 'buFont', typeface='Arial'); put(pPr, 'buChar', char='•')
    sp = put(pPr, 'spcAft')
    etree.SubElement(sp, qn('a:spcPts')).set('val', '400')
remove(shapes[98])  # "Image if any" -> system-overview diagram

LAV = RGBColor(0xE6, 0xE1, 0xF4); TERM = RGBColor(0xE2, 0xE2, 0xE2); ACC = RGBColor(0xDC, 0xEB, 0xFB)

def node(x, y, w, h, text, fill=WHITE, line=PURPLE, fg=INK, size=14, name=None):
    b = s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(x), Inches(y), Inches(w), Inches(h))
    b.name = name or text.split('\n')[0]
    b.adjustments[0] = 0.12
    b.fill.solid(); b.fill.fore_color.rgb = fill
    b.line.color.rgb = line; b.line.width = Pt(1.75)
    b.shadow.inherit = False
    tf = b.text_frame; tf.word_wrap = True; tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    tf.margin_left = tf.margin_right = Inches(0.05); tf.margin_top = tf.margin_bottom = Inches(0.02)
    for i, ln in enumerate(text.split('\n')):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = PP_ALIGN.CENTER
        r = p.add_run(); r.text = ln
        r.font.size = Pt(size if i == 0 else size - 3); r.font.bold = (i == 0)
        r.font.color.rgb = fg; r.font.name = 'Arial'
    return b

def seg(pts, dashed=False, head=True, color=GREY, name='Flow arrow'):
    """Polyline of straight connectors; arrowhead on the last segment."""
    for k in range(len(pts) - 1):
        (x1, y1), (x2, y2) = pts[k], pts[k + 1]
        c = s.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, Inches(x1), Inches(y1), Inches(x2), Inches(y2))
        c.name = name
        c.line.color.rgb = color; c.line.width = Pt(2)
        if dashed: c.line.dash_style = MSO_LINE_DASH_STYLE.DASH
        if head and k == len(pts) - 2:
            ln = c.line._get_or_add_ln()
            t = etree.SubElement(ln, qn('a:tailEnd')); t.set('type', 'triangle'); t.set('w', 'med'); t.set('len', 'med')

def elabel(x, y, w, text, align=PP_ALIGN.CENTER):
    textbox(x, y, w, 0.28, [text], 12, color=GREY, align=align, name='Edge label ' + text)

W, H, HT = 1.35, 0.85, 0.5                  # node width / height / terminal height
GAPS = [0.4, 0.75, 0.45, 0.45]               # wider gaps where edges carry labels
CX = [0.75]
for g in GAPS: CX.append(CX[-1] + W + g)
R = {0: 15.85, 1: 16.65, 2: 18.25, 3: 19.85, 4: 21.25}
cx = lambda c: CX[c] + W / 2
node(CX[2], R[0], W, HT, 'ESCALATE', fill=TERM, line=NEUTRAL)
node(CX[0], R[1], W, H, 'User request\n+ data schema')
node(CX[1], R[1], W, H, 'Intent parser\nprompted,\nfine-tuned,\noracle or none')
node(CX[2], R[1], W, H, 'Contract\nvalid?', fill=LAV)
node(CX[3], R[1], W, H, 'Intent contract\n(read-only)')
node(CX[0], R[2], W, H, 'Code\ngenerator')
node(CX[1], R[2], W, H, 'AST safety\nchecker', fill=LAV)
node(CX[2], R[2], W, H, 'Sandbox')
node(CX[3], R[2], W, H, 'Verifier\nnone or\nintent', fill=LAV)
node(CX[4], R[1] + 0.35, W, HT, 'ACCEPT', fill=ACC, line=BLUE, fg=BLUE)
node(CX[4], R[3] - 0.35, W, HT, 'ESCALATE', fill=TERM, line=NEUTRAL)
node(CX[1], R[3], W, H, 'Repair agent\n+ tools')
node(CX[2], R[3], W, H, 'Failure\nreport')
node(CX[1], R[4], W, HT, 'ESCALATE', fill=TERM, line=NEUTRAL)
node(CX[2], R[4], W, HT, 'REJECT', fill=TERM, line=NEUTRAL)
mid1 = R[1] + H / 2; mid2 = R[2] + H / 2; mid3 = R[3] + H / 2
# row 1
seg([(CX[0] + W, mid1), (CX[1], mid1)])
seg([(CX[1] + W, mid1), (CX[2], mid1)])
seg([(cx(2), R[1]), (cx(2), R[0] + HT)]); elabel(cx(2) + 0.05, R[0] + HT + 0.08, 0.5, 'no', PP_ALIGN.LEFT)
seg([(CX[2] + W, mid1), (CX[3], mid1)]); elabel(CX[2] + W, mid1 - 0.32, GAPS[2], 'yes')
# contract -> verifier, contract -.-> generator
seg([(cx(3) + 0.3, R[1] + H), (cx(3) + 0.3, R[2])])
yd = (R[1] + H + R[2]) / 2
seg([(CX[3] + 0.25, R[1] + H), (CX[3] + 0.25, yd), (CX[0] + 1.15, yd), (CX[0] + 1.15, R[2])], dashed=True)
elabel(CX[1] + 0.2, yd - 0.3, 3.4, 'contract given only in contract-aware systems')
seg([(CX[0] + 0.4, R[1] + H), (CX[0] + 0.4, R[2])])
# row 2
seg([(CX[0] + W, mid2), (CX[1], mid2)])
seg([(CX[1] + W, mid2), (CX[2], mid2)]); elabel(CX[1] + W, mid2 - 0.32, GAPS[1], 'allowed')
seg([(CX[2] + W, mid2), (CX[3], mid2)])
seg([(CX[3] + W, R[2] + 0.2), (CX[4] + 0.3, R[1] + 0.35 + HT)]); pmx = (CX[3] + W + CX[4] + 0.3) / 2; pmy = (R[2] + 0.2 + R[1] + 0.35 + HT) / 2
elabel(pmx + 0.08, pmy - 0.1, 0.6, 'PASS', PP_ALIGN.LEFT)
seg([(CX[3] + W, R[2] + H - 0.2), (CX[4] + 0.3, R[3] - 0.35)]); elabel(CX[4] - 0.1, R[2] + H + 0.12, 1.4, 'UNCERTAIN')
# failures
seg([(cx(3) + 0.2, R[2] + H), (cx(3) + 0.2, mid3), (CX[2] + W, mid3)]); elabel(cx(3) + 0.15 - 1.3, R[2] + H + 0.4, 1.3, 'VIOLATION', PP_ALIGN.RIGHT)
seg([(CX[1] + W - 0.15, R[2] + H), (CX[2] + 0.25, R[3])]); elabel(CX[1] + W + 0.15, R[2] + H + 0.08, 0.8, 'blocked', PP_ALIGN.LEFT)
seg([(cx(2) + 0.3, R[2] + H), (cx(2) + 0.3, R[3])]); elabel(cx(2) + 0.33, R[2] + H + 0.06, 1.3, 'crash / timeout', PP_ALIGN.LEFT)
seg([(CX[2], mid3), (CX[1] + W, mid3)]); elabel(CX[1] + W, mid3 - 0.32, GAPS[1], 'repair on')
seg([(CX[1] + 0.4, R[3]), (CX[1] + 0.4, R[2] + H)]); elabel(CX[0] + 0.1, R[2] + H + 0.3, CX[1] + 0.35 - CX[0] - 0.1, 'revised code', PP_ALIGN.RIGHT)
seg([(cx(1), R[3] + H), (cx(1), R[4])]); elabel(cx(1) + 0.05, R[3] + H + 0.1, 0.8, 'escalate', PP_ALIGN.LEFT)
seg([(cx(2), R[3] + H), (cx(2), R[4])]); elabel(cx(2) + 0.05, R[3] + H + 0.1, 2.3, 'repair off / budget used', PP_ALIGN.LEFT)

# ---------------------------------------------------------------- region boxes: two on top, one wide below
place(shapes[95], 10.0, 3.2, 7.4, 8.9)     # top left: intent contract (box drawn before its heading/text 96, 97)
place(shapes[109], 17.6, 3.2, 7.4, 8.9)    # top right: silent-error detection (before 110, 111)
place(shapes[83], 10.0, 12.2, 15.0, 9.8)   # bottom: end-to-end accuracy

def style_chart(chart, title):
    chart.has_title = True
    chart.chart_title.text_frame.text = title
    for p in chart.chart_title.text_frame.paragraphs:
        for r in p.runs:
            r.font.size = Pt(20); r.font.bold = True; r.font.color.rgb = INK; r.font.name = 'Arial'
    chart.has_legend = True
    chart.legend.position = XL_LEGEND_POSITION.TOP
    chart.legend.include_in_layout = False
    chart.legend.font.size = Pt(18); chart.legend.font.color.rgb = GREY; chart.legend.font.name = 'Arial'
    ca, va = chart.category_axis, chart.value_axis
    ca.tick_labels.font.size = Pt(18); ca.tick_labels.font.color.rgb = INK; ca.tick_labels.font.name = 'Arial'
    ca.format.line.color.rgb = RGBColor(0xBF, 0xBF, 0xBF)
    ca.has_major_gridlines = False
    va.visible = False; va.has_major_gridlines = False; va.minimum_scale = 0

def style_series(plot, colors, numfmt):
    plot.gap_width = 45; plot.overlap = -8
    for ser, col in zip(plot.series, colors):
        ser.format.fill.solid(); ser.format.fill.fore_color.rgb = col
        ser.format.line.color.rgb = WHITE; ser.format.line.width = Pt(1.5)
        dl = ser.data_labels
        dl.show_value = True; dl.number_format = numfmt; dl.number_format_is_linked = False
        dl.position = XL_LABEL_POSITION.OUTSIDE_END
        dl.font.size = Pt(18); dl.font.bold = True; dl.font.color.rgb = INK; dl.font.name = 'Arial'

C_GRAY = RGBColor(0x7A, 0x7A, 0x7A); C_VIOLET = RGBColor(0x8E, 0x6C, 0xC4); C_PROP = PURPLE
C_GOLD = RGBColor(0x2A, 0x78, 0xD6); C_WRONG = RGBColor(0xE8, 0x89, 0x2B); C_REJ = RGBColor(0xB8, 0x3B, 0x5E)
C_ESC = RGBColor(0xC9, 0xC9, 0xC9); GOOD = RGBColor(0x2E, 0x7D, 0x32)

def rect(x, y, w, h, fill, name, line=None, shape=MSO_SHAPE.RECTANGLE):
    r = s.shapes.add_shape(shape, Inches(x), Inches(y), Inches(w), Inches(h))
    r.name = name; r.shadow.inherit = False
    if fill is None: r.fill.background()
    else: r.fill.solid(); r.fill.fore_color.rgb = fill
    if line is None: r.line.fill.background()
    else: r.line.color.rgb = line; r.line.width = Pt(1.5)
    return r

def label(x, y, w, h, text, size, color, bold=True, align=PP_ALIGN.CENTER, italic=False, name=None):
    tb = textbox(x, y, w, h, [text], size, color=color, bold=bold, align=align, name=name, anchor=MSO_ANCHOR.MIDDLE)
    for p in tb.text_frame.paragraphs:
        for r in p.runs: r.font.italic = italic
    return tb

# ---------------------------------------------------------------- top left: intent contract
IX, IY = 10.4, 3.5
place(shapes[96], IX, IY, 6.9, 0.6)
set_text(shapes[96], [[('INTENT CONTRACT', True)]])
place(shapes[97], IX, IY + 0.7, 6.7, 2.9)
set_text(shapes[97], [
    [('A contract lists checkable predicates in two halves: ', False), ('must_change', True),
     (' (what the request asks for) and ', False), ('must_preserve', True), (' (what should stay the same).', False)],
    [('Both halves are needed:', False)],
    [('Recall with gold contracts: ', True), ('full ', False), ('0.94', True), ('  ·  change\u2011only ', False), ('0.73', True),
     ('  ·  preserve\u2011only ', False), ('0.45', True)]], size=20, align='l')
_rl = shapes[97].text_frame.paragraphs[2]
for _r in _rl.runs[1:]:
    if _r.font.bold: _r.font.color.rgb = PURPLE
_rl.space_before = Pt(6)
remove(shapes[105])

def card(x, y, w, h, fill, border, title, title_col, lines, name):
    b = rect(x, y, w, h, fill, name, line=border, shape=MSO_SHAPE.ROUNDED_RECTANGLE)
    b.adjustments[0] = 0.08
    tb = s.shapes.add_textbox(Inches(x + 0.15), Inches(y + 0.08), Inches(w - 0.3), Inches(h - 0.16))
    tb.name = name + ' text'
    tf = tb.text_frame; tf.word_wrap = True
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    r = tf.paragraphs[0].add_run(); r.text = title
    r.font.size = Pt(20); r.font.bold = True; r.font.color.rgb = title_col; r.font.name = 'Arial'
    for parts in lines:
        p = tf.add_paragraph(); p.space_before = Pt(6)
        for txt, mono in parts:
            r = p.add_run(); r.text = txt; r.font.size = Pt(18); r.font.color.rgb = INK
            r.font.name = 'Courier New' if mono else 'Arial'; r.font.bold = mono

CY = IY + 3.05
req = rect(IX, CY, 6.6, 1.2, WHITE, 'Contract example request', line=RGBColor(0xBF, 0xBF, 0xBF),
           shape=MSO_SHAPE.ROUNDED_RECTANGLE)
req.adjustments[0] = 0.08
tb = s.shapes.add_textbox(Inches(IX + 0.15), Inches(CY + 0.07), Inches(6.3), Inches(1.0)); tb.name = 'Contract example request text'
tf = tb.text_frame; tf.word_wrap = True; tf.vertical_anchor = MSO_ANCHOR.MIDDLE
tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
p = tf.paragraphs[0]
r = p.add_run(); r.text = 'Request: '; r.font.bold = True; r.font.size = Pt(18); r.font.color.rgb = INK; r.font.name = 'Arial'
r = p.add_run(); r.text = '"Remove passengers younger than 18. Passengers whose age is unknown should stay."'
r.font.italic = True; r.font.size = Pt(18); r.font.color.rgb = INK; r.font.name = 'Arial'
card(IX, CY + 1.4, 6.6, 1.6, RGBColor(0xE3, 0xF1, 0xE6), RGBColor(0x3C, 0x8D, 0x5A), 'must_change',
     RGBColor(0x1F, 0x6B, 0x3A), [[('• rows_removed_where ', True), ('Age < 18', False)],
                                  [('• row_count ', True), ('decreased', False)]], 'must_change card')
card(IX, CY + 3.2, 6.6, 1.6, RGBColor(0xEC, 0xE6, 0xF6), RGBColor(0x62, 0x50, 0xD6), 'must_preserve',
     PURPLE, [[('• rows_retained_where ', True), ('Age >= 18 or Age missing', False)],
              [('• columns_unchanged ', True), ('all columns', False)]], 'must_preserve card')
label(IX, CY + 4.85, 6.6, 0.32, 'Gold contract of benchmark task titanic_filt_003', 14, GREY, bold=False, italic=True)

# ---------------------------------------------------------------- top right: silent-error detection
DX, DY = 18.0, 3.5
place(shapes[110], DX, DY, 6.9, 0.6)
set_text(shapes[110], [[('DETECTING SILENT ERRORS', True)]])
place(shapes[111], DX, DY + 0.7, 6.7, 2.6)
set_text(shapes[111], [[('The verifier judges 600 LLM-written test programs. ', False),
    ('With intent contracts it catches 89–94% of the wrong ones', True),
    (', and the ', False), ('silent error rate', True),
    (' (share of accepted programs that are wrong) falls from 34% with no checks to 3–4%.', False)]], size=20, align='l')
remove(shapes[112])
cd1 = CategoryChartData()
# tool fine-tuned row uses the tool-use parser's own report numbers (Exp 6b: recall 0.930, FRR 0.139, silent 0.040)
cd1.categories = ['Gold contracts', 'Selective-facts 1.5B', 'Tool fine-tuned 1.5B', 'Prompted 7B', 'No checks']
cd1.add_series('Silent error rate', (0.028, 0.032, 0.040, 0.069, 0.338))
cd1.add_series('False rejection', (0.000, 0.120, 0.139, 0.254, 0.000))
cd1.add_series('Recall', (0.944, 0.944, 0.930, 0.892, 0.000))
g1 = s.shapes.add_chart(XL_CHART_TYPE.BAR_CLUSTERED, Inches(DX - 0.15), Inches(DY + 2.95), Inches(7.0), Inches(5.6), cd1)
g1.name = 'Silent-error detection chart'
c1 = g1.chart
style_chart(c1, 'Detection on the frozen program pool')
style_series(c1.plots[0], [RGBColor(0x1B, 0xAF, 0x7A), ORANGE, VIOLET], '0%')
c1.plots[0].gap_width = 120; c1.plots[0].overlap = 0
c1.value_axis.maximum_scale = 1.25
c1.category_axis.tick_labels.font.size = Pt(16)
c1.legend.font.size = Pt(16)
for ser in c1.plots[0].series: ser.data_labels.font.size = Pt(14)

# ---------------------------------------------------------------- bottom: end-to-end accuracy
EX, EY = 10.4, 12.5
place(shapes[89], EX, EY, 14.3, 0.6)
set_text(shapes[89], [[('END-TO-END ACCURACY', True)]])
place(shapes[88], EX, EY + 0.7, 14.3, 0.9)
set_text(shapes[88], [[('Full pipeline on 100 test tasks. ', True),
    ('The proposed system accepts ', False), ('2 wrong outputs instead of 20', True),
    (' and gives 82 correct outputs vs 78.', False)]], size=20, align='l')
remove(shapes[94])

BX0, BW = 15.15, 9.4
sx = lambda v: BX0 + BW * v / 100
RY0, RH, BH = EY + 2.2, 1.0, 0.66
rows = [  # title, subtitle, correct, wrong accepted, rejected, escalated, correct colour, title colour, hatched
    ('Baseline 1: LLM generation', 'direct generation, no checks', 78, 20, 2, 0, C_GRAY, INK, False),
    ('Baseline 2', '+ repair, contracts from a prompted 7B parser', 67, 3, 11, 19, C_VIOLET, INK, False),
    ('Fine-tuned parser', '+ fine-tuned 1.5B parser + contract self-check', 79, 2, 6, 13, C_VIOLET, INK, False),
    ('Proposed', '+ contract also given to the code generator', 82, 2, 5, 11, C_PROP, PURPLE, False),
    ('Proposed (tool use)', 'same, with the tool-use parser', 80, 1, 10, 9, C_VIOLET, INK, False),
    ('Gold contract', 'upper bound: correct contracts', 92, 0, 8, 0, C_GOLD, C_GOLD, False),
]
label(BX0, RY0 - 0.6, 5.0, 0.45, 'Outcome of each test task', 18, INK, align=PP_ALIGN.LEFT, name='Results column title')
for i, (t, sub, cor, wr, rej, esc, ccol, tcol, hatch) in enumerate(rows):
    y = RY0 + i * RH; by = y + (RH - BH) / 2
    tb = textbox(EX - 0.05, y + 0.03, 4.7, RH - 0.06, [t, sub], 18, color=tcol, bold=True, name=f'Row label {i+1}',
                 anchor=MSO_ANCHOR.MIDDLE)
    for r in tb.text_frame.paragraphs[1].runs:
        r.font.size = Pt(14); r.font.bold = False; r.font.color.rgb = GREY
    x = BX0
    for k, (val, col) in enumerate([(cor, ccol), (wr, C_WRONG), (rej, C_REJ), (esc, C_ESC)]):
        if val == 0: continue
        w = BW * val / 100
        sg = rect(x, by, w, BH, col, f'Row {i+1} segment {k+1}', line=WHITE)
        if k == 0 and hatch:
            sg.fill.patterned(); sg.fill.pattern = MSO_PATTERN.WIDE_UPWARD_DIAGONAL
            sg.fill.fore_color.rgb = C_VIOLET; sg.fill.back_color.rgb = RGBColor(0xE4, 0xDC, 0xF2)
            rect(x + w / 2 - 0.32, by + 0.08, 0.64, BH - 0.16, PURPLE, f'Row {i+1} label chip')
            label(x + w / 2 - 0.32, by + 0.08, 0.64, BH - 0.16, str(val), 17, WHITE)
        elif w >= 0.3:
            label(x, by, w, BH, str(val), 17, INK if col in (C_WRONG, C_ESC) else WHITE)
        else:
            label(x + w / 2 - 0.25, by - 0.3, 0.5, 0.3, str(val), 14, C_REJ if k == 2 else RGBColor(0xA8, 0x5A, 0x10))
        x += w
ref_x = sx(78)
ln = s.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, Inches(ref_x), Inches(RY0 - 0.12), Inches(ref_x), Inches(RY0 + len(rows) * RH + 0.05))
ln.name = 'Baseline 1 reference line'; ln.line.color.rgb = INK; ln.line.width = Pt(2); ln.line.dash_style = MSO_LINE_DASH_STYLE.DASH
label(ref_x - 3.6, RY0 - 0.58, 3.5, 0.4, 'Baseline 1 = 78 correct', 14, INK, bold=False, align=PP_ALIGN.RIGHT)
ay = RY0 + len(rows) * RH + 0.05
ax = s.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, Inches(BX0), Inches(ay), Inches(BX0 + BW), Inches(ay))
ax.name = 'Results axis'; ax.line.color.rgb = GREY; ax.line.width = Pt(1.25)
for v in range(0, 101, 20):
    label(sx(v) - 0.3, ay + 0.02, 0.6, 0.3, str(v), 13, GREY, bold=False)
lx, ly = EX, ay + 0.5
for txt, col, hatch in [('Correct', C_VIOLET, False), ('Wrong but accepted', C_WRONG, False), ('Rejected', C_REJ, False),
                        ('Escalated to a human', C_ESC, False)]:
    sw = rect(lx, ly + 0.08, 0.32, 0.24, col, f'Legend {txt}')
    if hatch:
        sw.fill.patterned(); sw.fill.pattern = MSO_PATTERN.WIDE_UPWARD_DIAGONAL
        sw.fill.fore_color.rgb = C_VIOLET; sw.fill.back_color.rgb = RGBColor(0xE4, 0xDC, 0xF2)
    wtxt = 0.115 * len(txt) + 0.1
    label(lx + 0.4, ly, wtxt, 0.4, txt, 14, INK, bold=False, align=PP_ALIGN.LEFT)
    lx += 0.4 + wtxt + 0.5


# ---------------------------------------------------------------- right column: conclusion, ack, refs
place(shapes[102], 25.5, 7.15, 6.9, 5.8)
set_text(shapes[102], [
    {'bullet': True, 'runs': [('Intent contracts catch ', False), ('93–94%', True), (' of silent errors with fine-tuned parsers; the silent error rate falls from 34% to 3–4%.', False)]},
    {'bullet': True, 'runs': [('Wrong outputs accepted fall from ', False), ('20 to 2', True), ('; correct outputs rise from 78 to 82.', False)]},
    {'bullet': True, 'runs': [('Contract quality is the bottleneck: ', True), ('with gold contracts the same pipeline solves 92/100.', False)]},
    {'bullet': True, 'runs': [('Fine-tuning beats size: ', True), ('a QLoRA 1.5B parser writes better contracts than a prompted 7B model.', False)]},
    {'bullet': True, 'runs': [('Future work: ', True), ('generalize intent contracts to all data transformations: joins and '
                                                        'merges, group-by aggregations, reshaping (pivot / melt) and multi-step pipelines.', False)]},
    {'bullet': True, 'runs': [('The same before/after checks apply to any engine, so contracts can verify SQL and Spark code, '
                               'not only pandas.', False)]},
], size=20, align='l')

place(shapes[100], 25.5, 13.1, 7.0, 0.6)
place(shapes[92], 25.5, 13.75, 7.0, 1.2)
set_text(shapes[92], [[('We thank our TA and the LMA course instructors for their feedback, and Kaggle for free '
                        'T4 GPU time.', False)]], size=20, align='l')

place(shapes[101], 25.5, 15.35, 7.0, 0.6)
place(shapes[93], 25.5, 16.0, 6.95, 5.8)
refs = ['[1] Dettmers et al. QLoRA: efficient finetuning of quantized LLMs. NeurIPS 2023.',
        '[2] Lai et al. DS-1000: a benchmark for data science code generation. ICML 2023.',
        '[3] Endres et al. Can LLMs transform natural language intent into formal postconditions? FSE 2024.',
        '[4] Chen et al. Teaching LLMs to self-debug. ICLR 2024.',
        '[5] Yao et al. ReAct: synergizing reasoning and acting in LMs. ICLR 2023.',
        '[6] Qwen Team. Qwen2.5 technical report. arXiv:2412.15115, 2024.']
set_text(shapes[93], [[(r, False)] for r in refs] + [[('Code: ', True), ('github.com/srinivas0307/lma-group-project', False)]],
         size=17, align='l')

# ---------------------------------------------------------------- authors
place(shapes[106], 0.7, 22.3, 21.0, 0.6)
set_text(shapes[106], [[('Authors: ', True), ('Akash Sigullapalli, Konapalli Jaswanth Reddy, Kotha Sai Veekshith, '
                                               'Srinivas Billuri', False)]], size=26)

# ---------------------------------------------------------------- team box (was CSTAR)
place(shapes[103], 25.4, 3.3, 7.2, 1.3)
set_text(shapes[103], [[('Team SSSS', True)]], size=60, align='ctr', color='47247B')
place(shapes[104], 25.4, 4.55, 7.2, 0.6)
set_text(shapes[104], [[('LMA Group Project 2026', False)]], size=26, align='ctr')

prs.save(OUT)
print('saved', OUT)
