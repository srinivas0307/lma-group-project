import re, subprocess, pathlib, sys
here = pathlib.Path(__file__).parent
src_name = sys.argv[1] if len(sys.argv) > 1 else 'final_report.md'
out_name = pathlib.Path(src_name).stem + '.tex'
src = (here.parent / src_name).read_text()

# split title block / abstract / body
m = re.search(r'^## Abstract\n(.*?)^## Introduction\n', src, re.S | re.M)
abstract_md = m.group(1).strip()
body_md = src[m.end() - len('## Introduction\n'):]
# mermaid diagram -> placeholder (drawn in TikZ)
body_md = re.sub(r'```mermaid.*?```', 'PIPELINEFIGURE', body_md, flags=re.S)
body_md = body_md.replace('\n---\n', '\n')

def pandoc(md, extra=()):
    r = subprocess.run(['pandoc', '-f', 'markdown-raw_tex-tex_math_dollars-raw_html-smart-auto_identifiers',
                        '-t', 'latex', '--lua-filter', str(here / 'filter.lua'), *extra],
                       input=md, capture_output=True, text=True)
    if r.returncode: raise SystemExit(r.stderr)
    return r.stdout

body = pandoc(body_md, ['--shift-heading-level-by=-1'])
abstract = pandoc(abstract_md)
fig = (here / 'figure.tex').read_text()
body = re.sub(r'PIPELINEFIGURE', lambda _: '', body)  # figure is inserted separately below
# put the figure after the paragraph that introduces the system overview
body = body.replace('\\subsection{System overview}', '\\subsection{System overview}\n' + fig, 1) \
    if '\\subsection{System overview}' in body else body
tpl = (here / 'template.tex').read_text()
(here / out_name).write_text(tpl.replace('%%ABSTRACT%%', abstract).replace('%%BODY%%', body))
print('wrote', out_name)
