import json,re,html,subprocess,textwrap,math
from pathlib import Path
from reportlab.pdfgen import canvas
from reportlab.platypus import *
from reportlab.platypus.tableofcontents import TableOfContents
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER,TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
ROOT=Path.cwd(); TMP=ROOT/'tmp/pdfs'; OUT=ROOT/'output/pdf/mid_submission_report.pdf'
for n,f in [('Serif','DejaVuSerif.ttf'),('SerifB','DejaVuSerif-Bold.ttf'),('SerifI','DejaVuSerif-Italic.ttf'),('Sans','DejaVuSans.ttf'),('SansB','DejaVuSans-Bold.ttf'),('Mono','DejaVuSansMono.ttf')]:
 p=Path('/usr/share/fonts/truetype/dejavu')/f
 pdfmetrics.registerFont(TTFont(n,str(p)))
pdfmetrics.registerFontFamily('Serif',normal='Serif',bold='SerifB',italic='SerifI',boldItalic='SerifB')
pdfmetrics.registerFontFamily('Sans',normal='Sans',bold='SansB',italic='Sans',boldItalic='SansB')
NAVY=colors.HexColor('#19364b'); TEAL=colors.HexColor('#216c7c'); GRAY=colors.HexColor('#566574')
W,H=A4; CW=W-100
styles={
 'body':ParagraphStyle('body',fontName='Serif',fontSize=9.6,leading=14,spaceAfter=7,alignment=TA_JUSTIFY,splitLongWords=True),
 'h1':ParagraphStyle('h1',fontName='SansB',fontSize=17,leading=22,textColor=NAVY,spaceBefore=10,spaceAfter=14,keepWithNext=True),
 'h2':ParagraphStyle('h2',fontName='SansB',fontSize=12,leading=16,textColor=NAVY,spaceBefore=13,spaceAfter=8,keepWithNext=True),
 'h3':ParagraphStyle('h3',fontName='SansB',fontSize=10,leading=14,textColor=TEAL,spaceBefore=10,spaceAfter=7,keepWithNext=True),
 'cell':ParagraphStyle('cell',fontName='Sans',fontSize=7.4,leading=10,splitLongWords=True),
 'th':ParagraphStyle('th',fontName='SansB',fontSize=7.4,leading=10,textColor=colors.white,splitLongWords=True),
 'caption':ParagraphStyle('caption',fontName='Sans',fontSize=8,leading=11,textColor=GRAY,spaceBefore=7,spaceAfter=12,alignment=TA_CENTER),
 'code':ParagraphStyle('code',fontName='Mono',fontSize=6.9,leading=9.5,leftIndent=9,rightIndent=9,spaceBefore=6,spaceAfter=10,backColor=colors.HexColor('#f1f4f6'),borderPadding=8),
}
def norm(s): return s.replace('\u2011','-').replace('\u2013','-').replace('\u2014','-')
def inline(xs):
 out=''
 for x in xs:
  t=x['t'];c=x.get('c')
  if t=='Str':out+=html.escape(norm(c))
  elif t in ['Space','SoftBreak']:out+=' '
  elif t=='LineBreak':out+='<br/>'
  elif t=='Strong':out+='<b>'+inline(c)+'</b>'
  elif t=='Emph':out+='<i>'+inline(c)+'</i>'
  elif t=='Code':out+='<font name="Mono" size="7.5">'+html.escape(norm(c[1]))+'</font>'
  elif t=='Link':
   label=inline(c[1]);url=c[2][0]
   out+=label if url.startswith('#') else '<link href="'+html.escape(url,quote=True)+'" color="#216c7c">'+label+'</link>'
  elif t=='Quoted':out+='“'+inline(c[1])+'”'
  elif t=='Math':out+=html.escape(c[1])
  else:
   if isinstance(c,str):out+=html.escape(c)
 return out

def node_text(s):
 s=re.sub(r'<br\s*/?>','\n',s);s=html.unescape(s).strip()
 lines=[]
 for part in s.splitlines():lines.extend(textwrap.wrap(part.strip(),31,break_long_words=False) or [''])
 return '\\n'.join(lines)
class Graph(Flowable):
 def __init__(self,src,idx):
  Flowable.__init__(self);nodes={};edges=[]
  tok=r'([A-Za-z][A-Za-z0-9_]*)(\[\([^\n]*?\)\]|\[\([^\n]*?\)\]|\[\([^\n]*?\)\]|\[\([^\n]*?\)\]|\[\([^\n]*?\)\]|\[\([^\n]*?\)\]|\[[^\n]*?\]|\{[^\n]*?\})?'
  # Node labels are local to a line; bracket forms include cylinders and rounded outcomes.
  tok=r'([A-Za-z][A-Za-z0-9_]*)(\(\[.*?\]\)|\[\(.*?\)\]|\[.*?\]|\{.*?\})?'
  def parse(s):
   m=re.fullmatch(tok,s.strip())
   if not m:raise ValueError('Cannot parse '+s)
   ident,shape=m.groups()
   if shape:
    label=shape[1:-1];kind='box'
    if shape.startswith('(['):label=shape[2:-2]
    elif shape.startswith('{'):kind='diamond'
    elif label.startswith('(') and label.endswith(')'):
     label=label[1:-1];kind='cylinder' if label.startswith(('Before','After','Input','Intent contract','Frozen','Contract')) else 'box'
    nodes[ident]=(node_text(label),kind)
   return ident
  for line in src.splitlines()[1:]:
   line=line.strip()
   if not line:continue
   m=re.match(r'^(.*?)\s*(<-->|-->)\s*(?:\|([^|]*)\|)?\s*(.*?)$',line)
   if m:
    a,arrow,label,b=m.groups();edges.append((parse(a),parse(b),label or '',arrow=='<-->'))
   else:parse(line)
  dot=['digraph G {','graph [rankdir=TB, bgcolor="transparent", nodesep=0.28, ranksep=0.38, pad=0.10];','node [fontname="DejaVu Sans", fontsize=10, shape=box, style="rounded,filled", color="#7995a7", fillcolor="#edf3f7", margin="0.12,0.08"];','edge [fontname="DejaVu Sans", fontsize=8, color="#627d8c", arrowsize=0.65];']
  for ident,(label,kind) in nodes.items():
   color='#e5f3e9' if 'ACCEPT' in label else '#fcebea' if 'REJECT' in label else '#fff2d5' if 'ESCALATE' in label else '#edf3f7'
   dot.append(f'{ident} [label={json.dumps(label).replace(chr(92)+chr(92),chr(92))}, shape={kind}, fillcolor="{color}"];')
  for a,b,label,bi in edges:
   dot.append(f'{a} -> {b} [label={json.dumps(node_text(label)).replace(chr(92)+chr(92),chr(92))}'+(', dir=both' if bi else '')+'];')
  if idx==5:
   dot=[x.replace('ranksep=0.38','ranksep=0.12').replace('nodesep=0.28','nodesep=0.16').replace('fontsize=10','fontsize=12').replace('fontsize=8','fontsize=10') for x in dot]
   dot=[x.replace('contract given to generator\\nremoved in the no-contract\\nablation','contract input*') for x in dot]
   dot.extend(['{rank=same; D; U;}', '{rank=same; RA; TR; REJ;}', 'RA -> TR [constraint=false, style=invis];'])
  dot.append('}')
  path=TMP/f'figure_{idx}.dot';path.write_text('\n'.join(dot))
  self.g=json.loads(subprocess.check_output(['dot','-Tjson',str(path)]));bb=list(map(float,self.g['bb'].split(',')))
  self.bw,self.bh=bb[2],bb[3];self.scale=min(CW/self.bw,(650 if idx==5 else 510)/self.bh,1.05)
  self.width=self.bw*self.scale;self.height=self.bh*self.scale;self.hAlign='CENTER'
 def draw(self):
  c=self.canv;c.saveState();c.scale(self.scale,self.scale)
  for obj in [self.g]+self.g.get('objects',[])+self.g.get('edges',[]):
   for key in ['_draw_','_hdraw_','_tdraw_','_ldraw_','_hldraw_','_tldraw_']:
    fill=False;size=10
    for op in obj.get(key,[]):
     t=op['op']
     if t in ['c','C']:
      val=op['color'];col=colors.HexColor(val[:7]) if val.startswith('#') else colors.toColor(val)
      (c.setStrokeColor if t=='c' else c.setFillColor)(col)
     elif t=='F':size=op['size'];c.setFont('Sans',size)
     elif t=='T':
      c.setFillColor(NAVY);x,y=op['pt'];align=op['align'];txt=op['text'];
      if align=='c':c.drawCentredString(x,y,txt)
      elif align=='r':c.drawRightString(x,y,txt)
      else:c.drawString(x,y,txt)
     elif t in ['P','p','L','b','B']:
      pts=op['points'];p=c.beginPath();p.moveTo(*pts[0])
      if t in ['b','B']:
       for i in range(1,len(pts),3):p.curveTo(*(pts[i]+pts[i+1]+pts[i+2]))
      else:
       for pt in pts[1:]:p.lineTo(*pt)
       if t!='L':p.close()
      c.drawPath(p,stroke=1,fill=int(t in ['P','B']))
     elif t in ['e','E']:
      x,y,rx,ry=op['rect'];c.ellipse(x-rx,y-ry,x+rx,y+ry,stroke=1,fill=int(t=='E'))
  c.restoreState()

class Report(BaseDocTemplate):
 def __init__(self,path):
  super().__init__(str(path),pagesize=A4,leftMargin=50,rightMargin=50,topMargin=53,bottomMargin=48,title='Intent-Gated Data Transformation Agent: Mid-Submission Report',author='Team SSSS')
  self.addPageTemplates(PageTemplate('normal',frames=[Frame(50,48,CW,H-101,id='body',leftPadding=0,rightPadding=0,topPadding=0,bottomPadding=0)],onPage=self.decorate))
 def decorate(self,c,doc):
  if doc.page==1:return
  c.setStrokeColor(colors.HexColor('#c9d5de'));c.setLineWidth(.5);c.line(50,H-33,W-50,H-33)
  c.setFont('Sans',7);c.setFillColor(GRAY);c.drawString(50,H-26,'TEAM SSSS  /  INTENT-GATED DATA TRANSFORMATION AGENT');c.drawString(50,29,'LMA Group Project 2026 | Mid-Submission Report');c.drawRightString(W-50,29,str(doc.page))
 def afterFlowable(self,f):
  if getattr(f,'toclevel',None) is not None:
   text=f.getPlainText();key='section-'+str(f.sectionkey);self.canv.bookmarkPage(key);self.canv.addOutlineEntry(text,key,level=f.toclevel,closed=False);self.notify('TOCEntry',(f.toclevel,text,self.page,key))

def P(s,style='body'):return Paragraph(s,styles[style])
def celltxt(bs):return '<br/>'.join(inline(b['c']) for b in bs if b['t'] in ['Plain','Para'])
def table(block):
 c=block['c'];heads=c[3][1];rows=list(heads)
 for body in c[4]:rows+=body[2]+body[3]
 rows+=c[5][1]
 raw=[[celltxt(cell[4]) for cell in row[1]] for row in rows];n=len(raw[0]);weights=[]
 for i in range(n):
  vals=[len(re.sub('<[^>]+>','',r[i])) for r in raw];weights.append(min(75,max(12,(sum(vals)/len(vals))**.85*2.5)))
 widths=[CW*v/sum(weights) for v in weights]
 data=[[P(v,'th' if ri<len(heads) else 'cell') for v in row] for ri,row in enumerate(raw)]
 t=Table(data,colWidths=widths,repeatRows=len(heads),hAlign='LEFT',spaceBefore=5,spaceAfter=12)
 t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,len(heads)-1),NAVY),('VALIGN',(0,0),(-1,-1),'TOP'),('LEFTPADDING',(0,0),(-1,-1),6),('RIGHTPADDING',(0,0),(-1,-1),6),('TOPPADDING',(0,0),(-1,-1),6),('BOTTOMPADDING',(0,0),(-1,-1),6),('ROWBACKGROUNDS',(0,len(heads)),(-1,-1),[colors.white,colors.HexColor('#f0f4f7')]),('LINEBELOW',(0,-1),(-1,-1),.5,colors.HexColor('#b7c7d3'))]))
 return t
story=[]
story+=[Spacer(1,65),P('LMA GROUP PROJECT 2026','caption'),Spacer(1,20)]
title=ParagraphStyle('title',fontName='SansB',fontSize=28,leading=36,textColor=NAVY,alignment=TA_CENTER)
story+=[Paragraph('Intent-Gated Data<br/>Transformation Agent',title),Spacer(1,19),Paragraph('Mid-Submission Report',ParagraphStyle('sub',fontName='Sans',fontSize=17,leading=23,alignment=TA_CENTER,textColor=TEAL)),Spacer(1,44),P('<b>Team SSSS</b>','caption')]
for name in ['Akash Sigullapalli (2023101104)','Konapalli Jaswanth Reddy (2023101025)','Kotha Sai Veekshith (2023101078)','Srinivas Billuri (2023101029)']:
 story.append(Paragraph(name,ParagraphStyle('author',fontName='Sans',fontSize=10,leading=20,alignment=TA_CENTER)))
story += [Spacer(1,42),P('Project repository','caption'),P('<link href="https://github.com/srinivas0307/lma-group-project" color="#216c7c">github.com/srinivas0307/lma-group-project</link>','caption'),PageBreak(),P('Contents','h1')]
toc=TableOfContents();toc.levelStyles=[ParagraphStyle('toc0',fontName='SansB',fontSize=10,leading=15,textColor=NAVY,spaceBefore=8),ParagraphStyle('toc1',fontName='Sans',fontSize=8.5,leading=12,leftIndent=14,spaceBefore=3)];story+=[toc]
blocks=json.load(open(TMP/'report.json'))['blocks'];start=next(i for i,b in enumerate(blocks) if b['t']=='Header' and inline(b['c'][2]).startswith('0. '));fig=0;sec=0
captions=['Contract validation and deterministic verification workflow.','Baseline 1: direct generation, safety checks, and sandbox execution.','Baseline 2: conventional data-quality verification.','Baseline 3: prompted intent parsing and deterministic verification.','Proposed architecture with contract-aware generation and bounded repair.']
def emit(b):
 global fig,sec
 t=b['t'];c=b.get('c')
 if t=='Header':
  lvl=c[0];txt=inline(c[2]);
  if lvl==2 and txt.startswith('0. '):story.append(PageBreak())
  p=P(txt,'h1' if lvl==2 else 'h2' if lvl==3 else 'h3')
  if lvl==2:p.toclevel=0;p.sectionkey=sec;sec+=1
  story.append(p)
 elif t in ['Para','Plain']:story.append(P(inline(c)))
 elif t=='Table':story.append(table(b))
 elif t=='CodeBlock':
  if 'mermaid' in c[0][1]:
   fig+=1;g=Graph(c[1],fig)
   if fig==5:story.append(PageBreak())
   story.append(KeepTogether([g,P(f'Figure {fig}. {captions[fig-1]}'+(' *Contract input is removed in the no-contract ablation.' if fig==5 else ''),'caption')]))
   if fig==5:story.append(PageBreak())
  else:
   lines=[]
   for line in c[1].splitlines():lines.extend(textwrap.wrap(norm(line),108,expand_tabs=False,replace_whitespace=False,drop_whitespace=False,subsequent_indent='    ') or [''])
   story.append(Preformatted('\n'.join(lines),styles['code']))
 elif t in ['BulletList','OrderedList']:
  items=c if t=='BulletList' else c[1]
  for i,item in enumerate(items,1):
   first=True
   for sub in item:
    if sub['t'] in ['Para','Plain']:
     st=ParagraphStyle('list',parent=styles['body'],leftIndent=15,firstLineIndent=0,bulletIndent=1,spaceAfter=5)
     story.append(Paragraph(inline(sub['c']),st,bulletText=('•' if t=='BulletList' else str(i)+'.') if first else None));first=False
    else:emit(sub)
 elif t=='HorizontalRule':pass
for b in blocks[start:]:emit(b)
Report(OUT).multiBuild(story)
print(OUT)
