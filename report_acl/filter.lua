-- Pandoc filter: ACL-friendly tables, breakable inline code, no horizontal rules.
local function esc(s)
  local map = {['\\']='\\textbackslash{}', ['{']='\\{', ['}']='\\}', ['_']='\\_', ['&']='\\&',
               ['%']='\\%', ['$']='\\$', ['#']='\\#', ['^']='\\^{}', ['~']='\\~{}'}
  return (s:gsub('[\\{}_&%%$#^~]', map))
end

function Code(el)
  local t = esc(el.text)
  t = t:gsub('(\\_)', '%1\\allowbreak{}'):gsub('/', '/\\allowbreak{}'):gsub('%.', '.\\allowbreak{}')
  return pandoc.RawInline('latex', '\\texttt{' .. t .. '}')
end

function HorizontalRule() return {} end

local inl
inl = function(list)
  local o = {}
  for _, e in ipairs(list) do
    local t = e.t
    if t == 'Str' then o[#o+1] = esc(e.text)
    elseif t == 'Space' or t == 'SoftBreak' then o[#o+1] = ' '
    elseif t == 'LineBreak' then o[#o+1] = ' '
    elseif t == 'Strong' then o[#o+1] = '\\textbf{' .. inl(e.content) .. '}'
    elseif t == 'Emph' then o[#o+1] = '\\emph{' .. inl(e.content) .. '}'
    elseif t == 'Code' then o[#o+1] = Code(e).text
    elseif t == 'Quoted' then o[#o+1] = '``' .. inl(e.content) .. "''"
    elseif t == "RawInline" then o[#o+1] = e.text
    elseif e.content then o[#o+1] = inl(e.content)
    else o[#o+1] = esc(pandoc.utils.stringify(e)) end
  end
  return table.concat(o)
end

local function cell_tex(blocks)
  local o = {}
  for _, b in ipairs(blocks) do
    if b.content and (b.t == 'Plain' or b.t == 'Para') then o[#o+1] = inl(b.content)
    else o[#o+1] = esc(pandoc.utils.stringify(b)) end
  end
  return table.concat(o, ' ')
end

local function cell_len(blocks) return #pandoc.utils.stringify(blocks) end

function Table(tbl)
  local ncols = #tbl.aligns
  local maxlen, sumlen = {}, {}
  for i = 1, ncols do maxlen[i] = 0; sumlen[i] = 0 end
  local rows = {}
  local header = {}
  local has_header = false
  for i = 1, ncols do
    header[i] = tbl.headers[i] or {}
    if #header[i] > 0 then has_header = true end
  end
  local function scan(row)
    for i = 1, ncols do
      local l = cell_len(row[i] or {})
      if l > maxlen[i] then maxlen[i] = l end
    end
  end
  if has_header then scan(header) end
  for _, r in ipairs(tbl.rows) do scan(r); rows[#rows + 1] = r end
  -- column spec
  local spec, est, nX = {}, 0, 0
  for i = 1, ncols do
    if maxlen[i] > 30 then
      spec[i] = '>{\\raggedright\\arraybackslash}X'; nX = nX + 1; est = est + 30
    else
      spec[i] = 'l'; est = est + maxlen[i] + 2
    end
  end
  local wide = (est > 44) or ncols >= 5
  local font = '\\footnotesize'
  if ncols >= 8 or est > 120 then font = '\\scriptsize' end
  local width = wide and '\\textwidth' or '\\columnwidth'
  local env = wide and 'table*' or 'center'
  local out = {}
  out[#out + 1] = wide and '\\begin{table*}[!tp]' or '\\vspace{4pt}\\begin{center}'
  out[#out + 1] = (wide and '\\centering' or '') .. font .. '\\setlength{\\tabcolsep}{3pt}\\renewcommand{\\arraystretch}{1.1}'
  if nX > 0 then
    out[#out + 1] = '\\begin{tabularx}{' .. width .. '}{' .. table.concat(spec) .. '}'
  else
    out[#out + 1] = '\\begin{adjustbox}{max width=' .. width .. '}\\begin{tabular}{' .. table.concat(spec) .. '}'
  end
  out[#out + 1] = '\\toprule'
  if has_header then
    local cs = {}
    for i = 1, ncols do cs[i] = '\\textbf{' .. cell_tex(header[i]) .. '}' end
    out[#out + 1] = table.concat(cs, ' & ') .. ' \\\\'
    out[#out + 1] = '\\midrule'
  end
  for _, r in ipairs(rows) do
    local cs = {}
    for i = 1, ncols do cs[i] = cell_tex(r[i] or {}) end
    out[#out + 1] = table.concat(cs, ' & ') .. ' \\\\'
  end
  out[#out + 1] = '\\bottomrule'
  out[#out + 1] = (nX > 0) and '\\end{tabularx}' or '\\end{tabular}\\end{adjustbox}'
  out[#out + 1] = wide and '\\end{table*}' or '\\end{center}\\vspace{2pt}'
  return pandoc.RawBlock('latex', table.concat(out, '\n'))
end
