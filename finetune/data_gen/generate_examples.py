"""Step 3 of the fine-tuning data (docs/finetune_data_guide.md §4-5): generate examples from patterns.

    .venv/bin/python finetune/data_gen/generate_examples.py          # -> finetune/data/raw/generated.jsonl

A pattern is one kind of transformation. For each training dataset it finds columns and values that fit, then
writes, together and consistently: several request wordings (styles plain / casual / domain / indirect), the
contract (vocabulary v1.0, following the annotation conventions of the gold contracts: implied row_count included,
`columns_unchanged "*"` + except, row_count unchanged for column edits, null_count unchanged for conversions of
columns with missing values) and the reference code. Values always come from the data, so contracts are
non-vacuous. Patterns are Python functions here rather than YAML files (one place, easier to keep consistent).
Templates were written without looking at the evaluation requests; check_examples.py tests for leakage.
"""
import argparse
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "finetune" / "datasets"

# ---------------------------------------------------------------- dataset knowledge (names people would use)
DATASETS = {
    "air_quality": dict(entity="readings", words={"AQI": "air quality index", "PM2_5": "PM2.5", "NO2": "nitrogen dioxide",
                        "SO2": "sulphur dioxide", "O3": "ozone", "WindSpeed": "wind speed", "RespiratoryCases": "respiratory cases",
                        "CardiovascularCases": "cardiovascular cases", "HospitalAdmissions": "hospital admissions",
                        "HealthImpactScore": "health impact score", "HealthImpactClass": "health impact class"},
                        sentinels={"O3": -999.0, "AQI": -1.0}),
    "avocado": dict(entity="price records", words={"AveragePrice": "average price", "Total Volume": "total volume",
                    "4046": "small Hass volume (PLU 4046)", "4225": "large Hass volume (PLU 4225)", "4770": "extra large Hass volume (PLU 4770)",
                    "Total Bags": "total bags", "type": "avocado type"}, sentinels={"AveragePrice": 0.0},
                    dates={"Date": None}, ws="region"),
    "bank": dict(entity="clients", words={"balance": "account balance", "default": "credit default flag", "housing": "housing loan flag",
                 "loan": "personal loan flag", "contact": "contact type", "day": "contact day", "duration": "call duration",
                 "campaign": "number of contacts in this campaign", "pdays": "days since last contact", "previous": "number of earlier contacts",
                 "poutcome": "previous campaign outcome", "y": "subscription outcome"},
                 sentinels={"contact": "unknown", "poutcome": "unknown", "education": "unknown", "pdays": -1}),
    "diamonds": dict(entity="diamonds", words={"x": "length in mm", "y": "width in mm", "z": "depth in mm", "depth": "total depth percentage",
                     "table": "table width"}, sentinels={"x": 0.0, "y": 0.0},
                     ordered={"cut": ["Fair", "Good", "Very Good", "Premium", "Ideal"],
                              "color": ["J", "I", "H", "G", "F", "E", "D"],
                              "clarity": ["I1", "SI2", "SI1", "VS2", "VS1", "VVS2", "VVS1", "IF"]}),
    "imdb": dict(entity="movies", words={"Series_Title": "title", "Released_Year": "release year", "IMDB_Rating": "IMDb rating",
                 "Meta_score": "Metascore", "No_of_Votes": "number of votes", "Star1": "lead actor", "Gross": "box office gross"},
                 fills=[("Meta_score", "mul", ["IMDB_Rating", 10], "ten times the IMDb rating")], ws="Series_Title"),
    "student": dict(entity="students", words={"Medu": "mother's education", "Fedu": "father's education", "Mjob": "mother's job",
                    "Fjob": "father's job", "traveltime": "travel time", "studytime": "study time", "famrel": "family relationship score",
                    "goout": "going-out score", "Dalc": "weekday alcohol use", "Walc": "weekend alcohol use", "G1": "first-period grade",
                    "G2": "second-period grade", "G3": "final grade", "absences": "number of absences", "higher": "wants higher education"},
                    fills=[("Mjob", "identity", ["Fjob"], "the father's job")]),
    "superstore": dict(entity="orders", words={}, dates={"Order Date": "%d/%m/%Y", "Ship Date": "%d/%m/%Y"}, ws="Customer Name", sentinels={"Sales": 0.0},
                       ordered={"Ship Mode": ["Same Day", "First Class", "Second Class", "Standard Class"]}),
    "walmart": dict(entity="store weeks", words={"Weekly_Sales": "weekly sales", "Holiday_Flag": "holiday flag", "Fuel_Price": "fuel price",
                    "CPI": "consumer price index", "Unemployment": "unemployment rate"}, sentinels={"Temperature": -99.0, "Weekly_Sales": 0.0},
                    dates={"Date": "%d-%m-%Y"}),
}
FMT_WORDS = {"%d/%m/%Y": "day/month/year", "%d-%m-%Y": "day-month-year", None: "year-month-day"}


class DS:
    def __init__(self, name):
        self.name, self.cfg = name, DATASETS[name]
        self.path = f"finetune/datasets/{name}.csv"
        self.df = pd.read_csv(ROOT / self.path)
        self.entity = self.cfg["entity"]
        df = self.df
        self.num = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c]) and df[c].nunique() > 2]
        self.qty = [c for c in self.num if not re.search(r"(?i)store|postal|code|year|^day$|flag|class|id$", str(c))]
        self.obj = [c for c in df.columns if df[c].dtype == object]
        self.cat = [c for c in self.obj if 2 <= df[c].nunique() <= 15 and c not in self.cfg.get("dates", {})]
        self.text = [c for c in self.obj if df[c].nunique() > 15 and c not in self.cfg.get("dates", {})]
        self.nulls = [c for c in df.columns if df[c].isna().any()]
        self.nonnull = lambda cols: [c for c in cols if not df[c].isna().any()]
        self.ints = [c for c in df.columns if pd.api.types.is_integer_dtype(df[c])]
        self.floats = [c for c in df.columns if pd.api.types.is_float_dtype(df[c])]

    def w(self, c):
        if c in self.cfg["words"]:
            return self.cfg["words"][c]
        s = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", str(c)).replace("_", " ").replace("-", " ")
        return s.lower() if not s.isupper() else s


# ---------------------------------------------------------------- helpers
def py(v):
    """numpy scalar -> plain python, floats rounded for display stability."""
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating, float)):
        v = float(v)
        return int(v) if v.is_integer() and abs(v) < 1e12 else v
    return v


def nice(x, is_int):
    if x == 0:
        return 0
    mag = 10 ** (int(np.floor(np.log10(abs(x)))) - 1)
    v = round(x / mag) * mag
    v = round(v, 6)
    return int(round(v)) if is_int or float(v).is_integer() else v


def show(v, quote=None):
    if isinstance(v, str):
        q = quote if quote is not None else random.random() < 0.6
        return f"'{v}'" if q or " " in v else v
    if isinstance(v, float):
        return f"{v:g}"
    return str(v)


def ref(ds, c):
    """How a request names a column: exact name (sometimes in backticks) or human words."""
    r = random.random()
    return f"`{c}`" if r < 0.25 else str(c) if r < 0.5 else ds.w(c)


def leaf(c, op, v=None):
    return {"column": c, "op": op} if op in ("is_null", "not_null") else {"column": c, "op": op, "value": v}


def _key(x):
    return (x.get("column", "~"), x.get("op", "~"), json.dumps(x.get("value"), sort_keys=True))


def all_(*xs):
    return {"all": sorted(xs, key=_key)}


def any_(*xs):
    return {"any": sorted(xs, key=_key)}


NEG = {"<": ">=", "<=": ">", ">": "<=", ">=": "<", "==": "!=", "!=": "==", "in": "not_in", "not_in": "in",
       "is_null": "not_null", "not_null": "is_null"}
PYOP = {"<": "<", "<=": "<=", ">": ">", ">=": ">=", "==": "==", "!=": "!="}


def contract(change, preserve):
    return {"contract_version": "1.0", "must_change": change, "must_preserve": preserve}


ROWS_DOWN = {"type": "row_count", "target": None, "op": "decreased"}
ROWS_SAME = {"type": "row_count", "target": None, "op": "unchanged"}


def cols_same(*exc):
    return {"type": "columns_unchanged", "target": "*", "except": list(exc)}


def q(c):
    return json.dumps(c)


def lit(v):
    return repr(v) if not isinstance(v, str) else json.dumps(v)


def code(*body):
    return "def transform(df):\n    out = df.copy()\n" + "".join(f"    {b}\n" for b in body) + "    return out\n"


def mask_code(c, op, v):
    if op == "in":
        return f"out[{q(c)}].isin({lit(v)})"
    if op == "not_in":
        return f"~out[{q(c)}].isin({lit(v)})"
    if op == "is_null":
        return f"out[{q(c)}].isna()"
    if op == "not_null":
        return f"out[{q(c)}].notna()"
    return f"(out[{q(c)}] {PYOP[op]} {lit(v)})"


def filter_example(keep_expr):
    return code(f"return out[{keep_expr}].copy()").replace("    return out\n", "") if False else \
        "def transform(df):\n    out = df.copy()\n" + f"    return out[{keep_expr}].copy()\n"


def threshold(ds, c, lo=0.08, hi=0.5, side="low"):
    s = ds.df[c].dropna()
    is_int = pd.api.types.is_integer_dtype(ds.df[c]) or (s % 1 == 0).all()
    for _ in range(12):
        qv = random.uniform(lo, hi) if side == "low" else random.uniform(1 - hi, 1 - lo)
        t = nice(float(s.quantile(qv)), is_int)
        frac = (s < t).mean() if side == "low" else (s > t).mean()
        if 0.04 <= frac <= 0.6:
            return py(t)
    return None


def pick(xs, k=1):
    xs = list(xs)
    return random.sample(xs, k) if len(xs) >= k else None


OPW = {"<": ["below", "under", "less than", "lower than"], "<=": ["at most", "no more than", "{T} or less"],
       ">": ["above", "over", "more than", "greater than"], ">=": ["at least", "{T} or more", "no less than"]}


def opw(op, t):
    w = random.choice(OPW[op])
    return w.replace("{T}", show(t)) if "{T}" in w else f"{w} {show(t)}"


# ---------------------------------------------------------------- patterns: each returns a list of instances
# instance = dict(contract=..., code=..., requests={style: text})
PATTERNS = {}


def pattern(name, ttype):
    def deco(f):
        PATTERNS[name] = (ttype, f)
        return f
    return deco


def std_filter(removed, retained, keep_expr):
    return contract([{"type": "rows_removed_where", "target": None, "condition": removed}, ROWS_DOWN],
                    [{"type": "rows_retained_where", "target": None, "condition": retained}, cols_same()]), \
        filter_example(keep_expr)


@pattern("filter_remove_low", "filtering")
def p_filter_low(ds):
    c = pick(ds.nonnull(ds.num))
    if not c:
        return []
    c = c[0]
    op = random.choice(["<", "<="])
    t = threshold(ds, c, side="low")
    if t is None:
        return []
    k, e, r = ds.w(c), ds.entity, ref(ds, c)
    con, cd = std_filter(leaf(c, op, t), leaf(c, NEG[op], t), mask_code(c, NEG[op], t))
    keep = opw(NEG[op], t)
    return [dict(contract=con, code=cd, requests={
        "plain": random.choice([f"Remove the rows where {r} is {opw(op, t)}.", f"Delete every row whose {r} is {opw(op, t)}."]),
        "casual": random.choice([f"get rid of anything with {k} {opw(op, t)}", f"ditch the {e} where {k} is {opw(op, t)} pls"]),
        "domain": f"Drop the {e} whose {k} is {opw(op, t)}.",
        "indirect": random.choice([f"Only keep {e} with a {k} of {keep}.", f"I just need the {e} where {r} is {keep}."])})]


@pattern("filter_remove_high", "filtering")
def p_filter_high(ds):
    c = pick(ds.nonnull(ds.num))
    if not c:
        return []
    c = c[0]
    op = random.choice([">", ">="])
    t = threshold(ds, c, side="high")
    if t is None:
        return []
    k, e, r = ds.w(c), ds.entity, ref(ds, c)
    con, cd = std_filter(leaf(c, op, t), leaf(c, NEG[op], t), mask_code(c, NEG[op], t))
    keep = opw(NEG[op], t)
    return [dict(contract=con, code=cd, requests={
        "plain": random.choice([f"Remove all rows with {r} {opw(op, t)}.", f"Filter out the rows where {r} is {opw(op, t)}."]),
        "casual": random.choice([f"toss the {e} with {k} {opw(op, t)}", f"no {e} with {k} {opw(op, t)} please"]),
        "domain": f"Exclude {e} with a {k} {opw(op, t)} from the data.",
        "indirect": f"Keep just the {e} whose {k} is {keep}."})]


@pattern("filter_keep_range", "filtering")
def p_filter_range(ds):
    c = pick(ds.nonnull(ds.num))
    if not c:
        return []
    c = c[0]
    a, b = threshold(ds, c, 0.05, 0.3, "low"), threshold(ds, c, 0.05, 0.3, "high")
    if a is None or b is None or not a < b:
        return []
    k, e, r = ds.w(c), ds.entity, ref(ds, c)
    con, cd = std_filter(any_(leaf(c, "<", a), leaf(c, ">", b)), all_(leaf(c, ">=", a), leaf(c, "<=", b)),
                         f"out[{q(c)}].between({lit(a)}, {lit(b)})")
    return [dict(contract=con, code=cd, requests={
        "plain": f"Keep only rows where {r} is between {show(a)} and {show(b)} (both ends included).",
        "casual": f"just keep {e} with {k} from {show(a)} to {show(b)}, inclusive",
        "domain": f"Restrict the {e} to a {k} in the range {show(a)}–{show(b)}, inclusive.",
        "indirect": f"Drop every row with {r} below {show(a)} or above {show(b)}."})]


def cat_value(ds, c, maxfrac=0.6):
    vc = ds.df[c].value_counts(normalize=True)
    ok = [v for v, f in vc.items() if 0.02 <= f <= maxfrac]
    return py(random.choice(ok)) if ok else None


@pattern("filter_remove_category", "filtering")
def p_filter_cat(ds):
    c = pick(ds.cat)
    if not c:
        return []
    c = c[0]
    v = cat_value(ds, c)
    if v is None:
        return []
    k, e, r = ds.w(c), ds.entity, ref(ds, c)
    has_null = ds.df[c].isna().any()
    retained = {"not": leaf(c, "==", v)} if has_null else leaf(c, "!=", v)
    con, cd = std_filter(leaf(c, "==", v), retained, f"out[{q(c)}] != {lit(v)}")
    extra = " Rows with no value there stay." if has_null else ""
    return [dict(contract=con, code=cd, requests={
        "plain": f"Remove the rows where {r} is {show(v)}.{extra}",
        "casual": f"drop all the {show(v)} ones ({k}).{extra}",
        "domain": f"Take out every {e[:-1] if e.endswith('s') else e} with {k} equal to {show(v)}.{extra}",
        "indirect": f"Keep everything except the rows whose {r} is {show(v)}.{extra}"})]


@pattern("filter_remove_categories", "filtering")
def p_filter_cats(ds):
    c = pick(ds.nonnull(ds.cat))
    if not c:
        return []
    c = c[0]
    vc = ds.df[c].value_counts(normalize=True)
    ok = [v for v, f in vc.items() if f <= 0.35]
    if len(ok) < 3:
        return []
    vals = sorted(py(v) for v in random.sample(ok, random.choice([2, 2, 3])))
    k, e, r = ds.w(c), ds.entity, ref(ds, c)
    qq = random.random() < 0.6
    lst = ", ".join(show(v, qq) for v in vals[:-1]) + f" or {show(vals[-1], qq)}"
    con, cd = std_filter(leaf(c, "in", vals), leaf(c, "not_in", vals), f"~out[{q(c)}].isin({lit(vals)})")
    return [dict(contract=con, code=cd, requests={
        "plain": f"Remove rows where {r} is {lst}.",
        "casual": f"get rid of the {lst} {e}",
        "domain": f"Exclude {e} whose {k} is one of {', '.join(show(v, qq) for v in vals)}.",
        "indirect": f"Keep all {e} apart from those with {k} {lst}."})]


@pattern("filter_keep_categories", "filtering")
def p_filter_keep_cats(ds):
    c = pick(ds.nonnull(ds.cat))
    if not c:
        return []
    c = c[0]
    vals_all = ds.df[c].value_counts().index.tolist()
    if len(vals_all) < 3:
        return []
    vals = sorted(py(v) for v in random.sample(vals_all, random.randint(1, len(vals_all) - 1)))
    k, e, r = ds.w(c), ds.entity, ref(ds, c)
    qq = random.random() < 0.6
    sv = [show(v, qq) for v in vals]
    l_or = sv[0] if len(sv) == 1 else ", ".join(sv[:-1]) + f" or {sv[-1]}"
    l_and = sv[0] if len(sv) == 1 else ", ".join(sv[:-1]) + f" and {sv[-1]}"
    con, cd = std_filter(leaf(c, "not_in", vals), leaf(c, "in", vals), f"out[{q(c)}].isin({lit(vals)})")
    return [dict(contract=con, code=cd, requests={
        "plain": f"Keep only the rows where {r} is {l_or}.",
        "casual": f"i only care about the {l_and} {e}, drop the rest",
        "domain": f"Limit the data to {e} whose {k} is {l_or}.",
        "indirect": f"Remove every row whose {r} is not {('one of ' + ', '.join(sv)) if len(sv) > 1 else sv[0]}."})]


@pattern("filter_and", "filtering")
def p_filter_and(ds):
    c, n = pick(ds.nonnull(ds.cat)), pick(ds.num)
    if not c or not n:
        return []
    c, n = c[0], n[0]
    v = cat_value(ds, c)
    t = threshold(ds, n, 0.15, 0.6, "low")
    if v is None or t is None or not ((ds.df[c] == v) & (ds.df[n] < t)).any():
        return []
    removed = all_(leaf(c, "==", v), leaf(n, "<", t))
    retained = {"not": removed} if ds.df[n].isna().any() else any_(leaf(c, "!=", v), leaf(n, ">=", t))
    con, cd = std_filter(removed, retained, f"~((out[{q(c)}] == {lit(v)}) & (out[{q(n)}] < {lit(t)}))")
    e = ds.entity
    return [dict(contract=con, code=cd, requests={
        "plain": f"Remove rows where {ref(ds, c)} is {show(v)} and {ref(ds, n)} is below {show(t)}.",
        "casual": f"drop {show(v)} {e} that have {ds.w(n)} under {show(t)}",
        "domain": f"Exclude the {e} with {ds.w(c)} {show(v)} whose {ds.w(n)} is less than {show(t)}; keep everything else.",
        "indirect": f"Only remove a row if both hold: {ds.w(c)} is {show(v)} and {ds.w(n)} < {show(t)}."})]


@pattern("filter_or", "filtering")
def p_filter_or(ds):
    cols = pick(ds.nonnull(ds.num), 2)
    if not cols:
        return []
    a, b = cols
    ta, tb = threshold(ds, a, 0.05, 0.25, "high"), threshold(ds, b, 0.05, 0.25, "low")
    if ta is None or tb is None:
        return []
    con, cd = std_filter(any_(leaf(a, ">", ta), leaf(b, "<", tb)), all_(leaf(a, "<=", ta), leaf(b, ">=", tb)),
                         f"(out[{q(a)}] <= {lit(ta)}) & (out[{q(b)}] >= {lit(tb)})")
    e = ds.entity
    return [dict(contract=con, code=cd, requests={
        "plain": f"Remove rows where {ref(ds, a)} is above {show(ta)} or {ref(ds, b)} is below {show(tb)}.",
        "casual": f"drop {e} with {ds.w(a)} over {show(ta)}, and also the ones with {ds.w(b)} under {show(tb)}",
        "domain": f"Discard any {e} with {ds.w(a)} greater than {show(ta)} or {ds.w(b)} less than {show(tb)}.",
        "indirect": f"Keep a row only if {ds.w(a)} <= {show(ta)} and {ds.w(b)} >= {show(tb)}."})]


@pattern("filter_with_nulls", "filtering")
def p_filter_nulls(ds):
    c = pick([x for x in ds.nulls if x in ds.num])
    if not c:
        return []
    c = c[0]
    t = threshold(ds, c, side="low")
    if t is None:
        return []
    k, e, r = ds.w(c), ds.entity, ref(ds, c)
    if random.random() < 0.5:        # nulls removed too
        con, cd = std_filter(any_(leaf(c, "<", t), leaf(c, "is_null")), leaf(c, ">=", t), f"out[{q(c)}] >= {lit(t)}")
        reqs = {"plain": f"Remove rows where {r} is below {show(t)}, and also rows where it is missing.",
                "casual": f"drop {e} with {k} under {show(t)} or no {k} at all",
                "domain": f"Keep only {e} with a known {k} of at least {show(t)}.",
                "indirect": f"Rows need a {k} of {show(t)} or more to stay; blanks go as well."}
    else:                            # nulls kept
        con, cd = std_filter(leaf(c, "<", t), any_(leaf(c, ">=", t), leaf(c, "is_null")), f"~(out[{q(c)}] < {lit(t)})")
        reqs = {"plain": f"Remove rows where {r} is below {show(t)}; keep the rows where it is missing.",
                "casual": f"drop {e} with {k} under {show(t)} but leave the blank ones alone",
                "domain": f"Discard {e} with {k} below {show(t)}. Keep {e} with missing {k} values.",
                "indirect": f"Only rows with a {k} below {show(t)} should go, not the ones where {k} is empty."}
    return [dict(contract=con, code=cd, requests=reqs)]


@pattern("filter_two_categories", "filtering")
def p_filter_two_cats(ds):
    cols = pick(ds.nonnull(ds.cat), 2)
    if not cols:
        return []
    a, b = cols
    va, vb = cat_value(ds, a), cat_value(ds, b)
    if va is None or vb is None or not ((ds.df[a] == va) & (ds.df[b] == vb)).any():
        return []
    removed = all_(leaf(a, "==", va), leaf(b, "==", vb))
    con, cd = std_filter(removed, any_(leaf(a, "!=", va), leaf(b, "!=", vb)),
                         f"~((out[{q(a)}] == {lit(va)}) & (out[{q(b)}] == {lit(vb)}))")
    e = ds.entity
    return [dict(contract=con, code=cd, requests={
        "plain": f"Remove rows where {ref(ds, a)} is {show(va)} and {ref(ds, b)} is {show(vb)}.",
        "casual": f"drop the {e} that are {show(va)} ({ds.w(a)}) and {show(vb)} ({ds.w(b)}) at the same time",
        "domain": f"Exclude {e} combining {ds.w(a)} {show(va)} with {ds.w(b)} {show(vb)}.",
        "indirect": f"Keep every row unless its {ds.w(a)} is {show(va)} and its {ds.w(b)} is {show(vb)}."})]


# ---------------------------------------------------------------- deduplication
# Review fix F1 (finetune/data_gen/fix_review_issues.py): every dedup request states first/last and every contract
# has "keep". Without "keep", duplicate_count eq 0 only checks uniqueness (an empty output passes), and every
# evaluation gold contract states a keep rule. The None option is gone; data v1 was fixed in place, not regenerated.
KEEP_TXT = {"first": ["keep the first one", "keep the earliest row", "keep whichever comes first"],
            "last": ["keep the last one", "keep the latest row", "keep whichever comes last"]}


def dedup(target, exc, keep, df_subset_code):
    pred = {"type": "duplicate_count", "target": target, "op": "eq", "value": 0}
    if exc is not None:
        pred["except"] = exc
    if keep:
        pred["keep"] = keep
    kc = keep or "first"
    return contract([pred, ROWS_DOWN], [cols_same()]), \
        "def transform(df):\n" + f"    return df.drop_duplicates({df_subset_code}keep={lit(kc)}).copy()\n"


def keep_phrase(keep):
    t = random.choice(KEEP_TXT[keep])
    return f" ({t})" if t else ""


@pattern("dedup_full_rows", "deduplication")
def p_dedup_full(ds):
    if not ds.df.duplicated().any():
        return []
    keep = random.choice(["first", "last"])
    con, cd = dedup("*", [], keep, "")
    e, kp = ds.entity, keep_phrase(keep)
    return [dict(contract=con, code=cd, requests={
        "plain": f"Remove {e} that are exact duplicates of another row{kp}.",
        "casual": f"there are repeated {e} rows in here, clean them out{kp}",
        "domain": f"Some {e} appear twice with identical values in every column. Deduplicate them{kp}.",
        "indirect": f"Every one of the {e} should be listed only once{kp}."})]


def key_ok(df, key, maxfrac=0.4):
    d = df.duplicated(subset=key).mean()    # pandas and the verifier both treat missing values as equal keys
    return 0.005 < d <= maxfrac


@pattern("dedup_on_key", "deduplication")
def p_dedup_key(ds):
    cand = list(ds.df.columns)
    for _ in range(20):
        key = sorted(random.sample(cand, random.choice([1, 2, 2, 3])), key=list(ds.df.columns).index)
        if key_ok(ds.df, key):
            break
    else:
        return []
    keep = random.choice(["first", "last", "first"])
    con, cd = dedup(key, None, keep, f"subset={lit(key)}, ")
    e, kp = ds.entity, keep_phrase(keep)
    names = " and ".join(ref(ds, c) for c in key)
    return [dict(contract=con, code=cd, requests={
        "plain": f"Remove duplicates based on {names}{kp}.",
        "casual": f"dedupe on {', '.join(map(str, key))}{kp}",
        "domain": f"Rows with the same {' and '.join(ds.w(c) for c in key)} are the same record; leave only one per combination{kp}.",
        "indirect": f"Each {'combination' if len(key) > 1 else 'value'} of {names} should appear only once{kp}."})]


@pattern("dedup_all_but", "deduplication")
def p_dedup_except(ds):
    cols = list(ds.df.columns)
    for _ in range(20):
        exc = sorted(random.sample(cols, random.choice([1, 1, 2])), key=cols.index)
        key = [c for c in cols if c not in exc]
        if key_ok(ds.df, key, 0.2):
            break
    else:
        return []
    keep = random.choice(["first", "last"])
    con, cd = dedup("*", exc, keep, f"subset={lit(key)}, ")
    con["must_change"][0]["except"] = exc
    cd = cd.replace(f"subset={lit(key)}", f"subset=[c for c in df.columns if c not in {lit(exc)}]")
    e, kp = ds.entity, keep_phrase(keep)
    ex = " and ".join(ref(ds, c) for c in exc)
    return [dict(contract=con, code=cd, requests={
        "plain": f"Remove rows that are duplicates in all columns other than {ex}{kp}.",
        "casual": f"dedupe ignoring {ex}{kp}",
        "domain": f"Two {e} count as the same if they only differ in {ex}. Remove the repeats{kp}.",
        "indirect": f"Treat rows as duplicates when everything except {ex} matches{kp}."})]


# ---------------------------------------------------------------- missing values
def fill_contract(c, fn_pred):
    return contract([fn_pred, {"type": "null_count", "target": c, "op": "eq", "value": 0}],
                    [{"type": "column_values_unchanged", "target": c, "where": leaf(c, "not_null")}, cols_same(c), ROWS_SAME])


def const_fill(ds, c, v, why):
    pred = {"type": "column_value_relation", "target": c, "where": leaf(c, "is_null"), "fn": "const", "args": [{"lit": v}]}
    return fill_contract(c, pred), code(f"out[{q(c)}] = out[{q(c)}].fillna({lit(v)})")


@pattern("fill_constant_number", "missing_values")
def p_fill_num(ds):
    c = pick([x for x in ds.nulls if x in ds.num])
    if not c:
        return []
    c = c[0]
    v = random.choice([0, 0, -1, 1, 999, 100, 10]) if (ds.df[c].dropna() % 1 == 0).all() else random.choice([0, -1, 0.5, 1.5, 99.9])
    v = py(v)
    con, cd = const_fill(ds, c, v, "")
    k, r = ds.w(c), ref(ds, c)
    return [dict(contract=con, code=cd, requests={
        "plain": f"Fill the missing values in {r} with {show(v)}.",
        "casual": f"where {k} is blank just put {show(v)}",
        "domain": f"Replace empty {k} entries with {show(v)}; other values stay as they are.",
        "indirect": f"There should be no missing {k} afterwards: use {show(v)} for the gaps."})]


@pattern("fill_constant_text", "missing_values")
def p_fill_text(ds):
    c = pick([x for x in ds.nulls if x in ds.obj])
    if not c:
        return []
    c = c[0]
    v = random.choice(["Unknown", "unknown", "Not available", "N/A", "Missing", "none", "Other", "not given", "?"])
    con, cd = const_fill(ds, c, v, "")
    k, r = ds.w(c), ref(ds, c)
    return [dict(contract=con, code=cd, requests={
        "plain": f"Put '{v}' wherever {r} is empty.",
        "casual": f"blank {k}? make it '{v}'",
        "domain": f"Where the {k} wasn't recorded, write '{v}' instead.",
        "indirect": f"Rows with no {k} value get the label '{v}'."})]


@pattern("fill_median", "missing_values")
def p_fill_median(ds):
    c = pick([x for x in ds.nulls if x in ds.num])
    if not c:
        return []
    c = c[0]
    v = py(ds.df[c].median())
    con, cd = const_fill(ds, c, v, "")
    cd = code(f"out[{q(c)}] = out[{q(c)}].fillna(out[{q(c)}].median())")
    k, r = ds.w(c), ref(ds, c)
    return [dict(contract=con, code=cd, requests={
        "plain": f"Impute missing {r} using the column median.",
        "casual": f"impute the empty {k} values with the median",
        "domain": f"Use the median {k} for {ds.entity} where it is missing.",
        "indirect": f"Replace each blank {k} by the column's median value."})]


@pattern("fill_most_common", "missing_values")
def p_fill_mode(ds):
    c = pick([x for x in ds.nulls if x in ds.cat])
    if not c:
        return []
    c = c[0]
    v = py(ds.df[c].mode().iloc[0])
    con, cd = const_fill(ds, c, v, "")
    cd = code(f"out[{q(c)}] = out[{q(c)}].fillna(out[{q(c)}].mode().iloc[0])")
    k, r = ds.w(c), ref(ds, c)
    return [dict(contract=con, code=cd, requests={
        "plain": f"Fill missing {r} with the most frequent value.",
        "casual": f"fill in blank {k} using whatever value is most common",
        "domain": f"Where {k} is unknown, assume the most common {k}.",
        "indirect": f"Impute {k} with its mode."})]


@pattern("fill_from_other", "missing_values")
def p_fill_other(ds):
    out = []
    for target, fn, args, desc in ds.cfg.get("fills", []):
        if not ds.df[target].isna().any():
            continue
        if fn == "identity":
            pred_args, expr = [{"col": args[0]}], f"out[{q(args[0])}]"
        else:
            pred_args, expr = [{"col": args[0]}, {"lit": args[1]}], f"out[{q(args[0])}] * {lit(args[1])}"
        pred = {"type": "column_value_relation", "target": target, "where": leaf(target, "is_null"), "fn": fn, "args": pred_args}
        con = fill_contract(target, pred)
        if ds.df.loc[ds.df[target].isna(), args[0]].isna().any():
            con["must_change"] = con["must_change"][:1]
        out.append(dict(contract=con, code=code(f"out[{q(target)}] = out[{q(target)}].fillna({expr})"), requests={
            "plain": f"Where {ref(ds, target)} is missing, fill it with {desc}.",
            "casual": f"no {ds.w(target)}? use {desc} instead",
            "domain": f"For {ds.entity} without a {ds.w(target)}, take {desc}.",
            "indirect": f"Gaps in {ds.w(target)} should be filled from {desc}; known values stay."}))
    return out


@pattern("sentinel_to_missing", "missing_values")
def p_sentinel(ds):
    s = ds.cfg.get("sentinels", {})
    if not s:
        return []
    c = random.choice(list(s))
    v = s[c]
    if not (ds.df[c] == v).any():
        return []
    v = py(v)
    con = contract([{"type": "column_value_relation", "target": c, "where": leaf(c, "==", v), "fn": "null"},
                    {"type": "null_count", "target": c, "op": "increased"}],
                   [{"type": "column_values_unchanged", "target": c, "where": leaf(c, "!=", v)}, cols_same(c), ROWS_SAME])
    cd = code(f"out[{q(c)}] = out[{q(c)}].replace({lit(v)}, np.nan)")
    k, r = ds.w(c), ref(ds, c)
    return [dict(contract=con, code=cd, requests={
        "plain": f"In {r}, the value {show(v)} means not recorded. Replace it with a missing value.",
        "casual": f"{show(v)} in {k} is really just 'no data', turn those into NaN",
        "domain": f"Treat {k} = {show(v)} as missing.",
        "indirect": f"Set {k} to null wherever it currently says {show(v)}."})]


@pattern("drop_missing_column", "missing_values")
def p_drop_null(ds):
    c = pick(ds.nulls)
    if not c:
        return []
    c = c[0]
    con, cd = std_filter(leaf(c, "is_null"), leaf(c, "not_null"), f"out[{q(c)}].notna()")
    k, e, r = ds.w(c), ds.entity, ref(ds, c)
    return [dict(contract=con, code=cd, requests={
        "plain": f"Remove rows with a missing {r}.",
        "casual": f"drop {e} that have no {k}",
        "domain": f"Discard {e} whose {k} is unknown.",
        "indirect": f"Keep only rows where {r} has a value."})]


@pattern("drop_missing_any", "missing_values")
def p_drop_any(ds):
    if len(ds.nulls) < 2:
        return []
    sub = sorted(random.sample(ds.nulls, random.randint(2, len(ds.nulls))))
    all_cols = len(sub) == len(ds.nulls)
    con, cd = std_filter(any_(*[leaf(c, "is_null") for c in sub]), all_(*[leaf(c, "not_null") for c in sub]),
                         "out.notna().all(axis=1)" if all_cols else f"out[{lit(sub)}].notna().all(axis=1)")
    e = ds.entity
    if all_cols:
        reqs = {"plain": "Remove every row that contains at least one missing value.",
                "casual": "throw away any row with a blank in it",
                "domain": f"Only complete {e} should remain: no empty fields anywhere.",
                "indirect": "Keep rows without any NaN."}
    else:
        names = ", ".join(ref(ds, c) for c in sub[:-1]) + f" or {ref(ds, sub[-1])}"
        reqs = {"plain": f"Remove rows missing {names}.",
                "casual": f"drop {e} with a gap in {names}",
                "domain": f"Discard {e} where any of {', '.join(ds.w(c) for c in sub)} is missing.",
                "indirect": f"Keep a row only if {', '.join(map(str, sub))} are all filled in."}
    return [dict(contract=con, code=cd, requests=reqs)]


# ---------------------------------------------------------------- type conversion
def conv_preserve(ds, c, rel=None):
    p = [rel or {"type": "column_values_unchanged", "target": c}]
    if ds.df[c].isna().any():
        p.append({"type": "null_count", "target": c, "op": "unchanged"})
    return p + [cols_same(c), ROWS_SAME]


@pattern("to_category", "type_conversion")
def p_to_cat(ds):
    c = pick(ds.cat)
    if not c:
        return []
    c = c[0]
    con = contract([{"type": "column_dtype", "target": c, "dtype": "category"}], conv_preserve(ds, c))
    k, r = ds.w(c), ref(ds, c)
    extra = " Missing values should stay missing." if ds.df[c].isna().any() else ""
    return [dict(contract=con, code=code(f"out[{q(c)}] = out[{q(c)}].astype('category')"), requests={
        "plain": f"Change the dtype of {r} to category.{extra}",
        "casual": f"make {k} a category dtype{extra}",
        "domain": f"{k.capitalize()} only takes a few values; store it as a pandas category.{extra}",
        "indirect": f"Change the type of {r} to category without changing any value.{extra}"})]


@pattern("to_ordered_category", "type_conversion")
def p_to_ordered(ds):
    out = []
    for c, order in ds.cfg.get("ordered", {}).items():
        if ds.df[c].isna().any() or not set(ds.df[c].unique()) <= set(order):
            continue
        con = contract([{"type": "column_dtype", "target": c, "dtype": "category", "ordered": True, "categories": order}],
                       conv_preserve(ds, c))
        cd = code(f"out[{q(c)}] = pd.Categorical(out[{q(c)}], categories={lit(order)}, ordered=True)")
        seq = " < ".join(order)
        k, r = ds.w(c), ref(ds, c)
        out.append(dict(contract=con, code=cd, requests={
            "plain": f"Make {r} an ordered categorical with the order {seq}.",
            "casual": f"turn {k} into an ordered category, lowest to highest: {', '.join(order)}",
            "domain": f"{k.capitalize()} has a natural ranking ({', '.join(order)}, from lowest to highest); encode it as an ordered category.",
            "indirect": f"Store {r} as a sorted category so that {order[0]} < ... < {order[-1]} (order: {', '.join(order)})."}))
    return out


@pattern("text_to_datetime", "type_conversion")
def p_to_datetime(ds):
    out = []
    for c, fmt in ds.cfg.get("dates", {}).items():
        args = [{"col": c}, {"lit": "datetime"}] + ([{"lit": fmt}] if fmt else [])
        rel = {"type": "column_value_relation", "target": c, "fn": "cast", "args": args}
        con = contract([{"type": "column_dtype", "target": c, "dtype": "datetime"}], conv_preserve(ds, c, rel))
        f = f", format={lit(fmt)}" if fmt else ""
        cd = code(f"out[{q(c)}] = pd.to_datetime(out[{q(c)}]{f})")
        fw = FMT_WORDS[fmt]
        k, r = ds.w(c), ref(ds, c)
        out.append(dict(contract=con, code=cd, requests={
            "plain": f"Convert {r} from text to datetime. The dates are written {fw}.",
            "casual": f"{k} is a string right now ({fw}), make it a proper date",
            "domain": f"Parse the {k} column as dates ({fw} format).",
            "indirect": f"I need {r} as a datetime type; the text uses {fw}."}))
    return out


@pattern("float_to_int", "type_conversion")
def p_float_int(ds):
    c = pick([x for x in ds.floats if not ds.df[x].isna().any() and (ds.df[x] % 1 == 0).all()])
    if not c:
        return []
    c = c[0]
    con = contract([{"type": "column_dtype", "target": c, "dtype": "int"}], conv_preserve(ds, c))
    k, r = ds.w(c), ref(ds, c)
    return [dict(contract=con, code=code(f"out[{q(c)}] = out[{q(c)}].astype('int64')"), requests={
        "plain": f"Convert {r} from float to integer.",
        "casual": f"{k} only has whole numbers, store it as int",
        "domain": f"Change {k} to an integer type; the values themselves don't change.",
        "indirect": f"Drop the decimal type on {r} and make it int64."})]


@pattern("int_to_float", "type_conversion")
def p_int_float(ds):
    cand = [x for x in ds.ints]
    if not cand:
        return []
    cols = sorted(random.sample(cand, min(len(cand), random.choice([1, 1, 2]))), key=list(ds.df.columns).index)
    con = contract([{"type": "column_dtype", "target": c, "dtype": "float"} for c in cols],
                   [{"type": "column_values_unchanged", "target": cols if len(cols) > 1 else cols[0]}, cols_same(*cols), ROWS_SAME])
    names = " and ".join(ref(ds, c) for c in cols)
    cd = code(f"out[{lit(cols)}] = out[{lit(cols)}].astype(float)") if len(cols) > 1 else code(f"out[{q(cols[0])}] = out[{q(cols[0])}].astype(float)")
    return [dict(contract=con, code=cd, requests={
        "plain": f"Convert {names} to float.",
        "casual": f"cast {names} to floats",
        "domain": f"{' and '.join(ds.w(c) for c in cols).capitalize()}: switch the dtype from int to float.",
        "indirect": f"{names[0].upper() + names[1:]} should have a floating-point dtype."})]


@pattern("number_to_text", "type_conversion")
def p_to_text(ds):
    c = pick([x for x in ds.ints if not ds.df[x].isna().any()])
    if not c:
        return []
    c = c[0]
    rel = {"type": "column_value_relation", "target": c, "fn": "cast", "args": [{"col": c}, {"lit": "string"}]}
    con = contract([{"type": "column_dtype", "target": c, "dtype": "string"}], conv_preserve(ds, c, rel))
    k, r = ds.w(c), ref(ds, c)
    return [dict(contract=con, code=code(f"out[{q(c)}] = out[{q(c)}].astype(str)"), requests={
        "plain": f"Convert {r} to text.",
        "casual": f"make {k} a string column",
        "domain": f"{k.capitalize()} is a label, not a quantity; store it as strings.",
        "indirect": f"I want {r} as str instead of a number."})]


@pattern("yes_no_to_bool", "type_conversion")
def p_bool(ds):
    c = pick([x for x in ds.obj if set(ds.df[x].dropna().unique()) == {"yes", "no"} and not ds.df[x].isna().any()])
    if not c:
        return []
    c = c[0]
    con = contract([{"type": "column_dtype", "target": c, "dtype": "bool"},
                    {"type": "column_value_relation", "target": c, "fn": "map", "args": [{"col": c}, {"lit": {"no": False, "yes": True}}]}],
                   [cols_same(c), ROWS_SAME])
    k, r = ds.w(c), ref(ds, c)
    return [dict(contract=con, code=code(f"out[{q(c)}] = out[{q(c)}].map({{'yes': True, 'no': False}})"), requests={
        "plain": f"Convert {r} from yes/no to boolean.",
        "casual": f"{k}: turn yes into True and no into False",
        "domain": f"Make {k} a True/False column (yes = True).",
        "indirect": f"Replace the yes/no strings in {r} with real booleans."})]


@pattern("text_to_number", "type_conversion")
def p_text_number(ds):
    out = []
    for c in ds.obj:
        num = pd.to_numeric(ds.df[c], errors="coerce")
        bad = num.isna() & ds.df[c].notna()
        if num.notna().mean() < 0.9 or ds.df[c].nunique() < 15:
            continue
        rel = {"type": "column_value_relation", "target": c, "fn": "cast", "args": [{"col": c}, {"lit": "float"}]}
        change = [{"type": "column_dtype", "target": c, "dtype": "float"}]
        if bad.any():
            change.append({"type": "null_count", "target": c, "op": "increased"})
        con = contract(change, [rel, cols_same(c), ROWS_SAME])
        k, r = ds.w(c), ref(ds, c)
        extra = " Entries that aren't numbers become missing." if bad.any() else ""
        out.append(dict(contract=con, code=code(f"out[{q(c)}] = pd.to_numeric(out[{q(c)}], errors='coerce').astype(float)"), requests={
            "plain": f"Convert {r} to a numeric (float) column.{extra}",
            "casual": f"{k} is stored as text, make it a float{extra}",
            "domain": f"Parse {k} as a number.{extra}",
            "indirect": f"I want {r} to be float, not object.{extra}"}))
    return out



def multi_conv(ds, cols, dtype, code_expr, rel_fn=None):
    """One conversion applied to several columns (one column_dtype per column)."""
    change = [{"type": "column_dtype", "target": c, "dtype": dtype} for c in cols]
    if rel_fn:
        pres = [{"type": "column_value_relation", "target": c, "fn": "cast", "args": [{"col": c}, {"lit": dtype}]} for c in cols]
    else:
        pres = [{"type": "column_values_unchanged", "target": cols}]
    pres += [{"type": "null_count", "target": c, "op": "unchanged"} for c in cols if ds.df[c].isna().any()]
    return contract(change, pres + [cols_same(*cols), ROWS_SAME]), code(f"out[{lit(cols)}] = out[{lit(cols)}]{code_expr}")


@pattern("several_to_category", "type_conversion")
def p_many_cat(ds):
    cols = pick(ds.cat, random.choice([2, 2, 3]))
    if not cols:
        return []
    cols = sorted(cols, key=list(ds.df.columns).index)
    con, cd = multi_conv(ds, cols, "category", ".astype('category')")
    names = ", ".join(ref(ds, c) for c in cols[:-1]) + f" and {ref(ds, cols[-1])}"
    return [dict(contract=con, code=cd, requests={
        "plain": f"Convert {names} to the category dtype.",
        "casual": f"make {names} categoricals",
        "domain": f"{', '.join(ds.w(c) for c in cols).capitalize()} are labels with few values; store each as a category.",
        "indirect": f"Change {names} to pandas categories; the values themselves stay the same."})]


@pattern("several_to_text", "type_conversion")
def p_many_text(ds):
    cols = pick([x for x in ds.ints if not ds.df[x].isna().any()], 2)
    if not cols:
        return []
    cols = sorted(cols, key=list(ds.df.columns).index)
    con, cd = multi_conv(ds, cols, "string", ".astype(str)", rel_fn="cast")
    names = f"{ref(ds, cols[0])} and {ref(ds, cols[1])}"
    return [dict(contract=con, code=cd, requests={
        "plain": f"Convert {names} to strings.",
        "casual": f"turn {names} into text columns",
        "domain": f"Store {ds.w(cols[0])} and {ds.w(cols[1])} as text instead of numbers.",
        "indirect": f"Both {names} should have a string dtype."})]


@pattern("several_to_float", "type_conversion")
def p_many_float(ds):
    cand = [x for x in ds.ints]
    cols = pick(cand, min(3, len(cand))) if len(cand) >= 3 else None
    if not cols:
        return []
    cols = sorted(cols, key=list(ds.df.columns).index)
    con, cd = multi_conv(ds, cols, "float", ".astype(float)")
    names = ", ".join(ref(ds, c) for c in cols[:-1]) + f" and {ref(ds, cols[-1])}"
    return [dict(contract=con, code=cd, requests={
        "plain": f"Cast {names} to float.",
        "casual": f"{names} -> floats please",
        "domain": f"We need {', '.join(ds.w(c) for c in cols)} as floating-point numbers.",
        "indirect": f"Change the dtype of {names} from integer to float."})]

# ---------------------------------------------------------------- column transformation
def new_col_contract(new, rel_fn, args):
    return contract([{"type": "columns_added", "target": [new]},
                     {"type": "column_value_relation", "target": new, "fn": rel_fn, "args": args}],
                    [cols_same(), ROWS_SAME])


def inplace_contract(c, rel_fn, args):
    return contract([{"type": "column_value_relation", "target": c, "fn": rel_fn, "args": args}], [cols_same(c), ROWS_SAME])


def snake(s):
    return re.sub(r"[^0-9a-z]+", "_", re.sub(r"(?<=[a-z])(?=[A-Z])", "_", str(s)).lower()).strip("_")


@pattern("rename_column", "column_transformation")
def p_rename(ds):
    c = random.choice(list(ds.df.columns))
    new = snake(c)
    if new == c or new in ds.df.columns:
        new = random.choice([f"{snake(c)}_value", f"{snake(c)}_raw", f"{ds.name}_{snake(c)}", f"col_{snake(c)}"])
    con = contract([{"type": "columns_added", "target": [new]}, {"type": "columns_removed", "target": [c]},
                    {"type": "column_value_relation", "target": new, "fn": "identity", "args": [{"col": c}]}],
                   [cols_same(c), ROWS_SAME])
    cd = "def transform(df):\n" + f"    return df.rename(columns={{{q(c)}: {q(new)}}})\n"
    return [dict(contract=con, code=cd, requests={
        "plain": f"Change the name of column {c!r} to {new!r}.",
        "casual": f"call the {c} column {new} instead",
        "domain": f"For consistency, the {ds.w(c)} column should be named {new}.",
        "indirect": f"{c} -> {new} (just the column name, nothing else)."})]


ARITH = {"add": ("+", "plus", "the sum of {a} and {b}", "{a}_plus_{b}"),
         "sub": ("-", "minus", "{a} minus {b}", "{a}_minus_{b}"),
         "mul": ("*", "times", "{a} multiplied by {b}", "{a}_x_{b}"),
         "div": ("/", "divided by", "{a} divided by {b}", "{a}_per_{b}")}


@pattern("new_column_two_columns", "column_transformation")
def p_arith2(ds):
    cols = pick(ds.qty, 2)
    if not cols:
        return []
    a, b = cols
    fn = random.choice(list(ARITH))
    if fn == "div" and (ds.df[b] == 0).any():
        return []
    sym, word, desc, nm = ARITH[fn]
    new = nm.format(a=snake(a), b=snake(b))
    con = new_col_contract(new, fn, [{"col": a}, {"col": b}])
    cd = code(f"out[{q(new)}] = out[{q(a)}] {sym} out[{q(b)}]")
    d = desc.format(a=ref(ds, a), b=ref(ds, b))
    return [dict(contract=con, code=cd, requests={
        "plain": f"Add a new column {new} with {d}.",
        "casual": f"new col {new} = {a} {sym} {b}",
        "domain": f"Create {new}, which is {desc.format(a=ds.w(a), b=ds.w(b))} for each of the {ds.entity}.",
        "indirect": f"I'd like an extra column, {new}, computed as {d}."})]


@pattern("scale_by_constant", "column_transformation")
def p_scale(ds):
    c = pick(ds.qty)
    if not c:
        return []
    c = c[0]
    fn, k = random.choice([("div", 1000), ("div", 100), ("mul", 100), ("mul", 1.1), ("mul", 2), ("div", 60)])
    inplace = random.random() < 0.5
    sym = "/" if fn == "div" else "*"
    word = "divided by" if fn == "div" else "multiplied by"
    if inplace:
        con = inplace_contract(c, fn, [{"col": c}, {"lit": k}])
        cd = code(f"out[{q(c)}] = out[{q(c)}] {sym} {lit(k)}")
        reqs = {"plain": f"Replace {ref(ds, c)} by its value {word} {show(k)}.",
                "casual": f"{sym}{show(k)} the {ds.w(c)} column in place",
                "domain": f"Rescale {ds.w(c)}: every value {word} {show(k)}, in the same column.",
                "indirect": f"Overwrite {c} with {c} {sym} {show(k)}."}
    else:
        new = f"{snake(c)}_{'div' if fn == 'div' else 'x'}{str(k).replace('.', '_')}"
        con = new_col_contract(new, fn, [{"col": c}, {"lit": k}])
        cd = code(f"out[{q(new)}] = out[{q(c)}] {sym} {lit(k)}")
        reqs = {"plain": f"Add a column {new} equal to {ref(ds, c)} {word} {show(k)}.",
                "casual": f"make {new} = {c} {sym} {show(k)}",
                "domain": f"Create {new} holding the {ds.w(c)} {word} {show(k)}; keep the original column.",
                "indirect": f"Without touching {c}, add {new} as {c} {sym} {show(k)}."}
    return [dict(contract=con, code=cd, requests=reqs)]


@pattern("round_column", "column_transformation")
def p_round(ds):
    cand = [x for x in ds.floats if (ds.df[x].dropna() % 1 != 0).any()]
    c = pick(cand)
    if not c:
        return []
    c = c[0]
    n = random.choice([0, 1, 1, 2])
    if (ds.df[c].dropna().round(n) == ds.df[c].dropna()).all():
        return []
    con = inplace_contract(c, "round", [{"col": c}, {"lit": n}])
    places = "a whole number" if n == 0 else f"{n} decimal place{'s' if n > 1 else ''}"
    return [dict(contract=con, code=code(f"out[{q(c)}] = out[{q(c)}].round({n})"), requests={
        "plain": f"Round the values in {ref(ds, c)} so they have {places}.",
        "casual": f"{ds.w(c)} with {places} please (just round it)",
        "domain": f"{ds.w(c).capitalize()} has too many decimals; round each value to {places}.",
        "indirect": f"Keep {c} but with values rounded to {places}."})]


@pattern("clip_column", "column_transformation")
def p_clip(ds):
    c = pick(ds.nonnull(ds.qty)) or pick(ds.qty)
    if not c:
        return []
    c = c[0]
    mode = random.choice(["hi", "lo", "both"])
    hi = threshold(ds, c, 0.03, 0.2, "high") if mode in ("hi", "both") else None
    lo = threshold(ds, c, 0.03, 0.2, "low") if mode in ("lo", "both") else None
    if (mode in ("hi", "both") and hi is None) or (mode in ("lo", "both") and lo is None) or (lo is not None and hi is not None and lo >= hi):
        return []
    if pd.api.types.is_integer_dtype(ds.df[c]) and any(isinstance(x, float) for x in (lo, hi) if x is not None):
        return []
    con = inplace_contract(c, "clip", [{"col": c}, {"lit": lo}, {"lit": hi}])
    cd = code(f"out[{q(c)}] = out[{q(c)}].clip(lower={lit(lo)}, upper={lit(hi)})")
    k, r = ds.w(c), ref(ds, c)
    if mode == "hi":
        reqs = {"plain": f"Cap {r} at {show(hi)}: larger values become {show(hi)}.", "casual": f"anything above {show(hi)} in {k} should just be {show(hi)}",
                "domain": f"Limit {k} to a maximum of {show(hi)}.", "indirect": f"No {k} value may exceed {show(hi)}; set bigger ones to {show(hi)}."}
    elif mode == "lo":
        reqs = {"plain": f"Floor {r} at {show(lo)}: smaller values become {show(lo)}.", "casual": f"bump any {k} under {show(lo)} up to {show(lo)}",
                "domain": f"Set a minimum {k} of {show(lo)}.", "indirect": f"Values of {r} below {show(lo)} should be raised to {show(lo)}."}
    else:
        reqs = {"plain": f"Clip {r} to the range {show(lo)} to {show(hi)}.", "casual": f"squash {k} into [{show(lo)}, {show(hi)}]",
                "domain": f"Bound {k} between {show(lo)} and {show(hi)}; out-of-range values take the nearest bound.",
                "indirect": f"Make sure every {k} lies within {show(lo)}..{show(hi)} by clipping."}
    return [dict(contract=con, code=cd, requests=reqs)]


@pattern("change_case", "column_transformation")
def p_case(ds):
    cand = ds.cat + ds.text
    c = pick(cand)
    if not c:
        return []
    c = c[0]
    s = ds.df[c].dropna().astype(str)
    fn = random.choice(["lower", "upper"])
    if not (s != (s.str.lower() if fn == "lower" else s.str.upper())).any():
        return []
    con = inplace_contract(c, fn, [{"col": c}])
    word = "lower case" if fn == "lower" else "upper case"
    k, r = ds.w(c), ref(ds, c)
    return [dict(contract=con, code=code(f"out[{q(c)}] = out[{q(c)}].str.{fn}()"), requests={
        "plain": f"Convert the values of {r} to {word}.",
        "casual": f"make {k} all {word}",
        "domain": f"Standardise {k} by writing it in {word}.",
        "indirect": f"Every {k} entry should be {word}."})]


@pattern("strip_whitespace", "column_transformation")
def p_strip(ds):
    c = ds.cfg.get("ws")
    if not c:
        return []
    con = inplace_contract(c, "strip", [{"col": c}])
    k, r = ds.w(c), ref(ds, c)
    return [dict(contract=con, code=code(f"out[{q(c)}] = out[{q(c)}].str.strip()"), requests={
        "plain": f"Trim the spaces at the beginning and end of {r}.",
        "casual": f"{k} has stray spaces around it, strip them",
        "domain": f"Clean up {k} by removing surrounding whitespace.",
        "indirect": f"No {k} value should start or end with a space."})]


@pattern("map_values", "column_transformation")
def p_map(ds):
    c = pick(ds.cat)
    if not c:
        return []
    c = c[0]
    vals = [py(v) for v in ds.df[c].dropna().unique()]
    if len(vals) < 2:
        return []
    src = sorted(random.sample(vals, random.randint(1, min(3, len(vals)))), key=str)
    mapping = {v: (str(v).upper() if random.random() < 0.3 else f"{str(v).replace('_', ' ').title()} ({c})"
                   if random.random() < 0.3 else random.choice(["Other", "Group A", "Misc", f"{v}_new"])) for v in src}
    mapping = {k: v for k, v in mapping.items() if v != k}
    if not mapping:
        return []
    con = inplace_contract(c, "map", [{"col": c}, {"lit": {str(k): v for k, v in mapping.items()}}])
    pairs = ", ".join(f"{show(k, True)} -> {show(v, True)}" for k, v in mapping.items())
    k_, r = ds.w(c), ref(ds, c)
    return [dict(contract=con, code=code(f"out[{q(c)}] = out[{q(c)}].replace({lit(mapping)})"), requests={
        "plain": f"In {r}, replace these values: {pairs}. Leave other values unchanged.",
        "casual": f"recode {k_}: {pairs}",
        "domain": f"Relabel {k_} as follows: {pairs}; everything else keeps its label.",
        "indirect": f"Map {pairs} in the {c} column; nothing else changes."})]


@pattern("drop_column", "column_transformation")
def p_drop_col(ds):
    c = random.choice(list(ds.df.columns))
    con = contract([{"type": "columns_removed", "target": [c]}], [cols_same(c), ROWS_SAME])
    cd = "def transform(df):\n" + f"    return df.drop(columns=[{q(c)}])\n"
    return [dict(contract=con, code=cd, requests={
        "plain": f"Delete the {ref(ds, c)} column.",
        "casual": f"we don't need {c} anymore, remove that column",
        "domain": f"Drop the {ds.w(c)} field from the table.",
        "indirect": f"Keep every column except {c}."})]


@pattern("copy_column", "column_transformation")
def p_copy(ds):
    c = random.choice(list(ds.df.columns))
    new = f"{snake(c)}_{random.choice(['copy', 'orig', 'backup', 'raw'])}"
    con = new_col_contract(new, "identity", [{"col": c}])
    return [dict(contract=con, code=code(f"out[{q(new)}] = out[{q(c)}]"), requests={
        "plain": f"Add a column {new} that is a copy of {ref(ds, c)}.",
        "casual": f"duplicate {c} into a new column called {new}",
        "domain": f"Before we edit {ds.w(c)}, keep its current values in a new column {new}.",
        "indirect": f"{new} should hold exactly the same values as {c}."})]


@pattern("constant_column", "column_transformation")
def p_const(ds):
    new, v = random.choice([("source", ds.name), ("currency", "USD"), ("batch", 1), ("is_checked", False),
                            ("dataset_version", "v2"), ("region_code", "EU"), ("weight", 1.0)])
    if new in ds.df.columns:
        return []
    con = new_col_contract(new, "const", [{"lit": v}])
    return [dict(contract=con, code=code(f"out[{q(new)}] = {lit(v)}"), requests={
        "plain": f"Add a column {new} with the value {show(v, True)} in every row.",
        "casual": f"tack on a {new} column, all {show(v, True)}",
        "domain": f"Tag all {ds.entity} with {new} = {show(v, True)}.",
        "indirect": f"Every row needs a new field {new} set to {show(v, True)}."})]


# ---------------------------------------------------------------- driver
ABBR = {"filtering": "filt", "deduplication": "dedup", "missing_values": "miss", "type_conversion": "conv",
        "column_transformation": "col"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=ROOT / "finetune" / "data" / "raw" / "generated.jsonl")
    ap.add_argument("--tries", type=int, default=14, help="attempts per dataset x pattern (different columns/values)")
    ap.add_argument("--styles", type=int, default=2, help="request wordings kept per instance")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--type-cap", type=int, default=1100, help="max raw examples per transformation type")
    args = ap.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    rows, seen, counters = [], set(), defaultdict(int)
    for name in DATASETS:
        ds = DS(name)
        for pname, (ttype, fn) in PATTERNS.items():
            insts, sigs = [], set()
            small = ttype in ("deduplication", "missing_values", "type_conversion")
            for _ in range(args.tries * (6 if ttype == "deduplication" else 3 if small else 1)):
                for inst in fn(ds):
                    sig = json.dumps(inst["contract"], sort_keys=True)
                    if sig not in sigs:
                        sigs.add(sig)
                        insts.append(inst)
            for inst in insts:
                n_styles = 4 if small or len(insts) < 4 else args.styles
                for style in random.sample(list(inst["requests"]), min(n_styles, len(inst["requests"]))):
                    req = " ".join(inst["requests"][style].split())
                    req = req[0].upper() + req[1:] if style != "casual" else req
                    if req.lower() in seen:
                        continue
                    seen.add(req.lower())
                    counters[(name, ttype)] += 1
                    rows.append(dict(id=f"{name}_{ABBR[ttype]}_{counters[(name, ttype)]:04d}", source_dataset=name,
                                     data_file=ds.path, transformation_type=ttype, pattern=pname, style=style,
                                     request=req, contract=inst["contract"], reference_code=inst["code"]))
    # balance: no type above --type-cap, no pattern above 5% of the total (random subsample, seeded)
    by_type = defaultdict(list)
    for r in rows:
        by_type[r["transformation_type"]].append(r)
    rows = [r for t, rs in by_type.items() for r in (random.sample(rs, args.type_cap) if len(rs) > args.type_cap else rs)]
    for _ in range(4):                  # caps shrink the total, so apply them a few times
        for field, share in (("source_dataset", 0.18), ("pattern", 0.05)):
            cap = int(share * len(rows))
            groups = defaultdict(list)
            for r in rows:
                groups[r[field]].append(r)
            rows = [r for g, rs in groups.items() for r in (random.sample(rs, cap) if len(rs) > cap else rs)]
    rows.sort(key=lambda r: r["id"])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"{len(rows)} examples, {len(PATTERNS)} patterns -> {args.out}")
    print("by type:", dict(Counter(r["transformation_type"] for r in rows)))
    print("by dataset:", dict(Counter(r["source_dataset"] for r in rows)))
    pc = Counter(r["pattern"] for r in rows)
    print("patterns (max share %.3f):" % (max(pc.values()) / len(rows)), dict(pc))


if __name__ == "__main__":
    main()
