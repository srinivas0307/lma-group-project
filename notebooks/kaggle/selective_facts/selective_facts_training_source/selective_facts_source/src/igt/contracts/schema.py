"""Intent-contract schema, vocabulary v1.0 (docs/eval_set_and_contract_guide.md §5).

`load_contract(dict)` validates a contract (gold or parsed) and returns a `Contract`; it raises
`ContractError` for anything outside the vocabulary. `Contract.model_json_schema()` gives the JSON schema
(usable for constrained decoding by the parsers).
"""
from typing import Annotated, Any, Literal, Union

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

VOCABULARY_VERSION = "1.0"

COUNT_OPS_RELATIVE = ("unchanged", "decreased", "increased")
COUNT_OPS_ABSOLUTE = ("eq", "lt", "le", "gt", "ge")
COMPARE_OPS = ("==", "!=", "<", "<=", ">", ">=")
DTYPE_FAMILIES = ("int", "float", "bool", "string", "category", "datetime")
FUNCTIONS = ("identity", "const", "null", "add", "sub", "mul", "div", "round", "clip", "lower", "upper", "strip",
             "map", "cast")
FN_ARITY = {"identity": 1, "const": 1, "null": 0, "add": 2, "sub": 2, "mul": 2, "div": 2, "round": 2, "clip": 3,
            "lower": 1, "upper": 1, "strip": 1, "map": 2, "cast": (2, 3)}


class ContractError(ValueError):
    pass


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)


# ---------------- conditions (§5.3) ----------------
class Condition(_Strict):
    """Either a leaf {column, op, value} or one combinator: all / any / not."""
    column: str | None = None
    op: Literal["==", "!=", "<", "<=", ">", ">=", "in", "not_in", "is_null", "not_null"] | None = None
    value: Any = None
    all: list["Condition"] | None = None
    any: list["Condition"] | None = None
    not_: "Condition | None" = Field(default=None, alias="not")

    @model_validator(mode="after")
    def _one_shape(self):
        leaf = self.column is not None or self.op is not None
        shapes = [leaf, self.all is not None, self.any is not None, self.not_ is not None]
        if sum(shapes) != 1:
            raise ValueError("a condition is exactly one of: leaf {column, op, value}, all, any, not")
        if leaf:
            if self.column is None or self.op is None:
                raise ValueError("a leaf condition needs column and op")
            if self.op in ("is_null", "not_null"):
                if self.value is not None:
                    raise ValueError(f"op {self.op} takes no value")
            elif self.op in ("in", "not_in"):
                if not isinstance(self.value, list):
                    raise ValueError(f"op {self.op} needs a list value")
            elif self.value is None or isinstance(self.value, (list, dict)):
                raise ValueError(f"op {self.op} needs a scalar value")
        for kids in (self.all, self.any):
            if kids is not None and len(kids) < 1:
                raise ValueError("all/any need at least one child")
        return self

    def columns(self) -> list[str]:
        if self.column is not None:
            return [self.column]
        kids = self.all or self.any or ([self.not_] if self.not_ else [])
        return [c for k in kids for c in k.columns()]


# ---------------- relation arguments (§5.6) ----------------
class ColArg(_Strict):
    col: str


class LitArg(_Strict):
    lit: Any = None


Arg = Union[ColArg, LitArg]


# ---------------- predicates (§5.5) ----------------
class _Pred(_Strict):
    id: str | None = None


class _Counted(_Pred):
    op: Literal["unchanged", "decreased", "increased", "eq", "lt", "le", "gt", "ge"]
    value: int | None = None

    @model_validator(mode="after")
    def _op_value(self):
        if self.op in COUNT_OPS_RELATIVE and self.value is not None:
            raise ValueError(f"op {self.op} is relative and takes no value")
        if self.op in COUNT_OPS_ABSOLUTE and self.value is None:
            raise ValueError(f"op {self.op} needs a value")
        return self


ColumnSet = Union[Literal["*"], str, list[str]]


class RowCount(_Counted):
    type: Literal["row_count"]
    target: None = None


class RowsRemovedWhere(_Pred):
    type: Literal["rows_removed_where"]
    target: None = None
    condition: Condition


class RowsRetainedWhere(_Pred):
    type: Literal["rows_retained_where"]
    target: None = None
    condition: Condition


class DuplicateCount(_Counted):
    type: Literal["duplicate_count"]
    target: ColumnSet
    except_: list[str] = Field(default_factory=list, alias="except")
    keep: Literal["first", "last"] | None = None

    @model_validator(mode="after")
    def _keep_rule(self):
        if self.keep is not None and not (self.op == "eq" and self.value == 0):
            raise ValueError("keep is only allowed with op: eq, value: 0")
        return self


class NullCount(_Counted):
    type: Literal["null_count"]
    target: str


class ColumnValuesUnchanged(_Pred):
    type: Literal["column_values_unchanged"]
    target: ColumnSet
    except_: list[str] = Field(default_factory=list, alias="except")
    where: Condition | None = None


class ColumnsUnchanged(_Pred):
    type: Literal["columns_unchanged"]
    target: ColumnSet
    except_: list[str] = Field(default_factory=list, alias="except")


class ColumnDtype(_Pred):
    type: Literal["column_dtype"]
    target: str
    dtype: Literal["int", "float", "bool", "string", "category", "datetime"]
    ordered: bool | None = None
    categories: list[Any] | None = None

    @model_validator(mode="after")
    def _category_fields(self):
        if self.dtype != "category" and (self.ordered is not None or self.categories is not None):
            raise ValueError("ordered/categories only apply to dtype: category")
        return self


class ColumnValueRelation(_Pred):
    type: Literal["column_value_relation"]
    target: str
    fn: Literal["identity", "const", "null", "add", "sub", "mul", "div", "round", "clip", "lower", "upper", "strip",
                "map", "cast"]
    args: list[Arg] = Field(default_factory=list)
    where: Condition | None = None
    atol: float | None = None

    @model_validator(mode="after")
    def _arity(self):
        want = FN_ARITY[self.fn]
        ok = len(self.args) in want if isinstance(want, tuple) else len(self.args) == want
        if not ok:
            raise ValueError(f"fn {self.fn} takes {want} args, got {len(self.args)}")
        lit_only = {"const": [0], "round": [1], "clip": [1, 2], "map": [1], "cast": [1, 2]}
        col_first = {"identity", "round", "clip", "lower", "upper", "strip", "map", "cast"}
        if self.fn in col_first and not isinstance(self.args[0], ColArg):
            raise ValueError(f"fn {self.fn}: first argument must be a column")
        for i in lit_only.get(self.fn, []):
            if i < len(self.args) and not isinstance(self.args[i], LitArg):
                raise ValueError(f"fn {self.fn}: argument {i + 1} must be a literal")
        if self.fn == "map" and not isinstance(self.args[1].lit, dict):
            raise ValueError("fn map: second argument must be a literal mapping")
        if self.fn == "cast" and self.args[1].lit not in DTYPE_FAMILIES:
            raise ValueError(f"fn cast: dtype must be one of {DTYPE_FAMILIES}")
        return self


class ColumnsAdded(_Pred):
    type: Literal["columns_added"]
    target: list[str]


class ColumnsRemoved(_Pred):
    type: Literal["columns_removed"]
    target: list[str]


Predicate = Annotated[Union[RowCount, RowsRemovedWhere, RowsRetainedWhere, DuplicateCount, NullCount,
                            ColumnValuesUnchanged, ColumnsUnchanged, ColumnDtype, ColumnValueRelation,
                            ColumnsAdded, ColumnsRemoved], Field(discriminator="type")]
PREDICATE_TYPES = ("row_count", "rows_removed_where", "rows_retained_where", "duplicate_count", "null_count",
                   "column_values_unchanged", "columns_unchanged", "column_dtype", "column_value_relation",
                   "columns_added", "columns_removed")


class Contract(_Strict):
    contract_version: Literal["1.0"] = "1.0"
    must_change: list[Predicate] = Field(default_factory=list)
    must_preserve: list[Predicate] = Field(default_factory=list)

    def predicates(self, directions=("must_change", "must_preserve")):
        """(direction, predicate) pairs; gold ids are kept, missing ids become c1.. / p1.. by position."""
        out = []
        for d in directions:
            for i, p in enumerate(getattr(self, d), 1):
                out.append((d, p.id or f"{'c' if d == 'must_change' else 'p'}{i}", p))
        return out


def load_contract(data: dict) -> Contract:
    try:
        return Contract.model_validate(data)
    except ValidationError as e:
        msgs = "; ".join(f"{'.'.join(map(str, err['loc']))}: {err['msg']}" for err in e.errors()[:5])
        raise ContractError(f"invalid contract: {msgs}") from None


# ---------------- helpers ----------------
def expand_columns(target, except_, before_columns) -> list[str]:
    """'*' -> all before columns; a string -> [string]; minus `except`."""
    cols = list(before_columns) if target == "*" else [target] if isinstance(target, str) else list(target)
    return [c for c in cols if c not in set(except_ or [])]


def predicate_columns(p, before_columns) -> list[str]:
    """Columns a predicate expects to exist in the BEFORE data (for the unknown_column check)."""
    cols = []
    if getattr(p, "condition", None) is not None:
        cols += p.condition.columns()
    if getattr(p, "where", None) is not None:
        cols += p.where.columns()
    if p.type in ("duplicate_count", "column_values_unchanged", "columns_unchanged"):
        # Names in `except` only remove columns from the expansion; a name that isn't in the input (e.g. a column the
        # transformation adds) removes nothing, so it is not a reference to an unknown column.
        cols += expand_columns(p.target, p.except_, before_columns)
    elif p.type in ("null_count", "column_dtype"):
        cols.append(p.target)
    elif p.type == "column_value_relation":
        cols += [a.col for a in p.args if isinstance(a, ColArg)]       # target may be a new column
    elif p.type == "columns_removed":
        cols += p.target
    return list(dict.fromkeys(cols))
