"""Interval overlay (sweep-line) decomposition.

Given a finite relation of half-open ``[start, end)`` intervals, optionally
tagged with a ``source`` and partitioned by explicit keys, build the minimal
set of non-empty time slices induced by every valid endpoint. For each slice
the number of covering input intervals is reported per declared source.

The decomposition is compiled entirely from real relation nodes:

1. start/end endpoint events stacked with ``UNION ALL`` (``2 * n`` rows),
2. same-timestamp events merged with a ``GROUP BY`` *before* any prefix sum,
3. per-source delta columns pivoted with conditional aggregations,
4. running coverage counts computed over explicit ``ROWS`` windows,
5. slice boundaries produced with a partitioned ``LEAD`` window.

Every intermediate relation stays linear in the number of distinct
endpoints (``O(n)``): neither the full table is pulled back into Python nor
is any cross join over interval pairs generated.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

import ibis
import ibis.common.exceptions as com
from ibis import _

if TYPE_CHECKING:
    from collections.abc import Iterable

    import ibis.expr.types as ir

#: Fixed output column names for the slice boundaries.
WINDOW_START = "window_start"
WINDOW_END = "window_end"
#: Fixed output column name in source-less (single coverage) mode.
COUNT_NAME = "n"

_TS = "_overlay_ts"
_DELTA = "_overlay_delta"
_SOURCE = "_overlay_source"
_GUARD = "_overlay_guard"
_NEXT_TS = "_overlay_next_ts"
_CHANGED = "_overlay_changed"
_ISLAND = "_overlay_island"
_DELTA_PREFIX = "_overlay_delta_"

_INTERNAL_NAMES = frozenset({_TS, _DELTA, _SOURCE, _GUARD, _NEXT_TS, _CHANGED, _ISLAND})


def _resolve_partitions(
    table: ir.Table, partition_by: Any
) -> tuple[tuple[ir.Value, ...], tuple[str, ...]]:
    partition_by = ibis.util.promote_list(partition_by)
    cols = table.bind(*partition_by) if partition_by else ()
    names = tuple(col.get_name() for col in cols)
    if len(set(names)) != len(names):
        raise com.IbisInputError(
            f"Duplicate partition column names are not allowed: {names}"
        )
    return cols, names


def _validate_endpoints(start_col: ir.Value, end_col: ir.Value) -> None:
    start_dtype = start_col.type()
    end_dtype = end_col.type()

    for name, dtype in (("start", start_dtype), ("end", end_dtype)):
        if not (dtype.is_numeric() or dtype.is_temporal()):
            raise com.IbisTypeError(
                f"`{name}` must have a numeric or temporal dtype to be used "
                f"as an interval endpoint, got {dtype}"
            )

    if start_dtype != end_dtype:
        raise com.IbisTypeError(
            "`start` and `end` must have the same dtype, got "
            f"{start_dtype} and {end_dtype}; cast one of them explicitly"
        )


def _resolve_source_domain(
    table: ir.Table,
    source_col: ir.Value,
    sources: Iterable[Any] | None,
) -> tuple[list[Any], list[ir.Scalar]]:
    """Return the declared source values and matching literal expressions.

    The output schema is fixed at construction time: when `sources` is not
    provided the domain is discovered eagerly (mirroring `pivot_wider`) and
    the resulting column set stays constant for the returned expression.
    """
    source_dtype = source_col.type()

    if sources is None:
        observed = table.select(source_col).distinct().execute()
        raw_values = list(observed.iloc[:, 0])
        if any(value is None for value in raw_values):
            raise com.IbisInputError(
                "`source` contains NULL values but no source domain was "
                "declared; pass `sources` (and optionally `other`) or filter "
                "out NULL source rows"
            )
        values = raw_values
    else:
        values = list(sources)

    if not values:
        raise com.IbisInputError("`sources` must contain at least one source value")

    if len({repr(value) for value in values}) != len(values):
        raise com.IbisInputError(f"`sources` must be unique, got: {values!r}")

    literals = [ibis.literal(value, type=source_dtype) for value in values]
    return values, literals


def _count_column_names(
    values: list[Any],
    names_prefix: str,
    reserved: set[str],
) -> list[str]:
    names = [f"{names_prefix}{value}" for value in values]

    if len(set(names)) != len(names):
        raise com.IbisInputError(
            f"Source values produce duplicate count column names: {names}"
        )

    collisions = reserved.intersection(names)
    if collisions:
        raise com.IbisInputError(
            "Count column names collide with partition or boundary columns: "
            f"{sorted(collisions)}; use `names_prefix` to disambiguate"
        )
    return names


def _build_guard(
    *,
    start_col: ir.Value,
    end_col: ir.Value,
    source_col: ir.Value | None,
    source_literals: list[ir.Scalar] | None,
    other: str | None,
    null_endpoint: str,
) -> ir.Value:
    """Build a per-row expression that fails on invalid input rows.

    Invalid rows raise at execution time via a cast error carrying a
    descriptive message instead of being silently dropped.
    """
    branches = []

    if source_col is not None and other is None:
        branches.append(
            (
                source_col.isnull() | source_col.notin(source_literals),
                ibis.literal(
                    "overlay_intervals: source value is not part of the "
                    "declared `sources` domain"
                ).cast("int64"),
            )
        )

    if null_endpoint == "error":
        branches.append(
            (
                start_col.isnull() | end_col.isnull(),
                ibis.literal(
                    "overlay_intervals: NULL `start`/`end` is not allowed; "
                    "pass null_endpoint='drop' to exclude such rows"
                ).cast("int64"),
            )
        )

    branches.append(
        (
            start_col > end_col,
            ibis.literal(
                "overlay_intervals: `start` must be less than or equal to `end`"
            ).cast("int64"),
        )
    )

    return ibis.cases(*branches, else_=0)


def overlay_intervals(
    table: ir.Table,
    *,
    start: str | ir.Value,
    end: str | ir.Value,
    source: str | ir.Value | None = None,
    partition_by: Any = (),
    sources: Iterable[Any] | None = None,
    other: str | None = None,
    complete_only: bool = False,
    null_endpoint: Literal["error", "drop"] = "error",
    merge: bool = False,
    names_prefix: str = "",
) -> ir.Table:
    """Split half-open intervals into minimal non-empty coverage slices.

    The input relation is treated as a multiset of half-open ``[start, end)``
    intervals. Every distinct endpoint becomes a slice boundary; within each
    slice the running number of covering intervals is reported per source as
    a fixed set of integer count columns.

    Parameters
    ----------
    table
        The input table expression.
    start
        Column (name or expression) holding the inclusive interval start.
    end
        Column (name or expression) holding the exclusive interval end.
        Must share the same numeric or temporal dtype as `start`.
    source
        Optional column (name or expression) holding the source label. When
        omitted the expression runs in single-coverage mode and emits one
        count column named ``n``.
    partition_by
        Explicit partition keys. Windows and endpoint events are scoped to
        each partition; NULL keys form their own partition.
    sources
        Explicit ordered domain of `source` values defining the fixed output
        count columns. When omitted the domain is discovered eagerly with a
        ``DISTINCT`` query (this breaks laziness, like `pivot_wider`).
    other
        When set to a string, source values outside the declared `sources`
        domain (including NULL) are counted in an extra column of that name
        instead of raising. When unset, undeclared source values raise at
        execution time.
    complete_only
        When true, only slices covered by every declared source (strictly
        positive count for each) are returned. The `other` bucket is not
        considered part of the declared domain.
    null_endpoint
        Explicit policy for NULL endpoints: ``"error"`` (default) raises at
        execution time, ``"drop"`` excludes rows with a NULL `start` or
        `end` before events are generated.
    merge
        When false (default) every endpoint boundary is preserved, even when
        adjacent slices share identical counts. When true, adjacent slices
        with identical count vectors are merged via a gaps-and-islands
        grouping; slices separated by a gap are never merged.
    names_prefix
        Prefix added to every per-source count column name.

    Returns
    -------
    Table
        A table with the partition columns, ``window_start`` / ``window_end``
        boundary columns (same dtype as the endpoints) and one ``int64``
        count column per declared source (plus the optional `other` bucket),
        or a single ``n`` column in source-less mode. The output schema is
        fixed when the relation is built.
    """
    if null_endpoint not in ("error", "drop"):
        raise com.IbisInputError(
            f"`null_endpoint` must be either 'error' or 'drop', got {null_endpoint!r}"
        )

    start_col, end_col = table.bind(start, end)
    _validate_endpoints(start_col, end_col)

    part_cols, part_names = _resolve_partitions(table, partition_by)

    input_names = set(table.columns)
    internal_collisions = input_names.intersection(_INTERNAL_NAMES)
    if internal_collisions:
        raise com.IbisInputError(
            f"Input columns {sorted(internal_collisions)} use reserved "
            "internal names of overlay_intervals; rename them first"
        )

    reserved = set(part_names) | {WINDOW_START, WINDOW_END}

    multi_source = source is not None
    if multi_source:
        (source_col,) = table.bind(source)
        if other is not None and not isinstance(other, str):
            raise com.IbisInputError(
                f"`other` must be a column name string, got {other!r}"
            )
        if sources is None and other is not None:
            raise com.IbisInputError("`other` requires an explicit `sources` domain")
        values, source_literals = _resolve_source_domain(table, source_col, sources)
        count_names = _count_column_names(values, names_prefix, reserved)
        if other is not None:
            other_name = f"{names_prefix}{other}"
            if other_name in reserved or other_name in count_names:
                raise com.IbisInputError(
                    f"`other` column name {other_name!r} collides with an "
                    "existing output column"
                )
            count_names.append(other_name)
        declared_count_names = count_names[: len(values)]
    else:
        if sources is not None:
            raise com.IbisInputError(
                "`sources` can only be provided together with a `source` column"
            )
        if other is not None:
            raise com.IbisInputError(
                "`other` can only be provided together with a `source` column"
            )
        source_col = None
        source_literals = None
        values = []
        count_names = [f"{names_prefix}{COUNT_NAME}"]
        if count_names[0] in reserved:
            raise com.IbisInputError(
                f"Count column name {count_names[0]!r} collides with a "
                "partition or boundary column; use `names_prefix`"
            )
        declared_count_names = list(count_names)

    # Step 1: optionally remove invalid NULL endpoint rows (explicit policy)
    base = table
    if null_endpoint == "drop":
        base = base.filter(start_col.notnull() & end_col.notnull())

    guard = _build_guard(
        start_col=start_col,
        end_col=end_col,
        source_col=source_col,
        source_literals=source_literals,
        other=other,
        null_endpoint=null_endpoint,
    )

    # Step 2: endpoint events via UNION ALL (two rows per input interval)
    one = ibis.literal(1, type="int64")
    minus_one = ibis.literal(-1, type="int64")

    def _events(ts_col: ir.Value, delta: ir.Scalar) -> ir.Table:
        cols = [*part_cols, ts_col.name(_TS)]
        if multi_source:
            cols.append(source_col.name(_SOURCE))
        cols.append(delta.name(_DELTA))
        if guard is not None:
            cols.append(guard.name(_GUARD))
        return base.select(*cols)

    events = _events(start_col, one).union(_events(end_col, minus_one))

    # Step 3: merge same-timestamp events BEFORE updating prefix counts
    merge_keys = [*part_names, _TS]
    if multi_source:
        merge_keys.append(_SOURCE)

    grouped = events.group_by(*merge_keys)
    if guard is not None:
        # the HAVING clause forces evaluation of the failing cast on every row
        grouped = grouped.having(_[_GUARD].max() == 0)
    merged = grouped.aggregate(**{_DELTA: _[_DELTA].sum()})

    # Step 4: pivot the per-source deltas into a fixed, stable schema
    delta_pairs: list[tuple[str, str]] = []
    delta_aggs: dict[str, ir.Value] = {}

    if multi_source:
        for index, (name, literal) in enumerate(
            zip(declared_count_names, source_literals)
        ):
            delta_name = f"{_DELTA_PREFIX}{index}"
            delta_pairs.append((delta_name, name))
            delta_aggs[delta_name] = ibis.coalesce(
                _[_DELTA].sum(where=_[_SOURCE] == literal), 0
            )

        if other is not None:
            delta_name = f"{_DELTA_PREFIX}other"
            delta_pairs.append((delta_name, other_name))
            delta_aggs[delta_name] = ibis.coalesce(
                _[_DELTA].sum(
                    where=_[_SOURCE].isnull() | _[_SOURCE].notin(source_literals)
                ),
                0,
            )
    else:
        delta_name = f"{_DELTA_PREFIX}0"
        delta_pairs.append((delta_name, count_names[0]))
        delta_aggs[delta_name] = _[_DELTA].sum()

    wide = merged.group_by(*part_names, _TS).aggregate(**delta_aggs)

    # Step 5: running coverage counts over explicit ROWS windows
    cumulative_window = ibis.window(
        rows=(None, 0),
        group_by=part_names or None,
        order_by=_TS,
    )
    running = wide.mutate(
        **{
            count_name: _[delta_name].sum().over(cumulative_window)
            for delta_name, count_name in delta_pairs
        }
    )

    # Step 6: lead() turns distinct endpoints into half-open slices
    lead_window = ibis.window(
        group_by=part_names or None,
        order_by=_TS,
    )
    with_next = running.mutate(**{_NEXT_TS: _[_TS].lead().over(lead_window)})
    # the last endpoint per partition has no successor and yields no slice;
    # distinct timestamps guarantee every remaining slice has positive width
    sliced = with_next.filter(_[_NEXT_TS].notnull())

    result = sliced.select(
        *part_names,
        _[_TS].name(WINDOW_START),
        _[_NEXT_TS].name(WINDOW_END),
        *count_names,
    )

    # Step 7 (optional): keep only fully covered slices
    if complete_only:
        result = result.filter(
            ibis.and_(*(_[name] > 0 for name in declared_count_names))
        )

    # Step 8 (optional): merge adjacent identical slices (gaps & islands)
    if merge:
        lag_window = ibis.window(
            group_by=part_names or None,
            order_by=WINDOW_START,
        )
        change_conditions = [
            ibis.coalesce(_[name] != _[name].lag().over(lag_window), True)
            for name in count_names
        ]
        # do not merge across gaps left by complete_only filtering
        change_conditions.append(
            ibis.coalesce(_[WINDOW_END].lag().over(lag_window) != _[WINDOW_START], True)
        )
        with_changed = result.mutate(
            **{_CHANGED: ibis.or_(*change_conditions).cast("int64")}
        )
        island_window = ibis.window(
            rows=(None, 0),
            group_by=part_names or None,
            order_by=WINDOW_START,
        )
        with_island = with_changed.mutate(
            **{_ISLAND: _[_CHANGED].sum().over(island_window)}
        )
        result = (
            with_island.group_by(*part_names, _ISLAND)
            .aggregate(
                **{WINDOW_START: _[WINDOW_START].min()},
                **{WINDOW_END: _[WINDOW_END].max()},
                **{name: _[name].max() for name in count_names},
            )
            .drop(_ISLAND)
        )

    return result.order_by(*part_names, WINDOW_START)
