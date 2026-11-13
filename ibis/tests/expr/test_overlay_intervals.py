from __future__ import annotations

import pytest

import ibis
import ibis.common.exceptions as com
import ibis.expr.datatypes as dt
import ibis.expr.operations as ops


def _interval_table(**extra):
    schema = {
        "pid": "int64",
        "source": "string",
        "start": "int64",
        "end": "int64",
        **extra,
    }
    return ibis.table(schema, name="t")


def test_overlay_output_schema_is_fixed_from_declared_sources():
    t = _interval_table()
    expr = t.overlay_intervals(
        start="start",
        end="end",
        source="source",
        partition_by="pid",
        sources=["a", "b"],
    )
    assert expr.schema() == ibis.schema(
        {
            "pid": dt.int64,
            "window_start": dt.int64,
            "window_end": dt.int64,
            "a": dt.int64,
            "b": dt.int64,
        }
    )


def test_overlay_schema_timestamps_other_bucket_and_prefix():
    t = ibis.table(
        {
            "pid": "int64",
            "source": "string",
            "start": "timestamp",
            "end": "timestamp",
        },
        name="t",
    )
    expr = t.overlay_intervals(
        start="start",
        end="end",
        source="source",
        partition_by="pid",
        sources=["a"],
        other="rest",
        names_prefix="n_",
    )
    assert expr.schema() == ibis.schema(
        {
            "pid": dt.int64,
            "window_start": dt.timestamp,
            "window_end": dt.timestamp,
            "n_a": dt.int64,
            "n_rest": dt.int64,
        }
    )


def test_overlay_single_source_mode_schema():
    t = ibis.table({"start": "date", "end": "date"}, name="t")
    expr = t.overlay_intervals(start="start", end="end")
    assert expr.schema() == ibis.schema(
        {"window_start": dt.date, "window_end": dt.date, "n": dt.int64}
    )


def test_overlay_rejects_incomparable_endpoint_dtypes():
    t = ibis.table({"start": "string", "end": "string", "source": "string"}, name="t")
    with pytest.raises(com.IbisTypeError, match="numeric or temporal"):
        t.overlay_intervals(start="start", end="end")

    mismatched = ibis.table(
        {"start": "int64", "end": "timestamp", "source": "string"}, name="t"
    )
    with pytest.raises(com.IbisTypeError, match="same dtype"):
        mismatched.overlay_intervals(start="start", end="end")


def test_overlay_eager_argument_validation():
    t = _interval_table()

    with pytest.raises(com.IbisInputError, match="null_endpoint"):
        t.overlay_intervals(start="start", end="end", null_endpoint="explode")

    with pytest.raises(com.IbisInputError, match="together with a `source`"):
        t.overlay_intervals(start="start", end="end", sources=["a"])

    with pytest.raises(com.IbisInputError, match="together with a `source`"):
        t.overlay_intervals(start="start", end="end", other="rest")

    with pytest.raises(com.IbisInputError, match="explicit `sources`"):
        t.overlay_intervals(start="start", end="end", source="source", other="rest")

    with pytest.raises(com.IbisInputError, match="unique"):
        t.overlay_intervals(
            start="start",
            end="end",
            source="source",
            sources=["a", "a"],
        )

    with pytest.raises(com.IbisInputError, match="at least one"):
        t.overlay_intervals(start="start", end="end", source="source", sources=[])


def test_overlay_rejects_output_column_collisions():
    t = _interval_table(window_start="int64")
    with pytest.raises(com.IbisInputError, match="collide"):
        t.overlay_intervals(
            start="start",
            end="end",
            source="source",
            partition_by="pid",
            sources=["window_start"],
        )

    internal = _interval_table(_overlay_ts="int64")
    with pytest.raises(com.IbisInputError, match="reserved internal names"):
        internal.overlay_intervals(
            start="start", end="end", source="source", sources=["a"]
        )


def test_overlay_builds_real_relation_nodes_without_joins():
    t = _interval_table()
    expr = t.overlay_intervals(
        start="start",
        end="end",
        source="source",
        partition_by="pid",
        sources=["a", "b"],
    )
    nodes = list(expr.op().find(ops.Node))
    names = {type(node).__name__ for node in nodes}

    assert "Union" in names  # endpoint UNION ALL
    assert "Aggregate" in names  # same-timestamp merge + pivot
    assert "WindowFunction" in names  # cumulative ROWS sums
    assert "Lead" in names  # slice boundaries
    assert "Filter" in names  # trailing endpoint removal
    assert not any("Join" in name for name in names)

    # exactly one UNION ALL of the two endpoint relations
    unions = list(expr.op().find(ops.Union))
    assert len(unions) == 1
    assert unions[0].distinct is False


def test_overlay_cumulative_window_uses_explicit_rows_frame():
    t = _interval_table()
    expr = t.overlay_intervals(
        start="start",
        end="end",
        source="source",
        partition_by="pid",
        sources=["a", "b"],
    )
    window_fns = list(expr.op().find(ops.WindowFunction))
    cumulative = [node for node in window_fns if isinstance(node.func, ops.Sum)]
    # one cumulative sum window per declared source
    assert len(cumulative) == 2
    for node in cumulative:
        assert node.how == "rows"
        # unbounded preceding up to and including the current row
        assert node.start is None
        assert node.end is not None and node.end.preceding is False
        assert node.end.value.value == 0
        assert len(node.group_by) == 1
        assert len(node.order_by) == 1


def test_overlay_merge_adds_lag_based_island_window():
    t = _interval_table()
    expr = t.overlay_intervals(
        start="start",
        end="end",
        source="source",
        partition_by="pid",
        sources=["a", "b"],
        merge=True,
    )
    names = {type(node).__name__ for node in expr.op().find(ops.Node)}
    assert "Lag" in names
    assert "WindowFunction" in names
    # the final grouping collapses islands instead of emitting raw slices
    assert "Aggregate" in names


def test_overlay_compiles_on_duckdb_and_postgres_dialects():
    t = _interval_table()
    expr = t.overlay_intervals(
        start="start",
        end="end",
        source="source",
        partition_by="pid",
        sources=["a", "b"],
    )
    for dialect in ("duckdb", "postgres"):
        sql = str(ibis.to_sql(expr, dialect=dialect))
        upper = sql.upper()
        assert "UNION ALL" in upper
        assert "CROSS JOIN" not in upper
        assert "ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW" in upper
