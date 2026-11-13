from __future__ import annotations

import itertools
import random
from collections import defaultdict

import pytest

import ibis
import ibis.common.exceptions as com
from ibis import _

pd = pytest.importorskip("pandas")
sg = pytest.importorskip("sqlglot")

# Generic SQL (UNION ALL, GROUP BY, ROWS windows, LEAD) with no backend
# specific rewrites; only DuckDB and PostgreSQL are validated initially.
pytestmark = pytest.mark.notyet(
    [
        "athena",
        "bigquery",
        "clickhouse",
        "databricks",
        "datafusion",
        "druid",
        "exasol",
        "flink",
        "impala",
        "materialize",
        "mssql",
        "mysql",
        "oracle",
        "polars",
        "pyspark",
        "risingwave",
        "singlestoredb",
        "snowflake",
        "sqlite",
        "trino",
    ],
    reason="overlay_intervals is only validated on DuckDB and PostgreSQL initially",
)

SOURCES = ["a", "b", "c"]


# ---------------------------------------------------------------------------
# Pure-Python sweep-line reference, mirroring the SQL semantics:
# same-timestamp start/end events merge into one delta before prefix counts.
# ---------------------------------------------------------------------------
def sweep_line(rows, *, sources, partition_names, has_source):
    by_partition = defaultdict(list)
    for row in rows:
        key = tuple(row[name] for name in partition_names)
        by_partition[key].append(row)

    bucket_names = sources if has_source else ["n"]
    expected = []
    for key, partition_rows in by_partition.items():
        events = defaultdict(lambda: defaultdict(int))
        for row in partition_rows:
            if row["start"] is None or row["end"] is None:
                continue
            bucket = row["source"] if has_source else "n"
            events[row["start"]][bucket] += 1
            events[row["end"]][bucket] -= 1

        timestamps = sorted(events)
        running = defaultdict(int)
        for ts, next_ts in itertools.pairwise(timestamps):
            for bucket, delta in events[ts].items():
                running[bucket] += delta
            record = dict(zip(partition_names, key))
            record["window_start"] = ts
            record["window_end"] = next_ts
            for bucket in bucket_names:
                record[bucket] = running.get(bucket, 0)
            expected.append(record)
    return expected


def _normalized(df, columns):
    return sorted(map(tuple, df[columns].itertuples(index=False, name=None)))


def _assert_matches(result_df, expected, columns):
    expected_df = pd.DataFrame(expected, columns=columns)
    assert _normalized(result_df, columns) == _normalized(expected_df, columns)


# ---------------------------------------------------------------------------
# Basic semantics
# ---------------------------------------------------------------------------
def test_overlay_basic(con):
    t = ibis.memtable(
        {
            "pid": [1, 1, 1],
            "source": ["a", "b", "a"],
            "start": [1, 2, 5],
            "end": [4, 6, 8],
        }
    )
    expr = t.overlay_intervals(
        start="start",
        end="end",
        source="source",
        partition_by="pid",
        sources=SOURCES[:2],
    )
    assert list(expr.schema().names) == [
        "pid",
        "window_start",
        "window_end",
        "a",
        "b",
    ]
    assert expr.schema()["a"].is_int64()
    assert expr.schema()["window_start"].is_int64()

    result = con.execute(expr)
    expected = sweep_line(
        [
            {"pid": 1, "source": "a", "start": 1, "end": 4},
            {"pid": 1, "source": "b", "start": 2, "end": 6},
            {"pid": 1, "source": "a", "start": 5, "end": 8},
        ],
        sources=["a", "b"],
        partition_names=["pid"],
        has_source=True,
    )
    _assert_matches(result, expected, ["pid", "window_start", "window_end", "a", "b"])


def test_overlay_nested_intervals(con):
    # fully nested intervals of the same source accumulate coverage
    t = ibis.memtable(
        {
            "source": ["a", "a", "a"],
            "start": [1, 2, 3],
            "end": [10, 9, 8],
        }
    )
    result = con.execute(
        t.overlay_intervals(start="start", end="end", source="source", sources=["a"])
    )
    expected = [
        {"window_start": 1, "window_end": 2, "a": 1},
        {"window_start": 2, "window_end": 3, "a": 2},
        {"window_start": 3, "window_end": 8, "a": 3},
        {"window_start": 8, "window_end": 9, "a": 2},
        {"window_start": 9, "window_end": 10, "a": 1},
    ]
    _assert_matches(result, expected, ["window_start", "window_end", "a"])


def test_overlay_handoff_preserves_cut_points(con):
    # touching half-open intervals: [1, 5) and [5, 10)
    t = ibis.memtable({"source": ["a", "a"], "start": [1, 5], "end": [5, 10]})
    result = con.execute(
        t.overlay_intervals(start="start", end="end", source="source", sources=["a"])
    )
    expected = [
        {"window_start": 1, "window_end": 5, "a": 1},
        {"window_start": 5, "window_end": 10, "a": 1},
    ]
    _assert_matches(result, expected, ["window_start", "window_end", "a"])

    merged = con.execute(
        t.overlay_intervals(
            start="start",
            end="end",
            source="source",
            sources=["a"],
            merge=True,
        )
    )
    merged_expected = [{"window_start": 1, "window_end": 10, "a": 1}]
    _assert_matches(merged, merged_expected, ["window_start", "window_end", "a"])


def test_overlay_duplicate_intervals_are_a_bag(con):
    # the same interval twice must count twice (no set deduplication)
    t = ibis.memtable(
        {
            "source": ["a", "a", "b"],
            "start": [1, 1, 1],
            "end": [5, 5, 5],
        }
    )
    result = con.execute(
        t.overlay_intervals(
            start="start", end="end", source="source", sources=["a", "b"]
        )
    )
    expected = [{"window_start": 1, "window_end": 5, "a": 2, "b": 1}]
    _assert_matches(result, expected, ["window_start", "window_end", "a", "b"])


def test_overlay_same_point_start_and_end_events_merge_first(con):
    # at t=5 one source ends and another starts: events merge before counts
    t = ibis.memtable(
        {
            "source": ["a", "b"],
            "start": [1, 5],
            "end": [5, 9],
        }
    )
    result = con.execute(
        t.overlay_intervals(
            start="start", end="end", source="source", sources=["a", "b"]
        )
    )
    expected = [
        {"window_start": 1, "window_end": 5, "a": 1, "b": 0},
        {"window_start": 5, "window_end": 9, "a": 0, "b": 1},
    ]
    _assert_matches(result, expected, ["window_start", "window_end", "a", "b"])

    # zero-width interval [t, t) cancels itself and creates no slice
    zero_width = ibis.memtable(
        {
            "source": ["a", "a", "b"],
            "start": [1, 3, 3],
            "end": [5, 3, 7],
        }
    )
    result = con.execute(
        zero_width.overlay_intervals(
            start="start", end="end", source="source", sources=["a", "b"]
        )
    )
    assert all(row.window_start < row.window_end for row in result.itertuples())


def test_overlay_single_source_mode(con):
    # no source column: single `n` column, structurally distinct from 0 counts
    t = ibis.memtable({"start": [1, 3], "end": [5, 7]})
    expr = t.overlay_intervals(start="start", end="end")
    assert list(expr.schema().names) == ["window_start", "window_end", "n"]
    result = con.execute(expr)
    expected = [
        {"window_start": 1, "window_end": 3, "n": 1},
        {"window_start": 3, "window_end": 5, "n": 2},
        {"window_start": 5, "window_end": 7, "n": 1},
    ]
    _assert_matches(result, expected, ["window_start", "window_end", "n"])


def test_overlay_gap_has_zero_counts_without_merging(con):
    # disjoint intervals leave an uncovered middle slice reported explicitly
    t = ibis.memtable({"source": ["a", "a"], "start": [1, 6], "end": [3, 8]})
    result = con.execute(
        t.overlay_intervals(start="start", end="end", source="source", sources=["a"])
    )
    expected = [
        {"window_start": 1, "window_end": 3, "a": 1},
        {"window_start": 3, "window_end": 6, "a": 0},
        {"window_start": 6, "window_end": 8, "a": 1},
    ]
    _assert_matches(result, expected, ["window_start", "window_end", "a"])

    complete = con.execute(
        t.overlay_intervals(
            start="start",
            end="end",
            source="source",
            sources=["a"],
            complete_only=True,
        )
    )
    complete_expected = [
        {"window_start": 1, "window_end": 3, "a": 1},
        {"window_start": 6, "window_end": 8, "a": 1},
    ]
    _assert_matches(complete, complete_expected, ["window_start", "window_end", "a"])

    # merge must not bridge the removed gap
    merged = con.execute(
        t.overlay_intervals(
            start="start",
            end="end",
            source="source",
            sources=["a"],
            complete_only=True,
            merge=True,
        )
    )
    _assert_matches(merged, complete_expected, ["window_start", "window_end", "a"])


def test_overlay_merge_only_combines_adjacent_identical_slices(con):
    t = ibis.memtable(
        {
            "source": ["a", "b", "a", "a"],
            "start": [1, 3, 4, 1],
            "end": [10, 4, 5, 5],
        }
    )
    kwargs = dict(start="start", end="end", source="source", sources=["a", "b"])
    unmerged = con.execute(t.overlay_intervals(**kwargs))
    merged = con.execute(t.overlay_intervals(**kwargs, merge=True))
    # every adjacent pair here changes at least one count, so merge is a no-op
    _assert_matches(
        merged,
        sweep_line(
            [
                {"source": "a", "start": 1, "end": 10},
                {"source": "b", "start": 3, "end": 4},
                {"source": "a", "start": 4, "end": 5},
                {"source": "a", "start": 1, "end": 5},
            ],
            sources=["a", "b"],
            partition_names=[],
            has_source=True,
        ),
        ["window_start", "window_end", "a", "b"],
    )
    assert len(merged) == len(unmerged) == 4


def test_overlay_null_partition_keys_group_together(con):
    t = ibis.memtable(
        pd.DataFrame(
            {
                "pid": [None, None, 1, 1],
                "source": ["a", "a", "a", "a"],
                "start": [1, 2, 1, 2],
                "end": [3, 4, 3, 4],
            }
        ),
        schema={
            "pid": "int64",
            "source": "string",
            "start": "int64",
            "end": "int64",
        },
    )
    result = con.execute(
        t.overlay_intervals(
            start="start",
            end="end",
            source="source",
            partition_by="pid",
            sources=["a"],
        )
    )
    null_part = result[result["pid"].isna()].sort_values("window_start")
    assert null_part[["window_start", "window_end", "a"]].values.tolist() == [
        [1, 2, 1],
        [2, 3, 2],
        [3, 4, 1],
    ]
    one_part = result[result["pid"] == 1].sort_values("window_start")
    assert one_part[["window_start", "window_end", "a"]].values.tolist() == [
        [1, 2, 1],
        [2, 3, 2],
        [3, 4, 1],
    ]


def test_overlay_timestamps_and_derived_domain(con):
    df = pd.DataFrame(
        {
            "pid": [1, 1, 2],
            "source": ["a", "b", "a"],
            "start": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-02-01"]),
            "end": pd.to_datetime(["2024-01-04", "2024-01-03", "2024-02-05"]),
        }
    )
    t = ibis.memtable(df)
    # no explicit sources: domain discovered eagerly, schema still fixed
    expr = t.overlay_intervals(
        start="start",
        end="end",
        source="source",
        partition_by="pid",
    )
    assert set(expr.schema().names) == {
        "pid",
        "window_start",
        "window_end",
        "a",
        "b",
    }
    assert expr.schema()["window_start"].is_timestamp()
    result = con.execute(expr)
    expected = sweep_line(
        df.to_dict("records"),
        sources=["a", "b"],
        partition_names=["pid"],
        has_source=True,
    )
    _assert_matches(result, expected, ["pid", "window_start", "window_end", "a", "b"])


# ---------------------------------------------------------------------------
# Unknown sources / other bucket
# ---------------------------------------------------------------------------
def test_overlay_unknown_source_rejected_by_default(con):
    t = ibis.memtable({"source": ["zzz"], "start": [1], "end": [5]})
    expr = t.overlay_intervals(start="start", end="end", source="source", sources=["a"])
    with pytest.raises(Exception, match="not part of the declared"):
        con.execute(expr)


def test_overlay_other_bucket_captures_unknown_and_null(con):
    t = ibis.memtable(
        pd.DataFrame(
            {
                "source": ["a", "zzz", None],
                "start": [1, 2, 3],
                "end": [6, 5, 4],
            }
        ),
        schema={"source": "string", "start": "int64", "end": "int64"},
    )
    expr = t.overlay_intervals(
        start="start",
        end="end",
        source="source",
        sources=["a"],
        other="rest",
    )
    assert list(expr.schema().names) == [
        "window_start",
        "window_end",
        "a",
        "rest",
    ]
    result = con.execute(expr)
    expected = [
        {"window_start": 1, "window_end": 2, "a": 1, "rest": 0},
        {"window_start": 2, "window_end": 3, "a": 1, "rest": 1},
        {"window_start": 3, "window_end": 4, "a": 1, "rest": 2},
        {"window_start": 4, "window_end": 5, "a": 1, "rest": 1},
        {"window_start": 5, "window_end": 6, "a": 1, "rest": 0},
    ]
    _assert_matches(result, expected, ["window_start", "window_end", "a", "rest"])

    # `other` is not part of the declared domain for completeness
    complete = con.execute(
        t.overlay_intervals(
            start="start",
            end="end",
            source="source",
            sources=["a"],
            other="rest",
            complete_only=True,
        )
    )
    assert (complete["a"] > 0).all()


# ---------------------------------------------------------------------------
# NULL endpoints and invalid intervals
# ---------------------------------------------------------------------------
def test_overlay_null_endpoint_error_is_explicit(con):
    t = ibis.memtable(
        pd.DataFrame(
            {"source": ["a", "a"], "start": [None, 1], "end": [5, 4]},
        ),
        schema={"source": "string", "start": "int64", "end": "int64"},
    )
    with pytest.raises(Exception, match="NULL"):
        con.execute(
            t.overlay_intervals(
                start="start", end="end", source="source", sources=["a"]
            )
        )

    dropped = con.execute(
        t.overlay_intervals(
            start="start",
            end="end",
            source="source",
            sources=["a"],
            null_endpoint="drop",
        )
    )
    assert dropped[["window_start", "window_end", "a"]].values.tolist() == [[1, 4, 1]]


def test_overlay_start_greater_than_end_rejected(con):
    t = ibis.memtable({"source": ["a", "a"], "start": [1, 10], "end": [5, 4]})
    with pytest.raises(Exception, match="must be less than or equal"):
        con.execute(
            t.overlay_intervals(
                start="start", end="end", source="source", sources=["a"]
            )
        )


def test_overlay_accepts_expression_endpoints_and_partitions(con):
    t = ibis.memtable(
        {"k": [10, 10, 20], "source": ["a", "b", "a"], "x": [0, 1, 0], "y": [4, 5, 3]}
    )
    expr = t.overlay_intervals(
        start=_.x + 1,
        end=_.y + 1,
        source="source",
        partition_by=(_.k - 10).name("kp"),
        sources=["a", "b"],
    )
    result = con.execute(expr)
    expected = sweep_line(
        [
            {"kp": 0, "source": "a", "start": 1, "end": 5},
            {"kp": 0, "source": "b", "start": 2, "end": 6},
            {"kp": 10, "source": "a", "start": 1, "end": 4},
        ],
        sources=["a", "b"],
        partition_names=["kp"],
        has_source=True,
    )
    _assert_matches(result, expected, ["kp", "window_start", "window_end", "a", "b"])


def test_overlay_multiple_partition_columns(con):
    t = ibis.memtable(
        {
            "p1": [1, 1, 2],
            "p2": ["x", "x", "y"],
            "source": ["a", "a", "a"],
            "start": [1, 2, 1],
            "end": [3, 4, 3],
        }
    )
    result = con.execute(
        t.overlay_intervals(
            start="start",
            end="end",
            source="source",
            partition_by=["p1", "p2"],
            sources=["a"],
        )
    )
    assert list(result.columns) == [
        "p1",
        "p2",
        "window_start",
        "window_end",
        "a",
    ]
    assert len(result) == 4


def test_overlay_single_source_complete_only(con):
    t = ibis.memtable({"start": [1, 6], "end": [3, 8]})
    result = con.execute(
        t.overlay_intervals(start="start", end="end", complete_only=True)
    )
    assert result[["window_start", "window_end", "n"]].values.tolist() == [
        [1, 3, 1],
        [6, 8, 1],
    ]


def test_overlay_empty_input_keeps_fixed_schema(con):
    t = ibis.memtable(
        pd.DataFrame(
            {
                "pid": pd.Series([], dtype="int64"),
                "source": pd.Series([], dtype="string"),
                "start": pd.Series([], dtype="int64"),
                "end": pd.Series([], dtype="int64"),
            }
        )
    )
    expr = t.overlay_intervals(
        start="start",
        end="end",
        source="source",
        partition_by="pid",
        sources=SOURCES,
    )
    assert list(expr.schema().names) == [
        "pid",
        "window_start",
        "window_end",
        "a",
        "b",
        "c",
    ]
    assert con.execute(expr).empty


# ---------------------------------------------------------------------------
# Eager, construction-time validation (no backend execution required)
# ---------------------------------------------------------------------------
def test_overlay_eager_validation():
    string_table = ibis.table({"s": "string", "e": "string"}, name="t")
    with pytest.raises(com.IbisTypeError, match="numeric or temporal"):
        string_table.overlay_intervals(start="s", end="e")

    mismatched = ibis.table({"s": "int64", "e": "timestamp"}, name="t")
    with pytest.raises(com.IbisTypeError, match="same dtype"):
        mismatched.overlay_intervals(start="s", end="e")

    base = ibis.table({"start": "int64", "end": "int64", "source": "string"}, name="t")
    with pytest.raises(com.IbisInputError, match="null_endpoint"):
        base.overlay_intervals(start="start", end="end", null_endpoint="bogus")
    with pytest.raises(com.IbisInputError, match="together with a `source`"):
        base.overlay_intervals(start="start", end="end", sources=["a"])
    with pytest.raises(com.IbisInputError, match="explicit `sources`"):
        base.overlay_intervals(start="start", end="end", source="source", other="rest")
    with pytest.raises(com.IbisInputError, match="unique"):
        base.overlay_intervals(
            start="start", end="end", source="source", sources=["a", "a"]
        )

    colliding = ibis.table(
        {
            "start": "int64",
            "end": "int64",
            "source": "string",
            "window_start": "int64",
        },
        name="t",
    )
    with pytest.raises(com.IbisInputError, match="collide"):
        colliding.overlay_intervals(
            start="start",
            end="end",
            source="source",
            sources=["window_start"],
        )


# ---------------------------------------------------------------------------
# Randomized sweep-line cross-check (integer and timestamp endpoints)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("seed", [7, 23, 113])
def test_overlay_matches_python_sweep_line_integers(con, seed):
    rng = random.Random(seed)  # noqa: S311
    rows = []
    for pid in (1, 2, 3):
        for _i in range(40):
            left = rng.randrange(0, 10)
            right = left + rng.randrange(0, 4)  # equal endpoints allowed
            rows.append(
                {
                    "pid": pid,
                    "source": rng.choice(SOURCES),
                    "start": left,
                    "end": right,
                }
            )
    # explicit duplicate rows to exercise multiset counting
    rows.extend(rows[:5])

    t = ibis.memtable(pd.DataFrame(rows))
    expr = t.overlay_intervals(
        start="start",
        end="end",
        source="source",
        partition_by="pid",
        sources=SOURCES,
    )
    result = con.execute(expr)
    expected = sweep_line(
        rows,
        sources=SOURCES,
        partition_names=["pid"],
        has_source=True,
    )
    columns = ["pid", "window_start", "window_end", *SOURCES]
    _assert_matches(result, expected, columns)


def test_overlay_matches_python_sweep_line_timestamps(con):
    rng = random.Random(42)  # noqa: S311
    origin = pd.Timestamp("2024-01-01")
    rows = []
    for _i in range(60):
        left = origin + pd.Timedelta(days=rng.randrange(0, 12))
        right = left + pd.Timedelta(days=rng.randrange(0, 5))
        rows.append({"source": rng.choice(SOURCES[:2]), "start": left, "end": right})
    t = ibis.memtable(pd.DataFrame(rows))
    expr = t.overlay_intervals(
        start="start",
        end="end",
        source="source",
        sources=SOURCES[:2],
        merge=True,
    )
    result = con.execute(expr)

    # build the unmerged reference then apply the same adjacency merge
    raw = sweep_line(rows, sources=SOURCES[:2], partition_names=[], has_source=True)
    merged_expected = []
    for record in sorted(raw, key=lambda r: r["window_start"]):
        if (
            merged_expected
            and merged_expected[-1]["window_end"] == record["window_start"]
            and all(merged_expected[-1][s] == record[s] for s in SOURCES[:2])
        ):
            merged_expected[-1]["window_end"] = record["window_end"]
        else:
            merged_expected.append(dict(record))
    _assert_matches(
        result,
        merged_expected,
        ["window_start", "window_end", *SOURCES[:2]],
    )


def test_overlay_sweep_line_with_other_bucket(con):
    rng = random.Random(99)  # noqa: S311
    rows = []
    domain = ["a", "b", "x", "y"]
    for _i in range(50):
        left = rng.randrange(0, 8)
        right = left + rng.randrange(1, 4)
        rows.append({"source": rng.choice(domain), "start": left, "end": right})
    t = ibis.memtable(pd.DataFrame(rows))
    expr = t.overlay_intervals(
        start="start",
        end="end",
        source="source",
        sources=["a", "b"],
        other="rest",
    )
    result = con.execute(expr)

    reference_rows = [
        {
            **row,
            "source": row["source"] if row["source"] in ("a", "b") else "rest",
        }
        for row in rows
    ]
    expected = sweep_line(
        reference_rows,
        sources=["a", "b", "rest"],
        partition_names=[],
        has_source=True,
    )
    _assert_matches(
        result,
        expected,
        ["window_start", "window_end", "a", "b", "rest"],
    )


# ---------------------------------------------------------------------------
# SQL shape and growth bounds
# ---------------------------------------------------------------------------
def _parse_sql(expr, dialect):
    return sg.parse_one(str(ibis.to_sql(expr, dialect=dialect)), read=dialect)


def test_overlay_sql_uses_union_all_windows_and_no_joins():
    t = ibis.table(
        {
            "pid": "int64",
            "source": "string",
            "start": "int64",
            "end": "int64",
        },
        name="t",
    )
    expr = t.overlay_intervals(
        start="start",
        end="end",
        source="source",
        partition_by="pid",
        sources=SOURCES,
    )
    for dialect in ("duckdb", "postgres"):
        tree = _parse_sql(expr, dialect)
        assert not list(tree.find_all(sg.exp.Join)), "no joins allowed"
        unions = list(tree.find_all(sg.exp.Union))
        assert len(unions) == 1
        assert unions[0].args.get("distinct") is False
        windows = list(tree.find_all(sg.exp.Window))
        # one cumulative ROWS window per count column + one LEAD window
        assert len(windows) == len(SOURCES) + 1
        sql = tree.sql(dialect=dialect).upper()
        assert "UNION ALL" in sql
        assert "ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW" in sql
        assert "LEAD(" in sql
        assert "CROSS JOIN" not in sql


def test_overlay_intermediate_size_is_linear_in_endpoints(con):
    # n disjoint intervals in one partition: exactly 2n endpoints -> 2n-1
    # slices, regardless of n; never n^2 rows
    for n in (25, 100):
        rows = [{"source": "a", "start": 2 * i, "end": 2 * i + 1} for i in range(n)]
        expr = ibis.memtable(pd.DataFrame(rows)).overlay_intervals(
            start="start", end="end", source="source", sources=["a"]
        )
        result = con.execute(expr)
        assert len(result) == 2 * n - 1
        # gap slices carry an explicit 0 count while covered slices carry 1,
        # and total covered units equal the number of input intervals
        covered_units = (
            (result["window_end"] - result["window_start"]) * result["a"]
        ).sum()
        assert covered_units == n
        assert set(result["a"].unique()) == {0, 1}

    # n identical intervals collapse to a single slice whose count is n,
    # proving bag semantics and that nothing blows up with n
    n = 200
    identical = pd.DataFrame({"source": ["a"] * n, "start": [0] * n, "end": [10] * n})
    result = con.execute(
        ibis.memtable(identical).overlay_intervals(
            start="start", end="end", source="source", sources=["a"]
        )
    )
    assert len(result) == 1
    assert result["a"].iloc[0] == n

    # structural invariants hold as n grows: one UNION ALL, zero joins
    for n in (50, 200):
        frame = pd.DataFrame(
            {
                "source": ["a"] * n,
                "start": list(range(n)),
                "end": [v + 1 for v in range(n)],
            }
        )
        expr = ibis.memtable(frame).overlay_intervals(
            start="start", end="end", source="source", sources=["a"]
        )
        tree = _parse_sql(expr, "duckdb")
        assert not list(tree.find_all(sg.exp.Join))
        assert len(list(tree.find_all(sg.exp.Union))) == 1
        # number of window functions is constant (1 cumulative + 1 lead)
        assert len(list(tree.find_all(sg.exp.Window))) == 2


def test_overlay_postgres_dialect_compiles():
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
        sources=["a", "b"],
        other="rest",
        complete_only=True,
    )
    sql = str(ibis.to_sql(expr, dialect="postgres"))
    upper = sql.upper()
    compact = upper.replace(" ", "")
    assert "UNION ALL" in upper
    assert "LEAD(" in upper
    assert "ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW" in upper
    assert "FILTER(WHERE" in compact
    assert "CROSS JOIN" not in upper
