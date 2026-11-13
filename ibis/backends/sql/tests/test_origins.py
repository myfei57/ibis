from __future__ import annotations

import pytest
import sqlglot as sg
import sqlglot.expressions as sge

import ibis
import ibis.common.exceptions as com
import ibis.expr.operations as ops
from ibis.backends.sql import origins
from ibis.backends.sql.compilers import duckdb as duckdb_compiler_module


def compile_with_origins(expr, **kwargs):
    return ibis.duckdb.compile_with_origins(expr, **kwargs)


def assert_spans_decodable(result):
    """Every span must slice its exact token text out of the UTF-8 bytes."""
    raw = result.sql.encode("utf-8")
    assert len(raw) >= max(span.end for span in result.spans)
    previous = -1
    for span in result.spans:
        assert span.start >= previous
        assert span.start < span.end
        # byte offsets must land on UTF-8 character boundaries
        text = raw[span.start : span.end].decode("utf-8")
        assert text == result.span_text(span)
        previous = span.start
    assert "\x00" not in result.sql


def spans_with_text(result, text):
    return [span for span in result.spans if result.span_text(span) == text]


def test_compile_with_origins_matches_plain_compile():
    t = ibis.table({"id": "int64", "v": "string"}, name="t")
    expr = t.filter(t.id > 3).select(t.v, (t.id * 2).name("d"))

    result = compile_with_origins(expr)
    assert isinstance(result, origins.CompiledOrigins)
    assert result.sql == ibis.duckdb.compile(expr)
    assert result.truncated is False
    assert result.limit == origins.DEFAULT_MAX_ORIGIN_SPANS
    assert len(result.spans) > 0
    assert_spans_decodable(result)


def test_disabled_compilation_keeps_no_trace():
    t = ibis.table({"id": "int64"}, name="t")
    ibis.duckdb.compile(t)
    assert duckdb_compiler_module.compiler._origin_tracker is None

    compile_with_origins(t)
    assert duckdb_compiler_module.compiler._origin_tracker is None


def test_filter_merge_locates_predicate_origin():
    t = ibis.table({"id": "int64", "v": "string"}, name="t")
    # filter + projection fuse into a single SELECT (no subquery layer)
    expr = t.filter(t.id > 3).select(t.v)
    result = compile_with_origins(expr)

    assert result.sql.count("SELECT") == 1
    assert_spans_decodable(result)

    predicate_spans = [
        span
        for span in result.spans
        if any(result.origins[oid].op_type == "Greater" for oid in span.origins)
    ]
    assert predicate_spans
    # a precise span covers exactly the predicate token, while the enclosing
    # fused query block may carry the same origin as a derived superset
    precise = spans_with_text(result, '"t0"."id" > 3')
    assert precise
    predicate = min(precise, key=lambda s: s.end - s.start)
    assert any(result.origins[oid].op_type == "Greater" for oid in predicate.origins)

    field_in_filter = [
        span
        for span in result.spans
        if result.span_text(span) == '"t0"."id"'
        and span.start > result.sql.index("WHERE")
    ]
    assert len(field_in_filter) == 1
    field_origin = result.origins[field_in_filter[0].origins[0]]
    assert field_origin.kind == "field"
    assert field_origin.name == "id"


def test_window_layering_origins():
    t = ibis.table({"id": "int64", "v": "string", "g": "string"}, name="t")
    expr = t.mutate(rn=ibis.row_number().over(group_by=t.g, order_by=t.id))
    result = compile_with_origins(expr)
    assert_spans_decodable(result)

    window_infos = {
        oid: info
        for oid, info in result.origins.items()
        if info.op_type == "WindowFunction"
    }
    assert window_infos
    window_oid = next(iter(window_infos))
    window_spans = [span for span in result.spans if window_oid in span.origins]
    assert window_spans
    # the window function token itself is covered by a window-origin span
    assert any("ROW_NUMBER" in result.span_text(span) for span in window_spans)
    # partition/ordering fields keep their own precise spans
    partition_fields = spans_with_text(result, '"t0"."g"')
    assert partition_fields
    assert all(
        any(
            result.origins[oid].kind == "field" and result.origins[oid].name == "g"
            for oid in span.origins
        )
        for span in partition_fields
    )


def test_self_join_same_named_columns_have_distinct_origins():
    t = ibis.table({"id": "int64", "v": "string"}, name="t")
    left = t
    right = t.view()
    expr = left.join(right, left.id == right.id).select(
        left.id, right.id.name("id_right")
    )
    result = compile_with_origins(expr)
    assert_spans_decodable(result)

    left_tokens = spans_with_text(result, '"t1"."id"')
    right_tokens = spans_with_text(result, '"t2"."id"')
    assert left_tokens and right_tokens

    left_origin_ids = {oid for span in left_tokens for oid in span.origins}
    right_origin_ids = {oid for span in right_tokens for oid in span.origins}
    # same column name, but the two branches must map to distinct identities
    assert left_origin_ids and right_origin_ids
    assert left_origin_ids.isdisjoint(right_origin_ids)
    for oid in left_origin_ids | right_origin_ids:
        assert result.origins[oid].kind == "field"
        assert result.origins[oid].name == "id"


def test_database_error_position_locates_correct_branch():
    t = ibis.table({"id": "int64", "v": "string"}, name="t")
    left = t
    right = t.view()
    expr = left.join(right, left.id == right.id).select(
        left.id, right.id.name("id_right")
    )
    result = compile_with_origins(expr)

    right_span = spans_with_text(result, '"t2"."id"')[0]
    left_span = spans_with_text(result, '"t1"."id"')[0]

    # simulate a database error reported at a byte position inside each token
    right_hits = result.origins_at((right_span.start + right_span.end) // 2)
    left_hits = result.origins_at((left_span.start + left_span.end) // 2)
    right_ids = {info.id for info in right_hits if info.kind == "field"}
    left_ids = {info.id for info in left_hits if info.kind == "field"}
    assert right_ids and left_ids
    assert right_ids.isdisjoint(left_ids)

    # end-to-end: break the SQL at the right branch token and confirm the
    # database rejects it while the origin map still points at that branch
    duckdb = pytest.importorskip("duckdb")
    con = ibis.duckdb.connect()
    broken = result.sql.replace('"t2"."id"', '"t2"."missing_col"', 1)
    with pytest.raises(duckdb.Error):
        con.con.execute(broken)
    assert any(info.name == "id" and info.id in right_ids for info in right_hits)


def test_shared_cte_reuses_origin_at_independent_spans():
    t = ibis.table({"id": "int64", "v": "string"}, name="t")
    shared = t.filter(t.id > 1)
    mirrored = shared.view()
    expr = shared.join(mirrored, shared.id == mirrored.id)

    result = compile_with_origins(expr)
    assert_spans_decodable(result)
    assert result.sql.lstrip().startswith("WITH")

    # the filtered field identity appears at independent SQL positions
    positions: dict[int, list[tuple[int, int]]] = {}
    for span in result.spans:
        for oid in span.origins:
            positions.setdefault(oid, []).append((span.start, span.end))

    reused = {oid: ranges for oid, ranges in positions.items() if len(set(ranges)) >= 2}
    assert reused
    # every reused identity is a real, registered Ibis origin
    for oid in reused:
        assert oid in result.origins


def test_repeated_identical_fragments_keep_independent_spans():
    t = ibis.table({"id": "int64"}, name="t")
    expr = t.select((t.id + 1).name("a"), (t.id + 1).name("b"))
    result = compile_with_origins(expr)
    assert_spans_decodable(result)

    literal_spans = [span for span in result.spans if result.span_text(span) == "1"]
    assert len(literal_spans) == 2
    first, second = literal_spans
    assert (first.start, first.end) != (second.start, second.end)
    # identical fragments point at the same literal identity
    assert first.origins == second.origins
    # spans must not overlap each other
    assert first.end <= second.start or second.end <= first.start


def test_unicode_multibyte_byte_offsets():
    t = ibis.table({"名前": "string", "id": "int64"}, name="日本語テーブル")
    expr = t.select(t["名前"], ibis.literal("你好🌟").name("greeting"))
    result = compile_with_origins(expr)
    assert_spans_decodable(result)

    assert len(result.sql.encode("utf-8")) > len(result.sql)
    greeting_spans = [
        span for span in result.spans if "你好🌟" in result.span_text(span)
    ]
    assert greeting_spans
    span = greeting_spans[0]
    # the span covers multi-byte text, so its byte width exceeds its
    # character width
    text = result.span_text(span)
    assert span.end - span.start > len(text)
    assert len(text.encode("utf-8")) > len(text)


def test_quoted_identifier_span_covers_full_token():
    t = ibis.table({"id": "int64"}, name="t")
    # add a computed column so the field is rendered explicitly instead of
    # being collapsed into a SELECT *
    result = compile_with_origins(t.select(t.id, (t.id + 1).name("x")))
    field_spans = spans_with_text(result, '"t0"."id"')
    assert field_spans
    for span in field_spans:
        # quotes are part of the covered token, no regex-style partial match
        assert result.span_text(span).startswith('"')
        assert result.span_text(span).endswith('"')


def test_synthetic_null_wrapper_is_explicitly_synthetic():
    result = compile_with_origins(ibis.literal(None, type="int64"))
    assert_spans_decodable(result)

    null_spans = [span for span in result.spans if span.kind == "null"]
    assert null_spans
    assert all(span.synthetic for span in null_spans)
    assert all(span.origins == () for span in null_spans)
    assert result.span_text(null_spans[0]) == "NULL"

    # the surrounding cast derives from the literal operation itself
    cast_spans = [
        span for span in result.spans if result.span_text(span).startswith("CAST")
    ]
    assert cast_spans
    assert any(span.origins == (0,) for span in cast_spans)


def test_synthetic_cast_unions_subtree_origins():
    # unit-level: an untagged cast wrapping two tracked columns must map to
    # the union of their origins and be marked synthetic
    tracker = origins.OriginTracker()
    left = sge.column("a", quoted=True)
    right = sge.column("b", quoted=True)
    tracker._refs[id(left)] = ((1,), False, False, None)
    tracker._refs[id(right)] = ((2,), False, False, None)
    add = sge.Add(this=left, expression=right)
    cast = sge.Cast(this=add, to=sge.DataType.build("double"))

    ref = tracker.ref_for_sqlglot(cast)
    assert ref is not None
    assert ref[origins.REF_SYNTHETIC] is True
    assert ref[origins.REF_KIND] == "cast"
    assert set(ref[origins.REF_IDS]) == {1, 2}

    # a plain untagged node without cast/NULL shape has no synthetic ref
    assert tracker.ref_for_sqlglot(add) is None


def test_opaque_sql_is_recorded_as_one_span():
    opaque_op = ops.SQLQueryResult(
        "SELECT a + 1 AS x FROM weird_src",
        ibis.schema({"x": "int64"}),
        source=None,
    )
    result = compile_with_origins(opaque_op.to_expr())
    assert_spans_decodable(result)

    opaque_spans = [span for span in result.spans if span.opaque]
    assert len(opaque_spans) == 1
    opaque_span = opaque_spans[0]
    assert result.span_text(opaque_span) == result.sql.strip()
    # nothing inside the opaque region is subdivided
    interior = [
        span
        for span in result.spans
        if span is not opaque_span
        and span.start >= opaque_span.start
        and span.end <= opaque_span.end
    ]
    assert not interior


def test_opaque_string_view_is_recorded_as_one_span():
    table = ibis.table({"a": "int64", "x": "int64"}, name="src")
    view_op = ops.SQLStringView(
        parent=table.op(),
        query="SELECT a + 1 AS x FROM src",
        schema=ibis.schema({"x": "int64"}),
    )
    result = compile_with_origins(view_op.to_expr())
    assert_spans_decodable(result)

    opaque_spans = [span for span in result.spans if span.opaque]
    assert len(opaque_spans) == 1
    opaque_span = opaque_spans[0]
    assert "SELECT a + 1 AS x FROM src" in result.span_text(opaque_span)
    interior = [
        span
        for span in result.spans
        if span is not opaque_span
        and span.start >= opaque_span.start
        and span.end <= opaque_span.end
    ]
    assert not interior


def test_no_span_fakes_mapping_to_every_origin():
    t = ibis.table({"id": "int64", "v": "string", "f": "float64"}, name="t")
    expr = t.filter(t.id > 3).select(t.v, (t.f * 2).name("d"))
    result = compile_with_origins(expr)
    assert_spans_decodable(result)

    all_origin_ids = set(result.origins)
    assert all_origin_ids
    for span in result.spans:
        # even the enclosing root query block only derives from its immediate
        # children; it must never be mapped to the whole DAG's identity set
        assert set(span.origins) != all_origin_ids or len(all_origin_ids) == 1


def test_truncation_is_reported_explicitly():
    t = ibis.table({"id": "int64", "v": "string"}, name="t")
    expr = t.filter(t.id > 1).select(t.v, (t.id * 2).name("d"))

    result = compile_with_origins(expr, max_spans=3)
    assert result.truncated is True
    assert len(result.spans) == 3
    # the SQL itself remains complete and executable-looking
    assert result.sql == ibis.duckdb.compile(expr)


def test_origin_ids_are_stable_ints_within_protocol():
    t = ibis.table({"id": "int64", "v": "string"}, name="t")
    left = t
    right = t.view()
    expr = left.join(right, left.id == right.id)

    first = compile_with_origins(expr)
    second = compile_with_origins(expr)

    def field_id_map(result):
        return {
            (info.op_type, info.name): info.id
            for info in result.origins.values()
            if info.kind == "field"
        }

    assert field_id_map(first) == field_id_map(second)
    assert all(isinstance(oid, int) for oid in first.origins)
    assert max(first.origins) < len(first.origins)


def test_pretty_mode_spans_remain_valid():
    t = ibis.table({"id": "int64", "v": "string"}, name="t")
    expr = t.filter(t.id > 3).select(t.v)
    result = compile_with_origins(expr, pretty=True)
    assert_spans_decodable(result)
    assert "\n" in result.sql
    assert "".join(result.sql.split()) == "".join(ibis.duckdb.compile(expr).split())


def test_comments_do_not_break_span_decoding():
    tracker = origins.OriginTracker()
    parsed = sg.parse_one("SELECT a FROM t /* keep me */", read="duckdb")
    tracker._refs[id(parsed)] = ((0,), False, False, None)
    generator = tracker.make_generator(
        duckdb_compiler_module.compiler.dialect,
        pretty=False,
        unsupported_level=sg.ErrorLevel.RAISE,
        comments=True,
    )
    marked = generator.generate(parsed, copy=False)
    result = tracker.build_result(marked)

    assert "keep me" in result.sql
    assert_spans_decodable(result)
    assert any(span.origins == (0,) for span in result.spans)


def test_corrupt_marker_stream_raises():
    tracker = origins.OriginTracker()
    tracker._span_refs.append(((0,), False, False, None))
    with pytest.raises(com.IbisError, match="unmatched span marker"):
        tracker.decode("\x00C0\x00")
    with pytest.raises(com.IbisError, match="unclosed span marker"):
        tracker.decode("\x00O0\x00")
