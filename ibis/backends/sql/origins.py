"""SQL span to Ibis origin tracking.

This module implements the machinery behind
``SQLBackend.compile_with_origins``: it records, while SQLGlot *structurally*
generates a SQL string, which UTF-8 byte spans of the final SQL derive from
which original Ibis operations/fields.

The tracking is opt-in.  When disabled no trace state is maintained on the
compiler and compilation behaves exactly as before.

Design summary
--------------

* Stable origin identities (integers) are assigned to the nodes of the input
  DAG through a deterministic topological traversal.  The identifiers never
  depend on Python memory addresses and are stable for a given input DAG and
  compilation protocol.
* While translating the (possibly rewritten/lowered) DAG to SQLGlot
  expressions, every translated expression is registered on the per-compile
  tracker together with its origin reference.  Nodes synthesized by rewrites
  or lowering inherit the union of their children's origins; expressions
  without any Ibis provenance are explicitly marked synthetic.  The tracker
  is discarded after compilation, so shared SQLGlot constants (e.g. the
  module-level ``NULL``) are never mutated across compilations.
* A dialect-specific SQLGlot ``Generator`` subclass wraps the rendered SQL of
  each tracked expression with control-character markers.  Spans are therefore
  recorded on the structured generation path instead of being recovered by
  searching the final string, so quoted identifiers, comments, Unicode /
  multi-byte literals and repeated identical SQL fragments cannot be
  confused.
* Opaque SQL (``SQLStringView``/``SQLQueryResult``) is recorded as a single
  span over the whole parsed statement; its interior is never subdivided.
* The number of recorded spans is bounded; once the limit is reached further
  spans are dropped and the truncation is reported explicitly.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import sqlglot as sg
import sqlglot.expressions as sge
from public import public

import ibis.common.exceptions as com
import ibis.expr.operations as ops

if TYPE_CHECKING:
    from collections.abc import Mapping

# Origin reference tuple layout:
# (origin ids, synthetic flag, opaque flag, optional kind)
REF_IDS = 0
REF_SYNTHETIC = 1
REF_OPAQUE = 2
REF_KIND = 3

#: Default upper bound on the number of recorded spans.
DEFAULT_MAX_ORIGIN_SPANS = 10_000

# Markers use control characters that cannot collide with SQL keywords or
# identifiers.  If user-provided literal content ever reproduced the exact
# marker grammar, :meth:`OriginTracker.decode` fails loudly instead of
# producing misaligned spans.
_MARK = "\x00"
_MARKER_RE = re.compile(r"\x00([OC])(\d+)\x00")

_SYNTHETIC_SQLGLOT_NODES = {
    sge.Cast: "cast",
    sge.TryCast: "try_cast",
    sge.Null: "null",
}


@public
@dataclass(frozen=True)
class OriginInfo:
    """Description of a single original Ibis operation/field identity."""

    id: int
    """Stable identifier within the compiled input DAG."""

    kind: str
    """Either ``"field"`` or ``"operation"``."""

    name: str
    """Field name for fields, operation class name otherwise."""

    op_type: str
    """The Ibis operation class name, e.g. ``"Field"`` or ``"Add"``."""

    relation: str | None
    """The owning relation's name for fields, otherwise ``None``."""

    opaque: bool = False
    """Whether the origin is an opaque SQL string that cannot be subdivided."""


@public
@dataclass(frozen=True)
class OriginSpan:
    """A UTF-8 byte interval of the final SQL with its originating identities."""

    start: int
    """Inclusive UTF-8 byte offset into the compiled SQL."""

    end: int
    """Exclusive UTF-8 byte offset into the compiled SQL."""

    origins: tuple[int, ...]
    """Origin identifiers covering this span."""

    synthetic: bool = False
    """Whether the span has no direct Ibis node (wrapper/derived SQL)."""

    opaque: bool = False
    """Whether the span covers an opaque, non-subdividable SQL region."""

    kind: str | None = None
    """Optional qualifier, e.g. ``"cast"``, ``"null"`` or ``"derived"``."""


@public
@dataclass(frozen=True)
class CompiledOrigins:
    """Result of :meth:`SQLBackend.compile_with_origins`."""

    sql: str
    """The compiled SQL string, identical to the regular ``compile`` output."""

    spans: tuple[OriginSpan, ...]
    """Byte spans ordered by their position in :attr:`sql`."""

    origins: Mapping[int, OriginInfo]
    """Mapping of origin identifier to its description."""

    truncated: bool
    """Whether span recording stopped because :attr:`limit` was reached."""

    limit: int
    """The maximum number of spans that could be recorded."""

    def span_text(self, span: OriginSpan) -> str:
        """Return the exact SQL text covered by ``span``."""
        return self.sql.encode("utf-8")[span.start : span.end].decode("utf-8")

    def spans_at(self, byte_offset: int) -> tuple[OriginSpan, ...]:
        """Return all spans covering ``byte_offset`` (a UTF-8 byte offset)."""
        return tuple(
            span for span in self.spans if span.start <= byte_offset < span.end
        )

    def origins_at(self, byte_offset: int) -> tuple[OriginInfo, ...]:
        """Return the distinct origin descriptions covering ``byte_offset``."""
        seen: dict[int, OriginInfo] = {}
        for span in self.spans:
            if span.start <= byte_offset < span.end:
                for oid in span.origins:
                    seen.setdefault(oid, self.origins[oid])
        return tuple(seen.values())


def _make_origin_info(oid: int, node: ops.Node) -> OriginInfo:
    opaque = isinstance(node, (ops.SQLStringView, ops.SQLQueryResult))
    if isinstance(node, ops.Field):
        return OriginInfo(
            id=oid,
            kind="field",
            name=node.name,
            op_type=type(node).__name__,
            relation=getattr(node.rel, "name", None),
            opaque=False,
        )
    op_type = type(node).__name__
    return OriginInfo(
        id=oid,
        kind="operation",
        name=op_type,
        op_type=op_type,
        relation=None,
        opaque=opaque,
    )


def _collect_origin_ids(value: Any, refs: dict[int, tuple], ids: set[int]) -> None:
    """Recursively collect origin ids attached to translated SQLGlot nodes."""
    if isinstance(value, sge.Expression):
        ref = refs.get(id(value))
        if ref is not None:
            ids.update(ref[REF_IDS])
    elif isinstance(value, dict):
        for sub in value.values():
            _collect_origin_ids(sub, refs, ids)
    elif isinstance(value, (list, tuple)):
        for sub in value:
            _collect_origin_ids(sub, refs, ids)


class OriginTracker:
    """Per-compilation state mapping SQLGlot output back to Ibis origins.

    A tracker instance is only installed on a compiler for the duration of a
    single ``compile_with_origins`` call and is discarded afterwards.
    """

    def __init__(self, *, limit: int = DEFAULT_MAX_ORIGIN_SPANS) -> None:
        self.limit = limit
        self.truncated = False
        self._node_ids: dict[ops.Node, int] = {}
        self._infos: dict[int, OriginInfo] = {}
        # SQLGlot expression object id -> origin reference.  Keyed on object
        # identity rather than stored on the expression so that SQLGlot
        # constants shared across compilations are never mutated.
        self._refs: dict[int, tuple] = {}
        self._synthetic_refs: dict[int, tuple] = {}
        self._span_refs: list[tuple] = []
        self._opaque_depth = 0
        self._generator_cache: dict[type[sg.Dialect], type] = {}

    def prepare(self, root: ops.Node) -> None:
        """Assign stable origin ids to every node of the original DAG.

        The traversal is a deterministic breadth-first topological order.
        Value-equal nodes (ibis nodes are value-hashable) share a single
        identity, while same-named fields of different relations remain
        distinct.
        """
        ordered = root.map_nodes(lambda node, *_: None)
        for node in ordered:
            oid = len(self._node_ids)
            self._node_ids[node] = oid
            self._infos[oid] = _make_origin_info(oid, node)

    def tag(self, node: ops.Node, result: Any, kwargs: dict) -> None:
        """Attach the origin reference of ``node`` to its SQLGlot result."""
        if not isinstance(result, sge.Expression):
            return
        key = id(result)
        if key in self._refs:
            # Reused subexpressions already carry their own, more specific
            # origin (e.g. a no-op projection returning its parent relation).
            return
        ref = self._ref_for(node, kwargs)
        self._refs[key] = ref
        # Subquery wrappers can be unwrapped by translate/to_sqlglot before
        # generation, so register the wrapped statement with the same ref
        # (unless it already carries a more specific origin).
        if isinstance(result, sge.Subquery) and isinstance(result.this, sge.Expression):
            self._refs.setdefault(id(result.this), ref)

    def _ref_for(self, node: ops.Node, kwargs: dict) -> tuple:
        oid = self._node_ids.get(node)
        if oid is not None:
            opaque = isinstance(node, (ops.SQLStringView, ops.SQLQueryResult))
            return ((oid,), False, opaque, None)

        # The node was introduced by a rewrite or backend lowering: derive its
        # provenance from the already translated children.
        ids: set[int] = set()
        _collect_origin_ids(kwargs, self._refs, ids)
        if ids:
            return (tuple(sorted(ids)), False, False, "derived")
        # Purely synthesized SQL with no Ibis provenance (e.g. a NULL literal
        # generated by a lowering rule).
        return ((), True, False, type(node).__name__)

    def ref_for_sqlglot(self, expression: sge.Expression) -> tuple | None:
        """Resolve the origin reference of a SQLGlot node being generated."""
        key = id(expression)
        direct = self._refs.get(key)
        if direct is not None:
            return direct

        kind = _SYNTHETIC_SQLGLOT_NODES.get(type(expression))
        if kind is None:
            return None

        # Synthetic cast / NULL wrappers: map them to every origin found in
        # their subtree, or mark them as purely synthetic when no tracked
        # descendant exists.  Cached for the lifetime of this compilation.
        cached = self._synthetic_refs.get(key)
        if cached is not None:
            return cached

        ids: set[int] = set()
        for sub in expression.find_all(sge.Expression):
            ref = self._refs.get(id(sub))
            if ref is not None:
                ids.update(ref[REF_IDS])
        ref = (tuple(sorted(ids)), True, False, kind)
        self._synthetic_refs[key] = ref
        return ref

    def add_span(self, ref: tuple) -> int | None:
        """Reserve a span slot, returning its id or ``None`` when truncated."""
        if len(self._span_refs) >= self.limit:
            self.truncated = True
            return None
        self._span_refs.append(ref)
        return len(self._span_refs) - 1

    def enter_opaque(self) -> None:
        self._opaque_depth += 1

    def leave_opaque(self) -> None:
        self._opaque_depth -= 1

    def make_generator(self, dialect: type[sg.Dialect], **opts: Any):
        """Build a span-recording generator for ``dialect``."""
        generator_cls = self._generator_cache.get(dialect)
        if generator_cls is None:
            base_cls = dialect.generator_class
            tracker = self

            class OriginGenerator(base_cls):
                """Generator that wraps tracked nodes with span markers."""

                def sql(self, expression, key=None, comment=True):
                    if (
                        not key
                        and isinstance(expression, sge.Expression)
                        and tracker._opaque_depth == 0
                    ):
                        ref = tracker.ref_for_sqlglot(expression)
                        if ref is not None:
                            if ref[REF_OPAQUE]:
                                # Opaque SQL interiors cannot be subdivided.
                                tracker.enter_opaque()
                                try:
                                    rendered = super().sql(expression, key, comment)
                                finally:
                                    tracker.leave_opaque()
                            else:
                                rendered = super().sql(expression, key, comment)

                            span_id = tracker.add_span(ref)
                            if span_id is not None:
                                return (
                                    _MARK
                                    + f"O{span_id}"
                                    + _MARK
                                    + rendered
                                    + _MARK
                                    + f"C{span_id}"
                                    + _MARK
                                )
                            return rendered
                    return super().sql(expression, key, comment)

            generator_cls = OriginGenerator
            self._generator_cache[dialect] = generator_cls

        return generator_cls(dialect=dialect, **opts)

    def decode(self, marked_sql: str) -> tuple[str, tuple[OriginSpan, ...]]:
        """Strip the markers and translate them into UTF-8 byte spans."""
        parts: list[str] = []
        events: list[tuple[str, int, int]] = []
        clean_length = 0
        pos = 0
        for match in _MARKER_RE.finditer(marked_sql):
            parts.append(marked_sql[pos : match.start()])
            clean_length += len(parts[-1])
            events.append((match.group(1), int(match.group(2)), clean_length))
            pos = match.end()
        parts.append(marked_sql[pos:])
        clean_sql = "".join(parts)

        # Character offset -> UTF-8 byte offset, computed in a single pass.
        byte_at = [0]
        for char in clean_sql:
            byte_at.append(byte_at[-1] + len(char.encode("utf-8")))

        open_spans: dict[int, int] = {}
        decoded: list[tuple[int, int, int]] = []
        for kind, span_id, char_pos in events:
            if kind == "O":
                if span_id in open_spans:
                    raise com.IbisError(
                        "invalid origin marker stream: duplicated span marker"
                    )
                open_spans[span_id] = char_pos
            else:
                start = open_spans.pop(span_id, None)
                if start is None:
                    raise com.IbisError(
                        "invalid origin marker stream: unmatched span marker"
                    )
                decoded.append((span_id, byte_at[start], byte_at[char_pos]))
        if open_spans:
            raise com.IbisError("invalid origin marker stream: unclosed span marker")

        expected = set(range(len(self._span_refs)))
        seen = {span_id for span_id, _, _ in decoded}
        if seen != expected:
            raise com.IbisError(
                "invalid origin marker stream: marker corruption detected"
            )

        spans = (
            OriginSpan(
                start=start,
                end=end,
                origins=ref[REF_IDS],
                synthetic=ref[REF_SYNTHETIC],
                opaque=ref[REF_OPAQUE],
                kind=ref[REF_KIND],
            )
            for span_id, start, end in decoded
            for ref in [self._span_refs[span_id]]
        )
        # present spans in SQL position order; for nested spans starting at
        # the same offset, the enclosing (wider) span comes first
        spans = tuple(sorted(spans, key=lambda span: (span.start, -span.end)))
        return clean_sql, spans

    def build_result(self, marked_sql: str) -> CompiledOrigins:
        sql, spans = self.decode(marked_sql)
        return CompiledOrigins(
            sql=sql,
            spans=spans,
            origins=dict(self._infos),
            truncated=self.truncated,
            limit=self.limit,
        )
