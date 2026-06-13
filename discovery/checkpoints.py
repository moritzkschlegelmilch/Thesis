from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
import json
from pathlib import Path
import sys
from time import perf_counter
from typing import Any, Iterable, Iterator

from tqdm import tqdm


@dataclass(frozen=True)
class CheckpointRecord:
    name: str
    parent: str | None
    status: str
    started_at: float
    ended_at: float
    duration_seconds: float
    count: int = 0
    total: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class CheckpointManager:
    """Shared timing/progress utility for framework-level checkpoints."""

    def __init__(
        self,
        *,
        verbose: bool = False,
        collect: bool = True,
        file=None,
    ):
        self.verbose = verbose
        self.collect = collect
        self.file = file or sys.stdout
        self.records: list[CheckpointRecord] = []
        self._stack: list[CheckpointSpan] = []

    @property
    def current(self) -> "CheckpointSpan | None":
        return self._stack[-1] if self._stack else None

    @contextmanager
    def section(
        self,
        name: str,
        *,
        total: int | None = None,
        metadata: dict[str, Any] | None = None,
        unit: str = "step",
    ) -> Iterator["CheckpointSpan"]:
        span = CheckpointSpan(
            manager=self,
            name=name,
            total=total,
            metadata=metadata or {},
            unit=unit,
        )
        span.__enter__()
        try:
            yield span
        except Exception as exc:
            span.__exit__(type(exc), exc, exc.__traceback__)
            raise
        else:
            span.__exit__(None, None, None)

    def iter(
        self,
        iterable: Iterable[Any],
        name: str,
        *,
        total: int | None = None,
        metadata: dict[str, Any] | None = None,
        unit: str = "item",
    ) -> Iterator[Any]:
        if total is None:
            try:
                total = len(iterable)  # type: ignore[arg-type]
            except TypeError:
                total = None

        with self.section(name, total=total, metadata=metadata, unit=unit) as span:
            for item in iterable:
                yield item
                span.update()

    def as_dicts(self) -> list[dict[str, Any]]:
        return [
            {
                "name": record.name,
                "parent": record.parent,
                "status": record.status,
                "started_at": record.started_at,
                "ended_at": record.ended_at,
                "duration_seconds": record.duration_seconds,
                "count": record.count,
                "total": record.total,
                "metadata": record.metadata,
            }
            for record in self.records
        ]

    def to_json(self, output_path: str) -> None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            json.dump(self.as_dicts(), handle, default=str, indent=2, sort_keys=True)

    def print_summary(self) -> None:
        if not self.records:
            return

        print("Checkpoint summary:", file=self.file)
        for record in self.records:
            metadata = _format_metadata(record.metadata)
            count = (
                f" [{record.count}/{record.total}]"
                if record.total is not None
                else f" [{record.count}]"
            )
            suffix = f" | {metadata}" if metadata else ""
            print(
                f"  {record.duration_seconds:8.3f}s {record.status:>6} "
                f"{record.name}{count}{suffix}",
                file=self.file,
            )


class CheckpointSpan:
    def __init__(
        self,
        *,
        manager: CheckpointManager,
        name: str,
        total: int | None,
        metadata: dict[str, Any],
        unit: str,
    ):
        self.manager = manager
        self.name = name
        self.total = total
        self.metadata = dict(metadata)
        self.unit = unit
        self.count = 0
        self.started_at = 0.0
        self._bar = None
        self._bar_total = total if total is not None else 1

    def __enter__(self) -> "CheckpointSpan":
        self.started_at = perf_counter()
        depth = len(self.manager._stack)
        self.manager._stack.append(self)

        if self.manager.verbose:
            description = f"{'  ' * depth}{self.name}"
            self._bar = tqdm(
                total=self._bar_total,
                desc=description,
                unit=self.unit,
                file=self.manager.file,
                leave=False,
                dynamic_ncols=True,
                position=depth,
            )
            if self.metadata:
                self._bar.set_postfix_str(_format_metadata(self.metadata), refresh=False)

        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        ended_at = perf_counter()
        status = "failed" if exc_type is not None else "ok"

        if self._bar is not None:
            if self.total is None and self.count == 0:
                self.update()
            self._bar.set_postfix_str(
                f"{status}, {ended_at - self.started_at:.2f}s",
                refresh=False,
            )
            self._bar.close()

        if self.manager._stack and self.manager._stack[-1] is self:
            self.manager._stack.pop()
        elif self in self.manager._stack:
            self.manager._stack.remove(self)

        if self.manager.collect:
            parent = self.manager.current.name if self.manager.current is not None else None
            self.manager.records.append(CheckpointRecord(
                name=self.name,
                parent=parent,
                status=status,
                started_at=self.started_at,
                ended_at=ended_at,
                duration_seconds=ended_at - self.started_at,
                count=self.count,
                total=self.total,
                metadata=dict(self.metadata),
            ))

    def update(
        self,
        n: int = 1,
        *,
        metadata: dict[str, Any] | None = None,
        postfix: str | None = None,
    ) -> None:
        self.count += n
        if metadata:
            self.metadata.update(metadata)
        if self._bar is not None:
            self._bar.update(n)
            if postfix is not None:
                self._bar.set_postfix_str(postfix, refresh=False)
            elif metadata:
                self._bar.set_postfix_str(_format_metadata(self.metadata), refresh=False)

    def child(
        self,
        name: str,
        *,
        total: int | None = None,
        metadata: dict[str, Any] | None = None,
        unit: str = "step",
    ):
        return self.manager.section(
            name,
            total=total,
            metadata=metadata,
            unit=unit,
        )


def noop_checkpoint() -> CheckpointManager:
    return CheckpointManager(verbose=False, collect=False)


def _format_metadata(metadata: dict[str, Any]) -> str:
    return ", ".join(
        f"{key}={value}"
        for key, value in metadata.items()
        if value is not None
    )
