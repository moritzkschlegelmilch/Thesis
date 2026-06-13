from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping


@dataclass(frozen=True)
class ResourceForces:
    """Joined resource-indicator forces for one ordered object-type pair."""

    push: float
    pull: float


@dataclass(frozen=True)
class LayerAssignment:
    """Normalized process-area layer assignment.

    Layers are stored zero-indexed internally, but public layer ids follow the
    paper and start at 1.
    """

    layers: tuple[frozenset[str], ...]

    @classmethod
    def from_mapping(cls, object_to_layer: Mapping[str, int]) -> "LayerAssignment":
        if not object_to_layer:
            return cls(())

        normalized_layer_ids = {
            layer: index + 1
            for index, layer in enumerate(sorted(set(object_to_layer.values())))
        }
        grouped: dict[int, set[str]] = {}
        for object_type, layer in object_to_layer.items():
            grouped.setdefault(normalized_layer_ids[layer], set()).add(str(object_type))

        return cls(tuple(
            frozenset(grouped[layer])
            for layer in sorted(grouped)
        ))

    def layer_of(self, object_type: str) -> int:
        for index, object_types in enumerate(self.layers, start=1):
            if object_type in object_types:
                return index
        raise KeyError(f"Unknown object type {object_type!r}.")

    def object_types_at(self, layer: int) -> frozenset[str]:
        if layer < 1 or layer > len(self.layers):
            raise KeyError(f"Unknown layer {layer!r}.")
        return self.layers[layer - 1]

    def as_object_to_layer(self) -> dict[str, int]:
        return {
            object_type: layer
            for layer, object_types in enumerate(self.layers, start=1)
            for object_type in object_types
        }


@dataclass(frozen=True)
class ActivityLayerAssignment:
    activity_to_layer: Mapping[str, int]
    activities_by_layer: Mapping[int, frozenset[str]]


@dataclass
class AcceptingOCPN:
    raw: dict[str, Any]
    object_types: frozenset[str] = field(default_factory=frozenset)
    activities: frozenset[str] = field(default_factory=frozenset)

    @classmethod
    def from_raw(cls, raw: dict[str, Any] | None) -> "AcceptingOCPN | None":
        if raw is None:
            return None

        object_types = frozenset(str(object_type) for object_type in raw.get("petri_nets", {}))
        activities = frozenset(
            str(activity)
            for activity in raw.get("activities", ())
            if activity is not None
        )
        if not activities:
            activities = frozenset(
                str(transition.label)
                for net, _, _ in raw.get("petri_nets", {}).values()
                for transition in net.transitions
                if transition.label is not None
            )
        return cls(raw=raw, object_types=object_types, activities=activities)


@dataclass(frozen=True)
class SimpleSubprocess:
    object_type: str
    input_place: Any
    output_place: Any
    places: frozenset[Any]
    transitions: frozenset[Any]
    arcs: frozenset[Any]
    visible_activities: frozenset[str]
    raw: Any = field(default=None, compare=False, hash=False)

    @property
    def is_trivial(self) -> bool:
        return len(self.transitions) == 1


@dataclass(frozen=True)
class ObjectCentricSubprocess:
    simple_subprocesses: tuple[SimpleSubprocess, ...]
    visible_activities: frozenset[str]
    id: str | None = None
    raw: Any = field(default=None, compare=False, hash=False)

    @property
    def object_types(self) -> frozenset[str]:
        return frozenset(
            subprocess.object_type
            for subprocess in self.simple_subprocesses
        )


@dataclass
class AdvancedProcessArea:
    object_types: frozenset[str]
    activities: frozenset[str]
    net: AcceptingOCPN | None
    resources: Mapping[str, frozenset[str]]
    subprocesses: tuple[ObjectCentricSubprocess, ...] | tuple[Any, ...]


@dataclass
class AdvancedProcessAreaHierarchy:
    areas: tuple[AdvancedProcessArea, ...] = ()
    layer_assignment: LayerAssignment | None = None

    def object_types(self) -> frozenset[str]:
        return frozenset(
            object_type
            for area in self.areas
            for object_type in area.object_types
        )

    def activities(self) -> frozenset[str]:
        return frozenset(
            activity
            for area in self.areas
            for activity in area.activities
        )

    def append(self, area: AdvancedProcessArea) -> "AdvancedProcessAreaHierarchy":
        return AdvancedProcessAreaHierarchy(
            self.areas + (area,),
            layer_assignment=self.layer_assignment,
        )


@dataclass(frozen=True)
class PrecisionParameters:
    d: int | None = None
    l: int | None = None
    replay_budget: int | None = None
    sample_size: int | None = None
    random_seed: int | None = None


@dataclass
class DeltaReference:
    full_log: Any
    full_activities: frozenset[str]
    subprocess_activities: frozenset[str]
    full_net: AcceptingOCPN | None
    full_subprocesses: tuple[Any, ...]
    full_collapsed_net: AcceptingOCPN | None
    baseline_complexity: float
    baseline_precision: float
    precision_context: Any = None


@dataclass(frozen=True)
class QualityScores:
    simplicity_gain: float
    information_loss: float
    quality: float
    precision: float
    complexity: float
