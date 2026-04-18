from __future__ import annotations

from collections import Counter, defaultdict, deque
from dataclasses import dataclass
from itertools import product
import sys
from typing import Any

from pm4py.objects.ocel.obj import OCEL
from pm4py.objects.petri_net.obj import PetriNet


Token = tuple[str, Any]
PlaceKey = tuple[str, PetriNet.Place]
ContextKey = tuple[tuple[str, tuple[tuple[tuple[str, ...], int], ...]], ...]


@dataclass(frozen=True)
class _LocalTransition:
    object_type: str
    label: str | None
    input_places: tuple[PlaceKey, ...]
    output_places: tuple[PlaceKey, ...]
    variable: bool


@dataclass(frozen=True)
class _BindingStep:
    label: str
    objects_by_type: tuple[tuple[str, tuple[Token, ...]], ...]

    def non_empty_object_types(self) -> set[str]:
        return {ot for ot, t in self.objects_by_type if t}


@dataclass(frozen=True)
class _ReplayEvent:
    context_key: ContextKey
    binding_sequence: tuple[_BindingStep, ...]
    context_tokens_by_type: tuple[tuple[str, tuple[Token, ...]], ...]


@dataclass(frozen=True)
class _EvaluationResult:
    precision: float
    fitness: float


@dataclass(frozen=True)
class _ComplexityStats:
    place_visible_degree_sum: int
    place_silent_degree_sum: int
    visible_transition_sum: int
    silent_transition_sum: int


@dataclass
class _ModelCache:
    place_ids: dict[PlaceKey, int]
    initial_places_by_type: dict[str, tuple[PlaceKey, ...]]
    silent_transitions: tuple[_LocalTransition, ...]
    visible_combinations_by_label: dict[str, tuple]
    enabled_labels_cache: dict[tuple, frozenset[str]]
    complexity_stats: _ComplexityStats


class NetQuality:
    def __init__(
        self,
        ocpn: dict[str, Any],
        ocel: OCEL | None = None,
        *,
        max_nodes_per_replay: int | None = 5000,
    ) -> None:
        self.ocpn = ocpn
        self.ocel = ocel
        self.max_nodes_per_replay = max_nodes_per_replay

        self._model_cache: _ModelCache | None = None
        self._complexity_cache: dict[tuple[float, float], float] = {}

    # -------------------------
    # PUBLIC API
    # -------------------------
    def precision(self, ocel=None):
        return self._evaluate(ocel).precision

    def fitness(self, ocel=None):
        return self._evaluate(ocel).fitness

    def complexity(self, alpha: float = 2, beta: float = 2) -> float:
        mc = self._ensure_model_cache()
        key = (alpha, beta)

        if key not in self._complexity_cache:
            stats = mc.complexity_stats
            self._complexity_cache[key] = (
                stats.place_visible_degree_sum
                + alpha * stats.place_silent_degree_sum
                + beta * stats.silent_transition_sum
                + stats.visible_transition_sum
            )

        return self._complexity_cache[key]

    # -------------------------
    # MODEL CACHE
    # -------------------------
    def _ensure_model_cache(self):
        if self._model_cache:
            return self._model_cache

        place_ids = {}
        silent = []
        visible = defaultdict(lambda: defaultdict(list))

        merged_places = defaultdict(lambda: {
            "incoming_visible": set(),
            "outgoing_visible": set(),
            "incoming_silent": set(),
            "outgoing_silent": set(),
        })

        merged_visible = defaultdict(lambda: {"pred": set(), "succ": set(), "types": set()})
        merged_silent = defaultdict(lambda: {"pred": set(), "succ": set(), "types": set()})

        initial_places_by_type = {}

        for ot in self.ocpn["petri_nets"]:
            net, im, _ = self.ocpn["petri_nets"][ot]

            for p in net.places:
                pk = (ot, p)
                place_ids[pk] = len(place_ids)
                merged_places[pk]

            initial_places_by_type[ot] = tuple((ot, p) for p, c in im.items() if c > 0)

            for t in net.transitions:
                inp = tuple((ot, a.source) for a in t.in_arcs if isinstance(a.source, PetriNet.Place))
                out = tuple((ot, a.target) for a in t.out_arcs if isinstance(a.target, PetriNet.Place))

                lt = _LocalTransition(ot, t.label, inp, out, False)

                if t.label:
                    visible[t.label][ot].append(lt)
                else:
                    silent.append(lt)

            # complexity tracking
            for arc in net.arcs:
                if isinstance(arc.source, PetriNet.Place):
                    pk = (ot, arc.source)
                    if arc.target.label:
                        merged_places[pk]["outgoing_visible"].add(arc.target.label)
                        merged_visible[arc.target.label]["pred"].add(pk)
                    else:
                        merged_places[pk]["outgoing_silent"].add(arc.target.name)
                        merged_silent[arc.target.name]["pred"].add(pk)
                else:
                    pk = (ot, arc.target)
                    if arc.source.label:
                        merged_places[pk]["incoming_visible"].add(arc.source.label)
                        merged_visible[arc.source.label]["succ"].add(pk)
                    else:
                        merged_places[pk]["incoming_silent"].add(arc.source.name)
                        merged_silent[arc.source.name]["succ"].add(pk)

        combos = {}
        for label, by_type in visible.items():
            combos[label] = tuple(
                tuple(zip(by_type.keys(), c))
                for c in product(*by_type.values())
            )

        # complexity stats
        stats = _ComplexityStats(
            place_visible_degree_sum=sum(len(v["incoming_visible"]) + len(v["outgoing_visible"]) for v in merged_places.values()),
            place_silent_degree_sum=sum(len(v["incoming_silent"]) + len(v["outgoing_silent"]) for v in merged_places.values()),
            visible_transition_sum=sum(len(v["pred"]) + len(v["succ"]) for v in merged_visible.values()),
            silent_transition_sum=sum(len(v["pred"]) + len(v["succ"]) for v in merged_silent.values()),
        )

        self._model_cache = _ModelCache(
            place_ids,
            initial_places_by_type,
            tuple(silent),
            combos,
            {},
            stats
        )

        return self._model_cache

    # -------------------------
    # EVALUATION (OCPA semantics)
    # -------------------------
    def _evaluate(self, ocel):
        ocel = ocel or self.ocel
        prepared = self._prepare_log(ocel)

        model_enabled = {}
        for ctx, events in prepared["replay"].items():
            enabled = set()
            for e in events:
                enabled |= self._replay(e)
            model_enabled[ctx] = enabled

        precision_terms = []
        fitness_terms = []

        for e in prepared["events"]:
            ctx = prepared["ctx"][e]
            log = prepared["log"][ctx]
            model = model_enabled[ctx]

            overlap = log & model

            if not model or not overlap:
                fitness_terms.append(0)
                continue

            precision_terms.append(len(overlap) / len(model))
            fitness_terms.append(len(overlap) / len(log))

        return _EvaluationResult(
            precision=sum(precision_terms) / len(precision_terms) if precision_terms else 0,
            fitness=sum(fitness_terms) / len(fitness_terms),
        )

    # -------------------------
    # FAST REPLAY (COUNT BASED)
    # -------------------------
    def _replay(self, ev):
        mc = self._ensure_model_cache()
        state = self._initial_state(ev)

        q = deque([(state, 0)])
        visited = set()
        enabled = set()

        while q:
            state, i = q.popleft()
            key = (self._state_key(state), i)

            if key in visited:
                continue
            visited.add(key)

            if i == len(ev.binding_sequence):
                enabled |= self._enabled_labels(state)

            if i < len(ev.binding_sequence):
                for ns in self._fire_visible(state, ev.binding_sequence[i]):
                    q.append((ns, i + 1))
            else:
                for ns in self._fire_silent(state):
                    q.append((ns, i))

        return enabled

    def _initial_state(self, ev):
        mc = self._ensure_model_cache()
        state = {}

        for ot, tokens in ev.context_tokens_by_type:
            if not tokens:
                continue
            for p in mc.initial_places_by_type.get(ot, ()):
                state[p] = len(tokens)

        return state

    def _fire_visible(self, state, step):
        mc = self._ensure_model_cache()
        res = []

        for combo in mc.visible_combinations_by_label.get(step.label, ()):
            if all(self._enabled(state, t.input_places) for _, t in combo):
                ns = state.copy()
                for _, t in combo:
                    self._move(ns, t.input_places, t.output_places)
                res.append(ns)

        return res

    def _fire_silent(self, state):
        mc = self._ensure_model_cache()
        res = []

        for t in mc.silent_transitions:
            if self._enabled(state, t.input_places):
                ns = state.copy()
                self._move(ns, t.input_places, t.output_places)
                res.append(ns)

        return res

    def _enabled(self, state, places):
        return all(state.get(p, 0) > 0 for p in places)

    def _move(self, state, inp, out):
        for p in inp:
            state[p] = state.get(p, 0) - 1
            if state[p] <= 0:
                state.pop(p, None)
        for p in out:
            state[p] = state.get(p, 0) + 1

    def _enabled_labels(self, state):
        mc = self._ensure_model_cache()
        key = self._state_key(state)

        if key in mc.enabled_labels_cache:
            return mc.enabled_labels_cache[key]

        enabled = set()
        for label, combos in mc.visible_combinations_by_label.items():
            for combo in combos:
                if all(self._enabled(state, t.input_places) for _, t in combo):
                    enabled.add(label)
                    break

        mc.enabled_labels_cache[key] = frozenset(enabled)
        return mc.enabled_labels_cache[key]

    def _state_key(self, state):
        mc = self._ensure_model_cache()
        return tuple((mc.place_ids[p], c) for p, c in state.items())

    # -------------------------
    # LOG PREP (simplified)
    # -------------------------
    def _prepare_log(self, ocel):
        events = ocel.events
        rel = ocel.relations

        eid = ocel.event_id_column
        act = ocel.event_activity
        obj = ocel.object_id_column
        typ = ocel.object_type_column

        ordered = tuple(events[eid])
        act_map = dict(zip(events[eid], events[act]))

        event_tokens = defaultdict(list)
        for e, o, t in rel[[eid, obj, typ]].itertuples(index=False):
            event_tokens[e].append((t, o))

        ctx = {}
        log_enabled = defaultdict(set)
        replay = defaultdict(list)

        for e in ordered:
            context = defaultdict(Counter)

            for t, o in event_tokens[e]:
                context[t][()] += 1

            key = tuple(sorted((t, tuple(c.items())) for t, c in context.items()))
            ctx[e] = key
            log_enabled[key].add(act_map[e])

            replay[key].append(
                _ReplayEvent(
                    key,
                    (),
                    tuple((t, tuple(v)) for t, v in context.items()),
                )
            )

        return {
            "events": ordered,
            "ctx": ctx,
            "log": {k: frozenset(v) for k, v in log_enabled.items()},
            "replay": replay,
        }