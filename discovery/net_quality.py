from __future__ import annotations

from collections import Counter, defaultdict, deque
from dataclasses import dataclass
from itertools import product
import sys
from typing import Any

from pm4py.objects.ocel.obj import OCEL
from pm4py.objects.petri_net.obj import PetriNet
from tqdm import tqdm


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
        return {ot for ot, tokens in self.objects_by_type if tokens}


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
    visible_transitions_by_label_type: dict[str, dict[str, tuple[_LocalTransition, ...]]]
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
        self.max_nodes_per_replay = 5000

        # Private tuning parameter; public API stays unchanged.
        # Larger values are more faithful but slower.
        self._history_window = 5

        self._model_cache: _ModelCache | None = None
        self._complexity_cache: dict[tuple[float, float], float] = {}
        self._replay_cache: dict[tuple, frozenset[str]] = {}

    # -------------------------
    # PUBLIC API
    # -------------------------
    def precision(self, ocel=None, *, show_progress=False):
        return self._evaluate(ocel, show_progress=show_progress).precision

    def fitness(self, ocel=None, *, show_progress=False):
        return self._evaluate(ocel, show_progress=show_progress).fitness

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

            initial_places_by_type[ot] = tuple(
                (ot, p)
                for p, c in im.items()
                if c > 0
            )

            for t in net.transitions:
                inp = tuple(
                    (ot, arc.source)
                    for arc in t.in_arcs
                    if isinstance(arc.source, PetriNet.Place)
                )
                out = tuple(
                    (ot, arc.target)
                    for arc in t.out_arcs
                    if isinstance(arc.target, PetriNet.Place)
                )

                label = str(t.label) if t.label else None
                lt = _LocalTransition(
                    object_type=ot,
                    label=label,
                    input_places=inp,
                    output_places=out,
                    variable=False,
                )

                if label:
                    visible[label][ot].append(lt)
                else:
                    silent.append(lt)

            # Complexity tracking
            for arc in net.arcs:
                if isinstance(arc.source, PetriNet.Place):
                    pk = (ot, arc.source)

                    if arc.target.label:
                        label = str(arc.target.label)
                        merged_places[pk]["outgoing_visible"].add(label)
                        merged_visible[label]["pred"].add(pk)
                        merged_visible[label]["types"].add(ot)
                    else:
                        merged_places[pk]["outgoing_silent"].add(arc.target.name)
                        merged_silent[arc.target.name]["pred"].add(pk)
                        merged_silent[arc.target.name]["types"].add(ot)

                else:
                    pk = (ot, arc.target)

                    if arc.source.label:
                        label = str(arc.source.label)
                        merged_places[pk]["incoming_visible"].add(label)
                        merged_visible[label]["succ"].add(pk)
                        merged_visible[label]["types"].add(ot)
                    else:
                        merged_places[pk]["incoming_silent"].add(arc.source.name)
                        merged_silent[arc.source.name]["succ"].add(pk)
                        merged_silent[arc.source.name]["types"].add(ot)

        visible_by_label_type = {
            label: {
                ot: tuple(transitions)
                for ot, transitions in by_type.items()
            }
            for label, by_type in visible.items()
        }

        combos = {}
        for label, by_type in visible.items():
            combos[label] = tuple(
                tuple(zip(by_type.keys(), c))
                for c in product(*by_type.values())
            )

        stats = _ComplexityStats(
            place_visible_degree_sum=sum(
                len(v["incoming_visible"]) + len(v["outgoing_visible"])
                for v in merged_places.values()
            ),
            place_silent_degree_sum=sum(
                len(v["incoming_silent"]) + len(v["outgoing_silent"])
                for v in merged_places.values()
            ),
            visible_transition_sum=sum(
                len(v["pred"]) + len(v["succ"])
                for v in merged_visible.values()
            ),
            silent_transition_sum=sum(
                len(v["pred"]) + len(v["succ"])
                for v in merged_silent.values()
            ),
        )

        self._model_cache = _ModelCache(
            place_ids=place_ids,
            initial_places_by_type=initial_places_by_type,
            silent_transitions=tuple(silent),
            visible_transitions_by_label_type=visible_by_label_type,
            visible_combinations_by_label=combos,
            enabled_labels_cache={},
            complexity_stats=stats,
        )

        return self._model_cache

    # -------------------------
    # EVALUATION
    # -------------------------
    def _evaluate(self, ocel, show_progress=False):
        ocel = ocel or self.ocel
        prepared = self._prepare_log(ocel, show_progress=show_progress)

        model_enabled = {}

        replay_iter = prepared["replay"].items()
        if show_progress:
            replay_iter = tqdm(
                replay_iter,
                total=len(prepared["replay"]),
                desc="Replaying contexts",
                file=sys.stdout,
            )

        for ctx, events in replay_iter:
            enabled = set()

            for ev in events:
                enabled |= self._replay(ev)

            model_enabled[ctx] = enabled

        precision_terms = []
        fitness_terms = []

        event_iter = prepared["events"]
        if show_progress:
            event_iter = tqdm(
                prepared["events"],
                total=len(prepared["events"]),
                desc="Aggregating scores",
                file=sys.stdout,
            )

        for e in event_iter:
            ctx = prepared["ctx"][e]
            log = prepared["log"][ctx]
            model = model_enabled[ctx]

            overlap = log & model

            if not model or not overlap:
                fitness_terms.append(0)
                continue

            precision_terms.append(len(overlap) / len(model))
            fitness_terms.append(len(overlap) / len(log))

        result = _EvaluationResult(
            precision=sum(precision_terms) / len(precision_terms)
            if precision_terms
            else 0,
            fitness=sum(fitness_terms) / len(fitness_terms)
            if fitness_terms
            else 0,
        )
        if show_progress:
            print("Done.")
        return result

    # -------------------------
    # BOUNDED COUNT-BASED REPLAY
    # -------------------------
    def _replay(self, ev):
        replay_key = self._replay_signature(ev)

        if replay_key in self._replay_cache:
            return self._replay_cache[replay_key]

        state = self._initial_state(ev)

        q = deque([(state, 0)])
        visited = set()
        enabled = set()
        explored_nodes = 0

        while q:
            state, i = q.popleft()
            key = (self._state_key(state), i)

            if key in visited:
                continue

            if (
                self.max_nodes_per_replay is not None
                and explored_nodes >= self.max_nodes_per_replay
            ):
                break

            visited.add(key)
            explored_nodes += 1

            if i == len(ev.binding_sequence):
                enabled |= self._enabled_labels(state, ev.context_tokens_by_type)

            if i < len(ev.binding_sequence):
                for ns in self._fire_visible(state, ev.binding_sequence[i]):
                    q.append((ns, i + 1))
            else:
                for ns in self._fire_silent(state):
                    q.append((ns, i))

        result = frozenset(enabled)
        self._replay_cache[replay_key] = result
        return result

    def _replay_signature(self, ev):
        def objects_sig(objects_by_type):
            return tuple(
                (ot, len(tokens))
                for ot, tokens in objects_by_type
                if tokens
            )

        return (
            ev.context_key,
            tuple(
                (step.label, objects_sig(step.objects_by_type))
                for step in ev.binding_sequence
            ),
            objects_sig(ev.context_tokens_by_type),
            self.max_nodes_per_replay,
        )

    def _initial_state(self, ev):
        mc = self._ensure_model_cache()
        tokens_by_type = defaultdict(set)

        for ot, tokens in ev.context_tokens_by_type:
            tokens_by_type[ot].update(tokens)

        for step in ev.binding_sequence:
            for ot, tokens in step.objects_by_type:
                tokens_by_type[ot].update(tokens)

        state = {}

        for ot, tokens in tokens_by_type.items():
            if not tokens:
                continue

            for p in mc.initial_places_by_type.get(ot, ()):
                state[p] = len(tokens)

        return state

    def _fire_visible(self, state, step):
        mc = self._ensure_model_cache()
        by_type = mc.visible_transitions_by_label_type.get(step.label)

        if not by_type:
            return []

        object_counts = {
            ot: len(tokens)
            for ot, tokens in step.objects_by_type
            if tokens
        }

        if not object_counts:
            return []

        required_types = tuple(sorted(object_counts))

        # Approximation:
        # every object type present in the log binding must have a local
        # transition for this label. Object types not present in the binding
        # are ignored, which makes this tolerant of variable participation.
        if any(ot not in by_type for ot in required_types):
            return []

        res = []
        transition_choices = [by_type[ot] for ot in required_types]

        for combo in product(*transition_choices):
            if all(
                self._enabled(state, t.input_places, object_counts[ot])
                for ot, t in zip(required_types, combo)
            ):
                ns = state.copy()

                for ot, t in zip(required_types, combo):
                    self._move(
                        ns,
                        t.input_places,
                        t.output_places,
                        amount=object_counts[ot],
                    )

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

    def _enabled(self, state, places, amount: int = 1):
        if amount <= 0:
            return True

        return all(state.get(p, 0) >= amount for p in places)

    def _move(self, state, inp, out, amount: int = 1):
        if amount <= 0:
            return

        for p in inp:
            state[p] = state.get(p, 0) - amount

            if state[p] <= 0:
                state.pop(p, None)

        for p in out:
            state[p] = state.get(p, 0) + amount

    def _enabled_labels(self, state, context_tokens_by_type):
        mc = self._ensure_model_cache()

        object_counts = tuple(
            sorted(
                (ot, len(tokens))
                for ot, tokens in context_tokens_by_type
                if tokens
            )
        )

        key = (self._state_key(state), object_counts)

        if key in mc.enabled_labels_cache:
            return mc.enabled_labels_cache[key]

        if not object_counts:
            mc.enabled_labels_cache[key] = frozenset()
            return mc.enabled_labels_cache[key]

        required_types = tuple(ot for ot, _ in object_counts)
        counts_by_type = dict(object_counts)

        enabled = set()

        for label, by_type in mc.visible_transitions_by_label_type.items():
            if any(ot not in by_type for ot in required_types):
                continue

            transition_choices = [by_type[ot] for ot in required_types]

            for combo in product(*transition_choices):
                if all(
                    self._enabled(state, t.input_places, counts_by_type[ot])
                    for ot, t in zip(required_types, combo)
                ):
                    enabled.add(label)
                    break

        mc.enabled_labels_cache[key] = frozenset(enabled)
        return mc.enabled_labels_cache[key]

    def _state_key(self, state):
        mc = self._ensure_model_cache()
        return tuple(
            sorted(
                (mc.place_ids[p], c)
                for p, c in state.items()
            )
        )

    def _group_event_tokens_by_type(self, tokens):
        by_type = defaultdict(list)

        for ot, oid in tokens:
            by_type[ot].append((ot, oid))

        return tuple(
            (ot, tuple(grouped_tokens))
            for ot, grouped_tokens in sorted(by_type.items())
        )

    # -------------------------
    # LOG PREPARATION
    # -------------------------
    def _prepare_log(self, ocel, show_progress=False):
        if ocel is None:
            raise ValueError("No OCEL was provided.")

        events = ocel.events
        rel = ocel.relations

        eid = ocel.event_id_column
        act = ocel.event_activity
        obj = ocel.object_id_column
        typ = ocel.object_type_column

        ts_col = getattr(ocel, "event_timestamp", None)

        if ts_col is not None and ts_col in events.columns:
            ordered = tuple(events.sort_values(ts_col, kind="stable")[eid])
        else:
            ordered = tuple(events[eid])

        event_order_index = {
            event_id: i
            for i, event_id in enumerate(ordered)
        }

        act_map = {
            event_id: str(activity)
            for event_id, activity in zip(events[eid], events[act])
        }

        event_tokens = defaultdict(list)

        for e, o, t in rel[[eid, obj, typ]].itertuples(index=False):
            event_tokens[e].append((str(t), o))

        objects_by_type_cache = {}

        def objects_by_type_for_event(event_id):
            if event_id not in objects_by_type_cache:
                objects_by_type_cache[event_id] = self._group_event_tokens_by_type(
                    event_tokens.get(event_id, ())
                )
            return objects_by_type_cache[event_id]

        ctx = {}
        log_enabled = defaultdict(set)
        replay = defaultdict(list)

        # Recent history per concrete object.
        seen_by_object = defaultdict(lambda: deque(maxlen=self._history_window))

        ordered_iter = ordered
        if show_progress:
            ordered_iter = tqdm(
                ordered,
                total=len(ordered),
                desc="Preparing event contexts",
                file=sys.stdout,
            )

        for e in ordered_iter:
            current_tokens = event_tokens.get(e, ())
            current_objects_by_type = objects_by_type_for_event(e)

            context = defaultdict(Counter)
            pred_events = set()

            for ot, oid in current_tokens:
                history = tuple(seen_by_object[(ot, oid)])
                recent_history = history[-self._history_window:]

                pred_events.update(recent_history)

                history_labels = tuple(
                    act_map[pe]
                    for pe in recent_history
                )

                context[ot][history_labels] += 1

            key = tuple(
                sorted(
                    (
                        ot,
                        tuple(sorted(counter.items())),
                    )
                    for ot, counter in context.items()
                )
            )

            pred_events_ordered = tuple(
                sorted(
                    pred_events,
                    key=event_order_index.__getitem__,
                )
            )

            binding_sequence = tuple(
                _BindingStep(
                    label=act_map[pe],
                    objects_by_type=objects_by_type_for_event(pe),
                )
                for pe in pred_events_ordered
            )

            ctx[e] = key
            log_enabled[key].add(act_map[e])

            replay[key].append(
                _ReplayEvent(
                    context_key=key,
                    binding_sequence=binding_sequence,
                    context_tokens_by_type=current_objects_by_type,
                )
            )

            # Update histories only after preparing replay for this event.
            for ot, oid in current_tokens:
                seen_by_object[(ot, oid)].append(e)

        return {
            "events": ordered,
            "ctx": ctx,
            "log": {
                k: frozenset(v)
                for k, v in log_enabled.items()
            },
            "replay": replay,
        }
