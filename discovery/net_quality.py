from __future__ import annotations

from collections import Counter, defaultdict, deque
from dataclasses import dataclass
from itertools import product
import random
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
    transition_id: int
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
    # OCPA-style scope: all objects appearing in the preset of the event plus
    # the current event itself. These tokens are placed in the local initial
    # marking before the preset binding sequence is replayed.
    context_tokens_by_type: tuple[tuple[str, tuple[Token, ...]], ...]


@dataclass(frozen=True)
class _EvaluationResult:
    precision: float
    fitness: float


@dataclass(frozen=True)
class _ComplexityStats:
    place_count: int
    visible_transition_count: int
    silent_transition_count: int
    arc_count: int
    variable_arc_count: int
    visible_synchronization_penalty: int
    silent_synchronization_penalty: int


@dataclass
class _ModelCache:
    place_ids: dict[PlaceKey, int]
    initial_places_by_type: dict[str, tuple[PlaceKey, ...]]
    silent_transitions: tuple[_LocalTransition, ...]
    source_silent_transitions: tuple[_LocalTransition, ...]
    silent_transitions_by_input_place: dict[PlaceKey, tuple[_LocalTransition, ...]]
    visible_transitions_by_label_type: dict[str, dict[str, tuple[_LocalTransition, ...]]]
    enabled_labels_cache: dict[tuple, frozenset[str]]
    transition_enabled_cache: dict[tuple[tuple, int], bool]
    candidate_tokens_cache: dict[tuple[tuple, int], tuple[Token, ...]]
    complexity_stats: _ComplexityStats


class NetQuality:
    def __init__(
        self,
        ocpn: dict[str, Any],
        ocel: OCEL | None = None,
        *,
        max_nodes_per_replay: int | None = 1000,
        precision_context_sample_size: int | None = None,
        precision_context_depth: int | None = None,
        precision_context_length: int | None = None,
        random_seed: int | None = None,
    ) -> None:
        self.ocpn = ocpn
        self.ocel = ocel
        self.max_nodes_per_replay = max_nodes_per_replay
        self.precision_context_sample_size = precision_context_sample_size
        self.precision_context_depth = precision_context_depth
        self.precision_context_length = precision_context_length
        self.random_seed = random_seed

        self._model_cache: _ModelCache | None = None
        self._complexity_cache: float | None = None
        self._replay_cache: dict[tuple, frozenset[str]] = {}
        self._terminal_replay_cache: dict[tuple, tuple[Any, ...]] = {}

    # -------------------------
    # PUBLIC API
    # -------------------------
    def precision(self, ocel=None, *, show_progress=False):
        return self._evaluate(
            ocel,
            show_progress=show_progress,
            compute_precision=True,
            compute_fitness=False,
        ).precision

    def fitness(self, ocel=None, *, show_progress=False):
        return self._evaluate(
            ocel,
            show_progress=show_progress,
            compute_precision=False,
            compute_fitness=True,
        ).fitness

    def complexity(self) -> float:
        if self._complexity_cache is None:
            stats = self._ensure_model_cache().complexity_stats
            self._complexity_cache = (
                stats.place_count
                + stats.visible_transition_count
                + 2 * stats.silent_transition_count
                + stats.arc_count
                + stats.variable_arc_count
                + stats.visible_synchronization_penalty
                + 2 * stats.silent_synchronization_penalty
            )

        return self._complexity_cache

    # -------------------------
    # MODEL CACHE
    # -------------------------
    def _ensure_model_cache(self):
        if self._model_cache:
            return self._model_cache

        place_ids = {}
        silent = []
        visible = defaultdict(lambda: defaultdict(list))
        merged_visible = defaultdict(lambda: {"pre": set(), "post": set(), "types": set()})
        merged_silent = defaultdict(lambda: {"pre": set(), "post": set(), "types": set()})
        arc_count = 0
        variable_arc_count = 0

        initial_places_by_type = {}
        transition_id = 0

        for ot in self.ocpn["petri_nets"]:
            net, im, _ = self.ocpn["petri_nets"][ot]

            for p in net.places:
                pk = (ot, p)
                place_ids[pk] = len(place_ids)

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
                variable = any(
                    bool(getattr(arc, "variable", False))
                    for arc in tuple(t.in_arcs) + tuple(t.out_arcs)
                )

                lt = _LocalTransition(
                    transition_id=transition_id,
                    object_type=ot,
                    label=label,
                    input_places=inp,
                    output_places=out,
                    variable=variable,
                )
                transition_id += 1

                if label:
                    visible[label][ot].append(lt)
                    merged_visible[label]["pre"].update(inp)
                    merged_visible[label]["post"].update(out)
                    merged_visible[label]["types"].add(ot)
                else:
                    silent.append(lt)
                    merged_silent[t.name]["pre"].update(inp)
                    merged_silent[t.name]["post"].update(out)
                    merged_silent[t.name]["types"].add(ot)

            for arc in net.arcs:
                arc_count += 1
                if self._is_variable_arc(ot, arc):
                    variable_arc_count += 1

        visible_by_label_type = {
            label: {
                ot: tuple(transitions)
                for ot, transitions in by_type.items()
            }
            for label, by_type in visible.items()
        }

        source_silent = []
        silent_by_input_place = defaultdict(list)

        for t in silent:
            if not t.input_places:
                source_silent.append(t)
            else:
                # A transition may have multiple input places. We index it under
                # each distinct input place and deduplicate later in _fire_silent.
                for p in set(t.input_places):
                    silent_by_input_place[p].append(t)

        stats = _ComplexityStats(
            place_count=len(place_ids),
            visible_transition_count=len(merged_visible),
            silent_transition_count=len(merged_silent),
            arc_count=arc_count,
            variable_arc_count=variable_arc_count,
            visible_synchronization_penalty=self._synchronization_penalty(merged_visible),
            silent_synchronization_penalty=self._synchronization_penalty(merged_silent),
        )

        self._model_cache = _ModelCache(
            place_ids=place_ids,
            initial_places_by_type=initial_places_by_type,
            silent_transitions=tuple(silent),
            source_silent_transitions=tuple(source_silent),
            silent_transitions_by_input_place={
                p: tuple(transitions)
                for p, transitions in silent_by_input_place.items()
            },
            visible_transitions_by_label_type=visible_by_label_type,
            enabled_labels_cache={},
            transition_enabled_cache={},
            candidate_tokens_cache={},
            complexity_stats=stats,
        )

        return self._model_cache

    def _is_variable_arc(self, object_type: str, arc) -> bool:
        if bool(getattr(arc, "variable", False)) or bool(getattr(arc, "properties", {}).get("variable", False)):
            return True

        transition = None
        if isinstance(arc.source, PetriNet.Transition):
            transition = arc.source
        elif isinstance(arc.target, PetriNet.Transition):
            transition = arc.target

        if transition is None or transition.label is None:
            return False

        return bool(
            self.ocpn.get("double_arcs_on_activity", {})
            .get(object_type, {})
            .get(str(transition.label), False)
        )

    def _synchronization_penalty(self, transition_stats) -> int:
        penalty = 0

        for stats in transition_stats.values():
            object_type_count = len(stats["types"])
            if object_type_count < 2:
                continue

            incident_arc_count = len(stats["pre"]) + len(stats["post"])
            penalty += incident_arc_count * (
                object_type_count * (object_type_count - 1) // 2
            )

        return penalty

    # -------------------------
    # EVALUATION
    # -------------------------
    def _evaluate(
        self,
        ocel,
        show_progress=False,
        *,
        compute_precision=True,
        compute_fitness=True,
    ):
        ocel = ocel or self.ocel
        prepared = self._prepare_log(ocel, show_progress=show_progress)

        precision = 0.0
        fitness = 0.0
        model_enabled = None

        if compute_precision and self.precision_context_sample_size is not None:
            sample_counts = self._sample_precision_context_draw_counts(prepared)
            model_enabled = self._build_model_enabled_by_context(
                prepared,
                selected_contexts=sample_counts,
                show_progress=show_progress,
                progress_desc="Replaying sampled contexts",
            )

            weighted_precision = 0.0
            sampled_precision_events = 0
            sampled_iter = sample_counts.items()
            if show_progress:
                sampled_iter = tqdm(
                    sampled_iter,
                    total=len(sample_counts),
                    desc="Aggregating sampled precision",
                    file=sys.stdout,
                )

            for ctx, draw_count in sampled_iter:
                log = prepared["log"].get(ctx, frozenset())
                model = model_enabled.get(ctx, frozenset())
                overlap = log & model

                # Preserve the existing precision semantics: contexts with an
                # empty model-enabled set or zero overlap are skipped instead of
                # contributing an explicit zero.
                if not model or not overlap:
                    continue

                weighted_precision += draw_count * (len(overlap) / len(model))
                sampled_precision_events += draw_count

            precision = (
                weighted_precision / sampled_precision_events
                if sampled_precision_events
                else 0.0
            )

        if compute_fitness or (compute_precision and model_enabled is None):
            model_enabled = self._build_model_enabled_by_context(
                prepared,
                show_progress=show_progress,
            )

        if compute_precision and model_enabled is not None and self.precision_context_sample_size is None:
            context_weights = self._context_weights(prepared)
            weighted_precision = 0.0
            precision_weight = 0

            context_iter = context_weights.items()
            if show_progress:
                context_iter = tqdm(
                    context_iter,
                    total=len(context_weights),
                    desc="Aggregating scores",
                    file=sys.stdout,
                )

            for ctx, weight in context_iter:
                log = prepared["log"].get(ctx, frozenset())
                model = model_enabled.get(ctx, frozenset())
                overlap = log & model

                if not model or not overlap:
                    continue

                weighted_precision += weight * (len(overlap) / len(model))
                precision_weight += weight

            precision = weighted_precision / precision_weight if precision_weight else 0.0

        if compute_fitness and model_enabled is not None:
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

                fitness_terms.append(len(overlap) / len(log))

            fitness = sum(fitness_terms) / len(fitness_terms) if fitness_terms else 0.0

        result = _EvaluationResult(
            precision=precision,
            fitness=fitness,
        )
        if show_progress:
            print("Done.")
        return result

    def _build_model_enabled_by_context(
        self,
        prepared,
        *,
        selected_contexts=None,
        show_progress=False,
        progress_desc="Replaying contexts",
    ):
        if selected_contexts is None:
            context_items = tuple(prepared["replay"].items())
        else:
            selected_context_set = set(selected_contexts)
            context_items = tuple(
                (ctx, events)
                for ctx, events in prepared["replay"].items()
                if ctx in selected_context_set
            )

        replay_iter = context_items
        if show_progress:
            replay_iter = tqdm(
                context_items,
                total=len(context_items),
                desc=progress_desc,
                file=sys.stdout,
            )

        model_enabled = {}

        # OCPA-style: calculate model-enabled activities per event replay, then
        # union the results for events that share the same context.
        for ctx, events in replay_iter:
            enabled = set()

            for ev in events:
                enabled |= self._replay(ev)

            model_enabled[ctx] = enabled

        return model_enabled

    def _context_weights(self, prepared):
        return {
            ctx: len(events)
            for ctx, events in prepared["replay"].items()
        }

    def _sample_precision_context_draw_counts(self, prepared):
        sample_size = self.precision_context_sample_size
        if sample_size is None:
            return {}
        if sample_size <= 0:
            raise ValueError("precision_context_sample_size must be positive.")

        context_weights = self._context_weights(prepared)
        if not context_weights:
            return {}

        contexts = tuple(context_weights)
        weights = tuple(context_weights[ctx] for ctx in contexts)
        rng = random.Random(self.random_seed)
        return Counter(rng.choices(contexts, weights=weights, k=sample_size))

    # -------------------------
    # OCPA-STYLE REPLAY
    # -------------------------
    def _replay_terminal_states(self, ev):
        replay_key = self._replay_signature(ev)

        if replay_key in self._terminal_replay_cache:
            return self._terminal_replay_cache[replay_key]

        state = self._initial_state(ev)
        q = deque([(state, 0)])
        visited = set()
        terminal_states = {}
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
                terminal_states.setdefault(key[0], state)

            for ns in self._fire_silent(state):
                q.append((ns, i))

            if i < len(ev.binding_sequence):
                for ns in self._fire_visible(state, ev.binding_sequence[i]):
                    q.append((ns, i + 1))

        result = tuple(terminal_states.values())
        self._terminal_replay_cache[replay_key] = result
        return result

    def _replay(self, ev):
        replay_key = self._replay_signature(ev)

        if replay_key in self._replay_cache:
            return self._replay_cache[replay_key]

        enabled = set()
        for state in self._replay_terminal_states(ev):
            enabled |= self._enabled_labels(state)

        result = frozenset(enabled)
        self._replay_cache[replay_key] = result
        return result

    def _replay_signature(self, ev):
        def objects_sig(objects_by_type):
            return tuple(
                (ot, tuple(tokens))
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

        # OCPA-style local initial state: all objects in the context scope start
        # in their object type's initial place(s). The visible preset sequence is
        # then replayed on this local state.
        for ot, tokens in ev.context_tokens_by_type:
            tokens_by_type[ot].update(tokens)

        state = {}

        for ot, tokens in tokens_by_type.items():
            if not tokens:
                continue

            for p in mc.initial_places_by_type.get(ot, ()):
                state[p] = list(tokens)

        return self._canonical_state(state)

    def _fire_visible(self, state, step):
        mc = self._ensure_model_cache()
        by_type = mc.visible_transitions_by_label_type.get(step.label)

        if not by_type:
            return []

        objects_by_type = {
            ot: tuple(tokens)
            for ot, tokens in step.objects_by_type
            if tokens
        }

        if not objects_by_type:
            return []

        required_types = tuple(sorted(objects_by_type))

        # For the PM4Py dict-of-local-nets representation, a visible binding is
        # replayed by synchronizing local transitions with the same label for the
        # object types that actually occur in the event binding.
        if any(ot not in by_type for ot in required_types):
            return []

        res = []
        transition_choices = [by_type[ot] for ot in required_types]

        for combo in product(*transition_choices):
            if all(
                self._tokens_enabled(state, t.input_places, objects_by_type[ot])
                for ot, t in zip(required_types, combo)
            ):
                ns = state

                for ot, t in zip(required_types, combo):
                    ns = self._move_tokens(
                        ns,
                        t.input_places,
                        t.output_places,
                        objects_by_type[ot],
                    )

                res.append(ns)

        return res

    def _fire_silent(self, state):
        mc = self._ensure_model_cache()
        state_key = self._state_key(state)

        # Exact speed-up:
        # Instead of scanning every silent transition in the model, only consider
        # silent transitions that are indexed by at least one currently marked
        # input place. Source silent transitions are included separately.
        candidate_transitions = set(mc.source_silent_transitions)

        for p in state:
            candidate_transitions.update(
                mc.silent_transitions_by_input_place.get(p, ())
            )

        # Preserve the original model traversal order as much as possible. This
        # matters when max_nodes_per_replay truncates the BFS.
        ordered_candidates = sorted(
            candidate_transitions,
            key=lambda t: t.transition_id,
        )

        res = []

        for t in ordered_candidates:
            candidates = self._candidate_tokens_for_transition(
                state,
                t,
                state_key=state_key,
            )

            for token in candidates:
                ns = self._move_tokens(
                    state,
                    t.input_places,
                    t.output_places,
                    (token,),
                )
                if ns != state:
                    res.append(ns)

        return res

    def _enabled_labels(self, state):
        mc = self._ensure_model_cache()
        key = self._state_key(state)

        if key in mc.enabled_labels_cache:
            return mc.enabled_labels_cache[key]

        enabled = set()

        # Exact speed-up:
        # The previous implementation checked the Cartesian product of local
        # transitions across object types. For label-level enabledness, this is
        # equivalent to checking whether every object type has at least one
        # locally enabled transition.
        #
        # Old cost per label:
        #     O(product_o number_of_local_transitions(label, o))
        #
        # New cost per label:
        #     O(sum_o number_of_local_transitions(label, o))
        for label, by_type in mc.visible_transitions_by_label_type.items():
            label_enabled = True

            for ot in sorted(by_type):
                transitions = by_type[ot]

                if not any(
                    self._has_some_binding(state, t, state_key=key)
                    for t in transitions
                ):
                    label_enabled = False
                    break

            if label_enabled:
                enabled.add(label)

        mc.enabled_labels_cache[key] = frozenset(enabled)
        return mc.enabled_labels_cache[key]

    def _has_some_binding(
        self,
        state,
        transition: _LocalTransition,
        *,
        state_key=None,
    ) -> bool:
        # Source transitions are considered enabled, matching Petri-net
        # semantics and OCPA's model_enabled check for transitions without input
        # arcs.
        if not transition.input_places:
            return True

        mc = self._ensure_model_cache()
        if state_key is None:
            state_key = self._state_key(state)

        cache_key = (state_key, transition.transition_id)

        if cache_key in mc.transition_enabled_cache:
            return mc.transition_enabled_cache[cache_key]

        enabled = bool(
            self._candidate_tokens_for_transition(
                state,
                transition,
                state_key=state_key,
            )
        )

        mc.transition_enabled_cache[cache_key] = enabled
        return enabled

    def _candidate_tokens_for_transition(
        self,
        state,
        transition: _LocalTransition,
        *,
        state_key=None,
    ):
        if not transition.input_places:
            return tuple()

        mc = self._ensure_model_cache()
        if state_key is None:
            state_key = self._state_key(state)

        cache_key = (state_key, transition.transition_id)

        if cache_key in mc.candidate_tokens_cache:
            return mc.candidate_tokens_cache[cache_key]

        token_sets = []
        for p in transition.input_places:
            tokens = set(state.get(p, ()))
            if not tokens:
                mc.candidate_tokens_cache[cache_key] = tuple()
                return tuple()
            token_sets.append(tokens)

        candidates = set.intersection(*token_sets) if token_sets else set()
        result = tuple(sorted(candidates, key=self._token_sort_key))

        mc.candidate_tokens_cache[cache_key] = result
        return result

    def _tokens_enabled(self, state, places, tokens: tuple[Token, ...]) -> bool:
        if not tokens:
            return True

        needed = Counter(tokens)

        for p in places:
            available = Counter(state.get(p, ()))
            if any(available[token] < count for token, count in needed.items()):
                return False

        return True

    def _move_tokens(self, state, inp, out, tokens: tuple[Token, ...]):
        if not tokens:
            return state

        ns = {p: list(ts) for p, ts in state.items()}
        needed = Counter(tokens)

        for p in inp:
            remaining = Counter(ns.get(p, ()))
            remaining.subtract(needed)
            ns[p] = list(remaining.elements())
            if not ns[p]:
                ns.pop(p, None)

        for p in out:
            ns.setdefault(p, [])
            ns[p].extend(tokens)

        return self._canonical_state(ns)

    def _canonical_state(self, state):
        return {
            p: tuple(sorted(tokens, key=self._token_sort_key))
            for p, tokens in state.items()
            if tokens
        }

    def _state_key(self, state):
        mc = self._ensure_model_cache()
        return tuple(
            sorted(
                (mc.place_ids[p], tuple(sorted(tokens, key=self._token_sort_key)))
                for p, tokens in state.items()
            )
        )

    def _token_sort_key(self, token: Token):
        ot, oid = token
        return (str(ot), repr(oid))

    def _group_event_tokens_by_type(self, tokens):
        by_type = defaultdict(list)

        for ot, oid in tokens:
            by_type[str(ot)].append((str(ot), oid))

        return tuple(
            (ot, tuple(sorted(grouped_tokens, key=self._token_sort_key)))
            for ot, grouped_tokens in sorted(by_type.items())
        )

    # -------------------------
    # OCPA-STYLE LOG PREPARATION
    # -------------------------
    def _prepare_log(self, ocel, show_progress=False):
        if ocel is None:
            raise ValueError("No OCEL was provided.")
        if self.precision_context_depth is not None and self.precision_context_depth < 0:
            raise ValueError("precision_context_depth must be non-negative.")
        if self.precision_context_length is not None and self.precision_context_length < 0:
            raise ValueError("precision_context_length must be non-negative.")

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
            if e in event_order_index:
                event_tokens[e].append((str(t), o))

        # Build per-object event histories and the event-object graph. The EOG
        # has an edge e1 -> e2 when e1 and e2 are consecutive events of at least
        # one common object. The preset of e is the transitive ancestor set in
        # this graph, as in OCPA's nx.ancestors(eog, e) computation.
        object_events = defaultdict(list)
        for e in ordered:
            for token in event_tokens.get(e, ()):  # token = (object_type, object_id)
                object_events[token].append(e)

        reverse_adj = {e: set() for e in ordered}
        for history in object_events.values():
            history = sorted(history, key=event_order_index.__getitem__)
            for prev_e, next_e in zip(history, history[1:]):
                reverse_adj[next_e].add(prev_e)

        preset_depth = self.precision_context_depth
        presets = {}
        if preset_depth is None:
            for e in ordered:
                ancestors = set()
                for pred in reverse_adj[e]:
                    ancestors.add(pred)
                    ancestors.update(presets.get(pred, ()))
                presets[e] = frozenset(ancestors)
        else:
            ancestor_depths = {}
            for e in ordered:
                depths = {}
                if preset_depth > 0:
                    for pred in reverse_adj[e]:
                        depths[pred] = 1
                        for ancestor, distance in ancestor_depths.get(pred, {}).items():
                            next_distance = distance + 1
                            if next_distance > preset_depth:
                                continue

                            current_distance = depths.get(ancestor)
                            if current_distance is None or next_distance < current_distance:
                                depths[ancestor] = next_distance

                ancestor_depths[e] = depths
                presets[e] = frozenset(depths)

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

        ordered_iter = ordered
        if show_progress:
            desc = "Preparing OCPA-style event contexts"
            if self.precision_context_depth is not None:
                desc += f" (d={self.precision_context_depth})"
            if self.precision_context_length is not None:
                desc += f" (l={self.precision_context_length})"
            ordered_iter = tqdm(
                ordered,
                total=len(ordered),
                desc=desc,
                file=sys.stdout,
            )

        for e in ordered_iter:
            preset_events = presets[e]
            preset_ordered = tuple(sorted(preset_events, key=event_order_index.__getitem__))

            # OCPA-style context object scope:
            # all objects appearing in the preset plus the current event itself.
            context_tokens = set(event_tokens.get(e, ()))
            for pe in preset_events:
                context_tokens.update(event_tokens.get(pe, ()))

            context = defaultdict(Counter)

            for token in context_tokens:
                ot, _ = token
                prefix = tuple(
                    act_map[pe]
                    for pe in object_events.get(token, ())
                    if pe in preset_events
                )
                if self.precision_context_length is not None:
                    if self.precision_context_length == 0:
                        prefix = ()
                    else:
                        prefix = prefix[-self.precision_context_length:]
                context[ot][prefix] += 1

            key = tuple(
                sorted(
                    (
                        ot,
                        tuple(sorted(counter.items())),
                    )
                    for ot, counter in context.items()
                )
            )

            binding_sequence = tuple(
                _BindingStep(
                    label=act_map[pe],
                    objects_by_type=objects_by_type_for_event(pe),
                )
                for pe in preset_ordered
            )

            ctx[e] = key
            log_enabled[key].add(act_map[e])

            replay[key].append(
                _ReplayEvent(
                    context_key=key,
                    binding_sequence=binding_sequence,
                    context_tokens_by_type=self._group_event_tokens_by_type(context_tokens),
                )
            )

        return {
            "events": ordered,
            "ctx": ctx,
            "log": {
                k: frozenset(v)
                for k, v in log_enabled.items()
            },
            "replay": replay,
        }
