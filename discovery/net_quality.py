from __future__ import annotations

from bisect import bisect_left
from collections import Counter, defaultdict, deque
from dataclasses import dataclass
from itertools import product
import os
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
    local_prefix_states_by_token: tuple[
        tuple[Token, tuple[tuple[PlaceKey, ...], ...]],
        ...
    ] = ()
    replay_limit_exceeded: bool = False


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
        random_seed: int | None = None,
    ) -> None:
        self.ocpn = ocpn
        self.ocel = ocel
        self.max_nodes_per_replay = max_nodes_per_replay
        self.precision_context_sample_size = precision_context_sample_size
        self.precision_context_depth = precision_context_depth
        self.random_seed = random_seed

        self._model_cache: _ModelCache | None = None
        self._complexity_cache: float | None = None
        self._replay_cache: dict[tuple, frozenset[str]] = {}
        self._terminal_replay_cache: dict[tuple, tuple[Any, ...]] = {}
        self._terminal_replay_limit_hit_cache: set[tuple] = set()
        self._local_silent_closure_cache: dict[tuple, tuple[tuple[tuple[PlaceKey, ...], ...], bool]] = {}

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

    def _debug_precision(self, message, **data):
        if not os.environ.get("NET_QUALITY_DEBUG"):
            return

        details = ", ".join(
            f"{key}={value!r}"
            for key, value in data.items()
        )
        print(f"[precision-debug] {message}" + (f": {details}" if details else ""), flush=True)

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
        self._debug_precision(
            "evaluate start",
            compute_precision=compute_precision,
            compute_fitness=compute_fitness,
            sample_size=self.precision_context_sample_size,
            max_nodes_per_replay=self.max_nodes_per_replay,
            events=len(getattr(ocel, "events", ())) if ocel is not None else None,
            relations=len(getattr(ocel, "relations", ())) if ocel is not None else None,
        )
        precision_sampling_disabled = (
            compute_precision
            and self.precision_context_sample_size == 0
        )
        if precision_sampling_disabled and not compute_fitness:
            if show_progress:
                print("Precision context sampling disabled; skipping precision replay.")
            return _EvaluationResult(
                precision=0.0,
                fitness=0.0,
            )

        sampled_precision = (
            compute_precision
            and self.precision_context_sample_size is not None
            and self.precision_context_sample_size > 0
        )
        if (
            compute_precision
            and self.precision_context_sample_size is not None
            and self.precision_context_sample_size < 0
        ):
            raise ValueError("precision_context_sample_size must be positive.")

        sampled_event_prepared = sampled_precision and not compute_fitness
        if sampled_event_prepared:
            prepared = self._prepare_sampled_precision_log(
                ocel,
                self.precision_context_sample_size,
                show_progress=show_progress,
            )
        else:
            prepared = self._prepare_log(
                ocel,
                show_progress=show_progress,
                materialize_replay=not (
                    compute_precision
                    and self.precision_context_sample_size is not None
                    and not compute_fitness
                ),
            )
        context_weights_for_debug = self._context_weights(prepared)
        self._debug_precision(
            "prepared log",
            sampled_event_prepared=sampled_event_prepared,
            events=len(prepared.get("events", ())),
            contexts=len(context_weights_for_debug),
            draws=sum(context_weights_for_debug.values()),
            replay_materialized=len(prepared.get("replay", {})),
        )

        precision = 0.0
        fitness = 0.0
        model_enabled = None

        if (
            compute_precision
            and self.precision_context_sample_size is not None
            and self.precision_context_sample_size > 0
        ):
            sample_counts = (
                self._context_weights(prepared)
                if sampled_event_prepared
                else self._sample_precision_context_draw_counts(prepared)
            )
            model_enabled = self._build_model_enabled_by_context(
                prepared,
                selected_contexts=sample_counts,
                show_progress=show_progress,
                progress_desc="Replaying sampled contexts",
            )
            self._debug_model_enabled_summary(prepared, model_enabled, sample_counts)

            weighted_precision = 0.0
            sampled_precision_events = 0
            replay_limited_contexts = set(prepared.get("replay_limited_contexts", ()))
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

                if not model:
                    continue

                weighted_precision += draw_count * (len(overlap) / len(model))
                sampled_precision_events += draw_count

            precision = (
                weighted_precision / sampled_precision_events
                if sampled_precision_events
                else 0.0
            )
            self._debug_precision(
                "sampled precision aggregate",
                numerator=weighted_precision,
                denominator=sampled_precision_events,
                precision=precision,
                replay_limited_contexts=len(replay_limited_contexts),
            )

        if compute_fitness or (
            compute_precision
            and model_enabled is None
            and not precision_sampling_disabled
        ):
            model_enabled = self._build_model_enabled_by_context(
                prepared,
                show_progress=show_progress,
            )
            self._debug_model_enabled_summary(
                prepared,
                model_enabled,
                self._context_weights(prepared),
            )

        if (
            compute_precision
            and model_enabled is not None
            and self.precision_context_sample_size is None
        ):
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

                if not model:
                    continue

                weighted_precision += weight * (len(overlap) / len(model))
                precision_weight += weight

            precision = weighted_precision / precision_weight if precision_weight else 0.0
            self._debug_precision(
                "exact precision aggregate",
                numerator=weighted_precision,
                denominator=precision_weight,
                precision=precision,
                replay_limited_contexts=len(prepared.get("replay_limited_contexts", ())),
            )

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

    def _debug_model_enabled_summary(self, prepared, model_enabled, context_weights):
        if not os.environ.get("NET_QUALITY_DEBUG"):
            return

        empty_model = 0
        zero_overlap = 0
        overlap = 0
        weighted_empty_model = 0
        weighted_zero_overlap = 0
        weighted_overlap = 0
        model_label_counts = Counter()
        log_label_counts = Counter()

        for ctx, weight in context_weights.items():
            log_enabled = prepared.get("log", {}).get(ctx, frozenset())
            model = model_enabled.get(ctx, frozenset())
            log_label_counts.update(log_enabled)
            model_label_counts.update(model)

            if not model:
                empty_model += 1
                weighted_empty_model += weight
            elif not (log_enabled & model):
                zero_overlap += 1
                weighted_zero_overlap += weight
            else:
                overlap += 1
                weighted_overlap += weight

        self._debug_precision(
            "model enabled summary",
            contexts=len(context_weights),
            empty_model_contexts=empty_model,
            zero_overlap_contexts=zero_overlap,
            overlap_contexts=overlap,
            weighted_empty_model=weighted_empty_model,
            weighted_zero_overlap=weighted_zero_overlap,
            weighted_overlap=weighted_overlap,
            replay_limited_contexts=len(prepared.get("replay_limited_contexts", ())),
            top_log_labels=log_label_counts.most_common(10),
            top_model_labels=model_label_counts.most_common(10),
        )

    def _build_model_enabled_by_context(
        self,
        prepared,
        *,
        selected_contexts=None,
        show_progress=False,
        progress_desc="Replaying contexts",
    ):
        replay = self._materialize_replay_events(
            prepared,
            selected_contexts=selected_contexts,
        )
        if selected_contexts is None:
            context_items = tuple(replay.items())
        else:
            selected_context_set = set(selected_contexts)
            context_items = tuple(
                (ctx, events)
                for ctx, events in replay.items()
                if ctx in selected_context_set
            )

        replay_limited_contexts = set(prepared.get("replay_limited_contexts", ()))
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
            replay_limited = False

            for ev in events:
                if getattr(ev, "replay_limit_exceeded", False):
                    replay_limited = True
                enabled |= self._replay(ev)
                if self._replay_limit_was_hit(ev):
                    replay_limited = True

            if replay_limited:
                replay_limited_contexts.add(ctx)
            model_enabled[ctx] = enabled

        prepared["replay_limited_contexts"] = replay_limited_contexts
        return model_enabled

    def _context_weights(self, prepared):
        if "context_weights" in prepared:
            return dict(prepared["context_weights"])

        return {
            ctx: len(events)
            for ctx, events in prepared["replay"].items()
        }

    def _sample_precision_context_draw_counts(self, prepared):
        sample_size = self.precision_context_sample_size
        if sample_size is None:
            return {}
        if sample_size < 0:
            raise ValueError("precision_context_sample_size must be positive.")
        if sample_size == 0:
            return {}

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
        if getattr(ev, "replay_limit_exceeded", False):
            return tuple()

        original_ev = ev
        ev = self._canonical_replay_event(ev)
        replay_key = self._replay_signature(ev)

        if replay_key in self._terminal_replay_cache:
            return self._terminal_replay_cache[replay_key]

        if self._should_use_local_projection_replay(original_ev):
            result, replay_limit_hit = self._replay_terminal_states_local_projection(original_ev)
            self._terminal_replay_cache[replay_key] = result
            if replay_limit_hit:
                self._terminal_replay_limit_hit_cache.add(replay_key)
            else:
                self._terminal_replay_limit_hit_cache.discard(replay_key)
            return result

        initial_state = self._initial_state(ev)
        current_states = {
            self._state_key(initial_state): initial_state
        }
        replay_limit_hit = False

        for step in ev.binding_sequence:
            closure_states, limit_hit = self._silent_closure_shortest_first(
                current_states.values()
            )
            replay_limit_hit = replay_limit_hit or limit_hit

            next_states, limit_hit = self._fire_visible_from_closure(
                closure_states,
                step,
            )
            replay_limit_hit = replay_limit_hit or limit_hit

            if not next_states:
                result = tuple()
                self._terminal_replay_cache[replay_key] = result
                if replay_limit_hit:
                    self._terminal_replay_limit_hit_cache.add(replay_key)
                else:
                    self._terminal_replay_limit_hit_cache.discard(replay_key)
                return result

            current_states = next_states

        closure_states, limit_hit = self._silent_closure_shortest_first(
            current_states.values()
        )
        replay_limit_hit = replay_limit_hit or limit_hit
        result = tuple(closure_states)
        self._terminal_replay_cache[replay_key] = result
        if replay_limit_hit:
            self._terminal_replay_limit_hit_cache.add(replay_key)
        else:
            self._terminal_replay_limit_hit_cache.discard(replay_key)
        return result

    def _silent_closure_shortest_first(self, states):
        if self.max_nodes_per_replay is not None and self.max_nodes_per_replay <= 0:
            return tuple(), True

        q = deque(states)
        visited = {}
        limit_hit = False

        while q:
            state = q.popleft()
            key = self._state_key(state)

            if key in visited:
                continue

            if (
                self.max_nodes_per_replay is not None
                and len(visited) >= self.max_nodes_per_replay
            ):
                limit_hit = True
                break

            visited[key] = state

            for ns in self._fire_silent(state):
                q.append(ns)

        return tuple(visited.values()), limit_hit

    def _fire_visible_from_closure(self, closure_states, step):
        if self.max_nodes_per_replay is not None and self.max_nodes_per_replay <= 0:
            return {}, True

        next_states = {}
        limit_hit = False

        for state in closure_states:
            for ns in self._fire_visible(state, step):
                key = self._state_key(ns)
                if key in next_states:
                    continue

                if (
                    self.max_nodes_per_replay is not None
                    and len(next_states) >= self.max_nodes_per_replay
                ):
                    limit_hit = True
                    return next_states, limit_hit

                next_states[key] = ns

        return next_states, limit_hit

    def _replay_limit_was_hit(self, ev):
        if getattr(ev, "replay_limit_exceeded", False):
            return True

        ev = self._canonical_replay_event(ev)
        replay_key = self._replay_signature(ev)
        return replay_key in self._terminal_replay_limit_hit_cache

    def _should_use_local_projection_replay(self, ev):
        return (
            bool(getattr(ev, "local_prefix_states_by_token", ()))
            or (
                self.max_nodes_per_replay is not None
                and len(ev.binding_sequence) > self.max_nodes_per_replay
            )
        )

    def _replay_terminal_states_local_projection(self, ev):
        if ev.local_prefix_states_by_token:
            return self._terminal_states_from_local_prefix_states(
                ev.local_prefix_states_by_token
            )

        token_states = {}
        replay_limit_hit = False

        for ot, tokens in ev.context_tokens_by_type:
            initial_state = self._local_initial_state(ot)
            if not initial_state:
                continue
            for token in tokens:
                token_states[token] = (initial_state,)

        for step in ev.binding_sequence:
            for ot, tokens in step.objects_by_type:
                if not tokens:
                    continue

                updates, limit_hit = self._local_visible_step_successors(
                    ot,
                    step.label,
                    tokens,
                    token_states,
                )
                replay_limit_hit = replay_limit_hit or limit_hit

                if updates is None:
                    return tuple(), replay_limit_hit

                token_states.update(updates)

        return self._terminal_states_from_local_prefix_states(tuple(token_states.items()))

    def _terminal_states_from_local_prefix_states(self, token_states_items):
        marking = defaultdict(list)
        replay_limit_hit = False

        for token, states in token_states_items:
            if not states:
                return tuple(), replay_limit_hit

            ot = token[0]
            closure, limit_hit = self._local_silent_closure(ot, states)
            replay_limit_hit = replay_limit_hit or limit_hit
            if not closure:
                return tuple(), replay_limit_hit

            reachable_places = set()
            for local_state in closure:
                reachable_places.update(local_state)

            for place in sorted(
                reachable_places,
                key=lambda place: self._ensure_model_cache().place_ids[place],
            ):
                marking[place].append(token)

        return (self._canonical_state(marking),), replay_limit_hit

    def _local_initial_state(self, object_type):
        mc = self._ensure_model_cache()
        return tuple(
            sorted(
                mc.initial_places_by_type.get(object_type, ()),
                key=lambda place: mc.place_ids[place],
            )
        )

    def _local_visible_step_successors(self, object_type, label, tokens, token_states):
        mc = self._ensure_model_cache()
        transitions = mc.visible_transitions_by_label_type.get(label, {}).get(object_type, ())
        if not transitions:
            return None, False

        replay_limit_hit = False

        for transition in transitions:
            updates = {}
            transition_limit_hit = False
            transition_enabled = True

            for token in tokens:
                current_states = token_states.get(token)
                if not current_states:
                    current_states = (self._local_initial_state(object_type),)

                next_states = []
                seen = set()
                for state in current_states:
                    closure, limit_hit = self._local_silent_closure(object_type, (state,))
                    transition_limit_hit = transition_limit_hit or limit_hit

                    for closure_state in closure:
                        if not self._local_transition_enabled(closure_state, transition):
                            continue

                        next_state = self._local_move(closure_state, transition)
                        closure_after, limit_hit = self._local_silent_closure(
                            object_type,
                            (next_state,),
                        )
                        transition_limit_hit = transition_limit_hit or limit_hit

                        for candidate in closure_after:
                            if candidate in seen:
                                continue
                            seen.add(candidate)
                            next_states.append(candidate)
                            if (
                                self.max_nodes_per_replay is not None
                                and len(next_states) >= self.max_nodes_per_replay
                            ):
                                break
                        if (
                            self.max_nodes_per_replay is not None
                            and len(next_states) >= self.max_nodes_per_replay
                        ):
                            break
                    if (
                        self.max_nodes_per_replay is not None
                        and len(next_states) >= self.max_nodes_per_replay
                    ):
                        transition_limit_hit = True
                        break

                if not next_states:
                    transition_enabled = False
                    break

                updates[token] = tuple(next_states)

            replay_limit_hit = replay_limit_hit or transition_limit_hit
            if transition_enabled:
                return updates, replay_limit_hit

        return None, replay_limit_hit

    def _local_silent_closure(self, object_type, states):
        normalized_states = tuple(
            tuple(
                sorted(
                    state,
                    key=lambda place: self._ensure_model_cache().place_ids[place],
                )
            )
            for state in states
        )
        cache_key = (object_type, normalized_states, self.max_nodes_per_replay)
        if cache_key in self._local_silent_closure_cache:
            return self._local_silent_closure_cache[cache_key]

        if self.max_nodes_per_replay is not None and self.max_nodes_per_replay <= 0:
            result = (tuple(), True)
            self._local_silent_closure_cache[cache_key] = result
            return result

        mc = self._ensure_model_cache()
        q = deque(normalized_states)
        visited = {}
        limit_hit = False
        silent_transitions = tuple(
            transition
            for transition in mc.silent_transitions
            if transition.object_type == object_type and transition.input_places
        )

        while q:
            state = q.popleft()
            if state in visited:
                continue

            if (
                self.max_nodes_per_replay is not None
                and len(visited) >= self.max_nodes_per_replay
            ):
                limit_hit = True
                break

            visited[state] = state

            for transition in silent_transitions:
                if not self._local_transition_enabled(state, transition):
                    continue
                next_state = self._local_move(state, transition)
                if next_state != state:
                    q.append(next_state)

        result = (tuple(visited.values()), limit_hit)
        self._local_silent_closure_cache[cache_key] = result
        return result

    def _local_transition_enabled(self, state, transition):
        places = set(state)
        return all(place in places for place in transition.input_places)

    def _local_move(self, state, transition):
        mc = self._ensure_model_cache()
        places = set(state)
        for place in transition.input_places:
            places.discard(place)
        places.update(transition.output_places)
        return tuple(sorted(places, key=lambda place: mc.place_ids[place]))

    def _replay(self, ev):
        if getattr(ev, "replay_limit_exceeded", False):
            return frozenset()

        canonical_ev = self._canonical_replay_event(ev)
        replay_key = self._replay_signature(canonical_ev)

        if replay_key in self._replay_cache:
            return self._replay_cache[replay_key]

        enabled = set()
        for state in self._replay_terminal_states(ev):
            enabled |= self._enabled_labels(state)

        result = frozenset(enabled)
        self._replay_cache[replay_key] = result
        return result

    def _canonical_replay_event(self, ev):
        token_map = {}
        token_counters = defaultdict(int)

        def canonical_token(token):
            ot, oid = token
            ot = str(ot)
            key = (ot, oid)
            if key not in token_map:
                token_map[key] = (ot, token_counters[ot])
                token_counters[ot] += 1
            return token_map[key]

        def canonical_objects(objects_by_type):
            canonical = []

            for ot, tokens in objects_by_type:
                normalized_tokens = tuple(
                    sorted(
                        (canonical_token(token) for token in tokens),
                        key=self._token_sort_key,
                    )
                )
                if normalized_tokens:
                    canonical.append((str(ot), normalized_tokens))

            return tuple(canonical)

        binding_sequence = tuple(
            _BindingStep(
                label=step.label,
                objects_by_type=canonical_objects(step.objects_by_type),
            )
            for step in ev.binding_sequence
        )
        context_tokens_by_type = canonical_objects(ev.context_tokens_by_type)

        return _ReplayEvent(
            context_key=ev.context_key,
            binding_sequence=binding_sequence,
            context_tokens_by_type=context_tokens_by_type,
            replay_limit_exceeded=getattr(ev, "replay_limit_exceeded", False),
        )

    def _replay_signature(self, ev):
        if getattr(ev, "replay_limit_exceeded", False):
            return ("replay-limit-exceeded", self.max_nodes_per_replay)

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

        if len(tokens) == 1:
            token = tokens[0]
            return all(token in state.get(p, ()) for p in places)

        needed = Counter(tokens)

        for p in places:
            available = Counter(state.get(p, ()))
            if any(available[token] < count for token, count in needed.items()):
                return False

        return True

    def _move_tokens(self, state, inp, out, tokens: tuple[Token, ...]):
        if not tokens:
            return state

        ns = dict(state)

        if len(tokens) == 1:
            token = tokens[0]

            for p in inp:
                removed = False
                remaining = []
                for current_token in ns.get(p, ()):
                    if current_token == token and not removed:
                        removed = True
                        continue
                    remaining.append(current_token)
                current = tuple(remaining)
                if current:
                    ns[p] = current
                else:
                    ns.pop(p, None)

            for p in out:
                ns[p] = tuple(
                    sorted(
                        ns.get(p, ()) + (token,),
                        key=self._token_sort_key,
                    )
                )

            return ns

        needed = Counter(tokens)

        for p in inp:
            remaining = Counter(ns.get(p, ()))
            remaining.subtract(needed)
            current = tuple(sorted(remaining.elements(), key=self._token_sort_key))
            if current:
                ns[p] = current
            else:
                ns.pop(p, None)

        for p in out:
            ns[p] = tuple(
                sorted(
                    ns.get(p, ()) + tokens,
                    key=self._token_sort_key,
                )
            )

        return ns

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
                (mc.place_ids[p], tuple(tokens))
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

    def _ordered_event_ids(self, ocel):
        # Use the OCEL table order as the implementation's logical event time.
        # PM4Py's OCPN discovery consumes this order; precision must use the
        # same order or discovered lifecycles and replay contexts can disagree.
        return tuple(ocel.events[ocel.event_id_column])

    # -------------------------
    # OCPA-STYLE LOG PREPARATION
    # -------------------------
    def _prepare_log(self, ocel, show_progress=False, *, materialize_replay=True):
        if ocel is None:
            raise ValueError("No OCEL was provided.")
        if self.precision_context_depth is not None and self.precision_context_depth < 0:
            raise ValueError("precision_context_depth must be non-negative.")

        events = ocel.events
        rel = ocel.relations

        eid = ocel.event_id_column
        act = ocel.event_activity
        obj = ocel.object_id_column
        typ = ocel.object_type_column

        ordered = self._ordered_event_ids(ocel)

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

        presets = {}
        # The precision oracle follows the original full-prefix semantics of
        # van der Aalst et al.: every complete observed prefix is its own
        # context. Approximation is handled by context sampling and the replay
        # node budget, not by truncating the oracle context.
        for e in ordered:
            ancestors = set()
            for pred in reverse_adj[e]:
                ancestors.add(pred)
                ancestors.update(presets.get(pred, ()))
            presets[e] = frozenset(ancestors)

        ctx = {}
        log_enabled = defaultdict(set)
        context_weights = defaultdict(int)
        events_by_context = defaultdict(list)
        context_tokens_by_event = {}

        ordered_iter = ordered
        if show_progress:
            desc = "Preparing full-prefix precision contexts"
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

            ctx[e] = key
            log_enabled[key].add(act_map[e])
            context_weights[key] += 1
            events_by_context[key].append(e)
            context_tokens_by_event[e] = tuple(
                sorted(context_tokens, key=self._token_sort_key)
            )

        prepared = {
            "events": ordered,
            "ctx": ctx,
            "log": {
                k: frozenset(v)
                for k, v in log_enabled.items()
            },
            "context_weights": dict(context_weights),
            "events_by_context": {
                k: tuple(v)
                for k, v in events_by_context.items()
            },
            "replay": {},
            "_replay_data": {
                "act_map": act_map,
                "event_order_index": event_order_index,
                "event_tokens": {
                    event_id: tuple(tokens)
                    for event_id, tokens in event_tokens.items()
                },
                "object_events": {
                    token: tuple(history)
                    for token, history in object_events.items()
                },
                "object_event_positions": {
                    token: tuple(event_order_index[event_id] for event_id in history)
                    for token, history in object_events.items()
                },
                "context_tokens_by_event": context_tokens_by_event,
                "presets": presets,
                "objects_by_type_cache": {},
            },
        }

        if materialize_replay:
            self._materialize_replay_events(prepared)

        return prepared

    def _prepare_sampled_precision_log(self, ocel, sample_size, show_progress=False):
        if sample_size < 0:
            raise ValueError("precision_context_sample_size must be positive.")
        if sample_size == 0:
            return self._prepare_precision_log_for_event_counts(
                ocel,
                {},
                show_progress=show_progress,
            )

        ordered = self._ordered_event_ids(ocel)

        if not ordered:
            event_draw_counts = {}
        else:
            rng = random.Random(self.random_seed)
            sampled_events = rng.sample(ordered, k=min(sample_size, len(ordered)))
            event_draw_counts = Counter(sampled_events)

        return self._prepare_precision_log_for_event_counts(
            ocel,
            event_draw_counts,
            show_progress=show_progress,
        )

    def _prepare_precision_log_for_event_counts(
        self,
        ocel,
        event_draw_counts,
        show_progress=False,
    ):
        if ocel is None:
            raise ValueError("No OCEL was provided.")
        if self.precision_context_depth is not None and self.precision_context_depth < 0:
            raise ValueError("precision_context_depth must be non-negative.")

        events = ocel.events
        rel = ocel.relations

        eid = ocel.event_id_column
        act = ocel.event_activity
        obj = ocel.object_id_column
        typ = ocel.object_type_column

        ordered = self._ordered_event_ids(ocel)

        event_order_index = {
            event_id: i
            for i, event_id in enumerate(ordered)
        }
        event_draw_counts = Counter({
            event_id: count
            for event_id, count in dict(event_draw_counts).items()
            if count > 0 and event_id in event_order_index
        })

        act_map = {
            event_id: str(activity)
            for event_id, activity in zip(events[eid], events[act])
        }

        event_tokens = defaultdict(list)
        for e, o, t in rel[[eid, obj, typ]].itertuples(index=False):
            if e in event_order_index:
                event_tokens[e].append((str(t), o))

        object_events = defaultdict(list)
        for e in ordered:
            for token in event_tokens.get(e, ()):
                object_events[token].append(e)

        reverse_adj = {e: set() for e in ordered}
        for history in object_events.values():
            history = sorted(history, key=event_order_index.__getitem__)
            for prev_e, next_e in zip(history, history[1:]):
                reverse_adj[next_e].add(prev_e)

        ancestor_cache = {}

        def ancestors_for(event_id):
            if event_id in ancestor_cache:
                return ancestor_cache[event_id]

            ancestors = set()
            stack = list(reverse_adj.get(event_id, ()))
            while stack:
                pred = stack.pop()
                if pred in ancestors:
                    continue
                ancestors.add(pred)
                stack.extend(reverse_adj.get(pred, ()))

            ancestor_cache[event_id] = frozenset(ancestors)
            return ancestor_cache[event_id]

        ctx = {}
        log_enabled = defaultdict(set)
        context_weights = defaultdict(int)
        events_by_context = defaultdict(list)
        context_tokens_by_event = {}
        presets = {}

        selected_events = tuple(
            sorted(event_draw_counts, key=event_order_index.__getitem__)
        )
        event_iter = selected_events
        if show_progress:
            event_iter = tqdm(
                selected_events,
                total=len(selected_events),
                desc="Preparing sampled full-prefix precision contexts",
                file=sys.stdout,
            )

        for e in event_iter:
            preset_events = ancestors_for(e)
            presets[e] = preset_events

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

            ctx[e] = key
            log_enabled[key].add(act_map[e])
            context_weights[key] += event_draw_counts[e]
            events_by_context[key].append(e)
            context_tokens_by_event[e] = tuple(
                sorted(context_tokens, key=self._token_sort_key)
            )

        return {
            "events": selected_events,
            "ctx": ctx,
            "log": {
                k: frozenset(v)
                for k, v in log_enabled.items()
            },
            "context_weights": dict(context_weights),
            "events_by_context": {
                k: tuple(v)
                for k, v in events_by_context.items()
            },
            "replay": {},
            "_replay_data": {
                "act_map": act_map,
                "event_order_index": event_order_index,
                "event_tokens": {
                    event_id: tuple(tokens)
                    for event_id, tokens in event_tokens.items()
                },
                "object_events": {
                    token: tuple(history)
                    for token, history in object_events.items()
                },
                "object_event_positions": {
                    token: tuple(event_order_index[event_id] for event_id in history)
                    for token, history in object_events.items()
                },
                "context_tokens_by_event": context_tokens_by_event,
                "presets": presets,
                "objects_by_type_cache": {},
            },
        }

    def _materialize_replay_events(self, prepared, *, selected_contexts=None):
        replay = prepared.setdefault("replay", {})
        replay_data = prepared.get("_replay_data")
        if replay_data is None:
            return replay

        events_by_context = prepared.get("events_by_context", {})
        if selected_contexts is None:
            selected_context_keys = tuple(events_by_context)
        else:
            selected_context_set = set(selected_contexts)
            selected_context_keys = tuple(
                ctx
                for ctx in events_by_context
                if ctx in selected_context_set
            )

        act_map = replay_data["act_map"]
        event_order_index = replay_data["event_order_index"]
        event_tokens = replay_data["event_tokens"]
        object_events = replay_data.get("object_events", {})
        object_event_positions = replay_data.get("object_event_positions", {})
        context_tokens_by_event = replay_data.get("context_tokens_by_event", {})
        presets = replay_data["presets"]
        objects_by_type_cache = replay_data["objects_by_type_cache"]
        local_prefix_state_index = None

        def objects_by_type_for_event(event_id):
            if event_id not in objects_by_type_cache:
                objects_by_type_cache[event_id] = self._group_event_tokens_by_type(
                    event_tokens.get(event_id, ())
                )
            return objects_by_type_cache[event_id]

        for ctx in selected_context_keys:
            if ctx in replay:
                continue

            replay_events = []
            for e in events_by_context.get(ctx, ()):
                preset_events = presets[e]
                use_local_prefix = (
                    self.max_nodes_per_replay is not None
                    and len(preset_events) > self.max_nodes_per_replay
                )
                context_tokens = set(context_tokens_by_event.get(e, ()))
                if not context_tokens:
                    context_tokens = set(event_tokens.get(e, ()))
                    for pe in preset_events:
                        context_tokens.update(event_tokens.get(pe, ()))

                binding_sequence = ()
                local_prefix_states = ()

                if use_local_prefix:
                    if local_prefix_state_index is None:
                        local_prefix_state_index = self._ensure_local_prefix_state_index(
                            replay_data
                        )

                    event_position = event_order_index[e]
                    prefix_state_items = []
                    for token in sorted(context_tokens, key=self._token_sort_key):
                        positions = object_event_positions.get(token, ())
                        history = object_events.get(token, ())
                        history_index = bisect_left(positions, event_position) - 1
                        states = None
                        if history_index >= 0:
                            last_event = history[history_index]
                            states = local_prefix_state_index.get((last_event, token))
                        if states is None:
                            states = (self._local_initial_state(token[0]),)
                        prefix_state_items.append((token, states))
                    local_prefix_states = tuple(prefix_state_items)
                else:
                    preset_ordered = tuple(sorted(preset_events, key=event_order_index.__getitem__))
                    binding_sequence = tuple(
                        _BindingStep(
                            label=act_map[pe],
                            objects_by_type=objects_by_type_for_event(pe),
                        )
                        for pe in preset_ordered
                    )

                replay_events.append(
                    _ReplayEvent(
                        context_key=ctx,
                        binding_sequence=binding_sequence,
                        context_tokens_by_type=self._group_event_tokens_by_type(context_tokens),
                        local_prefix_states_by_token=local_prefix_states,
                    )
                )

            replay[ctx] = tuple(replay_events)

        return replay

    def _ensure_local_prefix_state_index(self, replay_data):
        if "local_prefix_state_index" in replay_data:
            return replay_data["local_prefix_state_index"]

        act_map = replay_data["act_map"]
        event_order_index = replay_data["event_order_index"]
        event_tokens = replay_data["event_tokens"]
        objects_by_type_cache = replay_data["objects_by_type_cache"]
        ordered_events = tuple(sorted(event_order_index, key=event_order_index.__getitem__))
        token_states = {}
        state_after_event_token = {}

        def objects_by_type_for_event(event_id):
            if event_id not in objects_by_type_cache:
                objects_by_type_cache[event_id] = self._group_event_tokens_by_type(
                    event_tokens.get(event_id, ())
                )
            return objects_by_type_cache[event_id]

        for event_id in ordered_events:
            label = act_map[event_id]
            for object_type, tokens in objects_by_type_for_event(event_id):
                updates, _ = self._local_visible_step_successors(
                    object_type,
                    label,
                    tokens,
                    token_states,
                )
                if updates is not None:
                    token_states.update(updates)

            for token in event_tokens.get(event_id, ()):
                states = token_states.get(token)
                if states is None:
                    states = (self._local_initial_state(token[0]),)
                state_after_event_token[(event_id, token)] = states

        replay_data["local_prefix_state_index"] = state_after_event_token
        return state_after_event_token
