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

    def tokens_for(self, object_type: str) -> tuple[Token, ...]:
        for bound_object_type, tokens in self.objects_by_type:
            if bound_object_type == object_type:
                return tokens
        return ()

    def non_empty_object_types(self) -> set[str]:
        return {
            object_type
            for object_type, tokens in self.objects_by_type
            if tokens
        }


@dataclass(frozen=True)
class _ReplayEvent:
    context_key: ContextKey
    binding_sequence: tuple[_BindingStep, ...]
    context_tokens_by_type: tuple[tuple[str, tuple[Token, ...]], ...]


@dataclass(frozen=True)
class _EvaluationResult:
    precision: float
    fitness: float


class _ProgressReporter:
    def __init__(self, enabled: bool) -> None:
        self.enabled = enabled
        self._last_percent_by_phase: dict[str, int] = {}

    def phase(self, label: str) -> None:
        if not self.enabled:
            return
        print(f"[NetQuality] {label}...", file=sys.stdout, flush=True)

    def step(self, label: str, current: int, total: int) -> None:
        if not self.enabled or total <= 0:
            return

        percent = int((current * 100) / total)
        previous_percent = self._last_percent_by_phase.get(label, -1)

        if current < total and percent not in {0, 25, 50, 75, 100}:
            return
        if percent == previous_percent:
            return

        self._last_percent_by_phase[label] = percent
        print(
            f"[NetQuality] {label}: {current}/{total} ({percent}%)",
            file=sys.stdout,
            flush=True,
        )

    def finish(self) -> None:
        if not self.enabled:
            return
        print("[NetQuality] Done.", file=sys.stdout, flush=True)


class NetQuality:
    """
    Replay-based fitness and precision for PM4Py object-centric Petri nets.

    Expects the OCPN dictionary returned by `pm4py.discover_oc_petri_net(...)`.
    An OCEL can be provided in the constructor or in `fitness()` / `precision()`.
    """

    def __init__(
        self,
        ocpn: dict[str, Any],
        ocel: OCEL | None = None,
        *,
        max_nodes_per_replay: int | None = 100_000,
        show_progress: bool = False,
    ) -> None:
        if ocpn is None or "petri_nets" not in ocpn:
            raise ValueError(
                "NetQuality expects the PM4Py OCPN dictionary returned by "
                "pm4py.discover_oc_petri_net(...)."
            )

        self.ocpn = ocpn
        self.ocel = ocel
        self.max_nodes_per_replay = max_nodes_per_replay
        self.show_progress = show_progress

        self._double_arcs = ocpn.get("double_arcs_on_activity", {})
        self._model_object_types = tuple(sorted(ocpn["petri_nets"]))
        self._place_ids: dict[PlaceKey, int] = {}
        self._places_by_type: dict[str, tuple[PlaceKey, ...]] = {}
        self._initial_places_by_type: dict[str, tuple[PlaceKey, ...]] = {}
        self._silent_transitions: tuple[_LocalTransition, ...] = ()
        self._visible_combinations_by_label: dict[
            str,
            tuple[tuple[tuple[str, _LocalTransition], ...], ...],
        ] = {}
        self._enabled_labels_cache: dict[tuple[Any, ...], frozenset[str]] = {}

        self._prepare_model()

        self._cached_ocel: OCEL | None = None
        self._cached_result: _EvaluationResult | None = None

    def fitness(
        self,
        ocel: OCEL | None = None,
        *,
        show_progress: bool | None = None,
    ) -> float:
        return self._evaluate(ocel, show_progress=show_progress).fitness

    def precision(
        self,
        ocel: OCEL | None = None,
        *,
        show_progress: bool | None = None,
    ) -> float:
        return self._evaluate(ocel, show_progress=show_progress).precision

    def _prepare_model(self) -> None:
        visible_parts_by_label: dict[str, dict[str, list[_LocalTransition]]] = defaultdict(
            lambda: defaultdict(list)
        )
        silent_transitions: list[_LocalTransition] = []

        for object_type in self._model_object_types:
            net, initial_marking, _ = self.ocpn["petri_nets"][object_type]
            place_keys = []

            for place in net.places:
                place_key = (object_type, place)
                self._place_ids[place_key] = len(self._place_ids)
                place_keys.append(place_key)

            self._places_by_type[object_type] = tuple(place_keys)
            self._initial_places_by_type[object_type] = self._extract_initial_places(
                object_type,
                net,
                initial_marking,
            )

            for transition in net.transitions:
                input_places = tuple(
                    (object_type, arc.source)
                    for arc in transition.in_arcs
                    if isinstance(arc.source, PetriNet.Place)
                )
                output_places = tuple(
                    (object_type, arc.target)
                    for arc in transition.out_arcs
                    if isinstance(arc.target, PetriNet.Place)
                )
                local_transition = _LocalTransition(
                    object_type=object_type,
                    label=transition.label,
                    input_places=input_places,
                    output_places=output_places,
                    variable=bool(
                        transition.label is not None
                        and self._double_arcs.get(object_type, {}).get(
                            transition.label,
                            False,
                        )
                    ),
                )

                if transition.label is None:
                    silent_transitions.append(local_transition)
                else:
                    visible_parts_by_label[transition.label][object_type].append(
                        local_transition
                    )

        self._silent_transitions = tuple(silent_transitions)

        for label, parts_by_type in visible_parts_by_label.items():
            ordered_parts = tuple(
                (object_type, tuple(parts_by_type[object_type]))
                for object_type in sorted(parts_by_type)
            )
            self._visible_combinations_by_label[label] = tuple(
                tuple(zip((object_type for object_type, _ in ordered_parts), combination))
                for combination in product(*(parts for _, parts in ordered_parts))
            )

    def _extract_initial_places(
        self,
        object_type: str,
        net: PetriNet,
        initial_marking: Any,
    ) -> tuple[PlaceKey, ...]:
        initial_places = tuple(
            (object_type, place)
            for place, count in initial_marking.items()
            if count > 0
        )
        if initial_places:
            return initial_places

        return tuple(
            (object_type, place)
            for place in net.places
            if not place.in_arcs
        )

    def _evaluate(
        self,
        ocel: OCEL | None,
        *,
        show_progress: bool | None = None,
    ) -> _EvaluationResult:
        log = ocel if ocel is not None else self.ocel
        if log is None:
            raise ValueError(
                "An OCEL is required. Pass it to NetQuality(..., ocel=...) or "
                "to fitness(ocel) / precision(ocel)."
            )

        progress_enabled = self.show_progress if show_progress is None else show_progress
        if (
            self._cached_ocel is log
            and self._cached_result is not None
            and not progress_enabled
        ):
            return self._cached_result

        evaluation = self._compute_metrics(log, show_progress=progress_enabled)
        self._cached_ocel = log
        self._cached_result = evaluation
        return evaluation

    def _compute_metrics(
        self,
        ocel: OCEL,
        *,
        show_progress: bool = False,
    ) -> _EvaluationResult:
        reporter = _ProgressReporter(show_progress)
        reporter.phase("Preparing event contexts")
        (
            ordered_event_ids,
            enabled_log_activities,
            event_context_keys,
            replay_events_by_context,
        ) = self._prepare_log(ocel, reporter=reporter)

        enabled_model_activities: dict[ContextKey, frozenset[str]] = {}
        replay_items = list(replay_events_by_context.items())
        reporter.phase("Replaying contexts")
        for index, (context_key, replay_events) in enumerate(replay_items, start=1):
            enabled_activities = set()
            for replay_event in replay_events:
                enabled_activities.update(self._replay_event(replay_event))
            enabled_model_activities[context_key] = frozenset(enabled_activities)
            reporter.step("Replaying contexts", index, len(replay_items))

        if not ordered_event_ids:
            reporter.finish()
            return _EvaluationResult(precision=0.0, fitness=0.0)

        fitness_terms = []
        precision_terms = []
        reporter.phase("Aggregating scores")
        for index, event_id in enumerate(ordered_event_ids, start=1):
            context_key = event_context_keys[event_id]
            enabled_log = enabled_log_activities[context_key]
            enabled_model = enabled_model_activities[context_key]
            overlap = enabled_log.intersection(enabled_model)

            fitness_terms.append(len(overlap) / len(enabled_log))
            if enabled_model:
                precision_terms.append(len(overlap) / len(enabled_model))
            reporter.step("Aggregating scores", index, len(ordered_event_ids))

        reporter.finish()
        return _EvaluationResult(
            precision=(
                sum(precision_terms) / len(precision_terms)
                if precision_terms
                else 0.0
            ),
            fitness=sum(fitness_terms) / len(fitness_terms),
        )

    def _prepare_log(
        self,
        ocel: OCEL,
        *,
        reporter: _ProgressReporter | None = None,
    ) -> tuple[
        list[Any],
        dict[ContextKey, frozenset[str]],
        dict[Any, ContextKey],
        dict[ContextKey, tuple[_ReplayEvent, ...]],
    ]:
        events = ocel.events.copy()
        relations = ocel.relations.copy()

        event_id_column = ocel.event_id_column
        activity_column = ocel.event_activity
        timestamp_column = ocel.event_timestamp
        object_id_column = ocel.object_id_column
        object_type_column = ocel.object_type_column

        events["_row_order"] = range(len(events))
        if timestamp_column in events.columns:
            events = events.sort_values(
                [timestamp_column, "_row_order"],
                kind="stable",
            )
        ordered_event_ids = events[event_id_column].tolist()
        event_position = {
            event_id: index
            for index, event_id in enumerate(ordered_event_ids)
        }
        activity_by_event = dict(zip(events[event_id_column], events[activity_column]))

        relations = relations[relations[event_id_column].isin(event_position)].copy()
        relations["_row_order"] = range(len(relations))
        relations["_event_position"] = relations[event_id_column].map(event_position)
        relations = relations.sort_values(
            ["_event_position", "_row_order"],
            kind="stable",
        )

        event_tokens: dict[Any, list[Token]] = {event_id: [] for event_id in ordered_event_ids}
        event_tokens_by_type: dict[Any, dict[str, list[Token]]] = {
            event_id: defaultdict(list)
            for event_id in ordered_event_ids
        }
        seen_event_objects: dict[Any, set[Any]] = {
            event_id: set()
            for event_id in ordered_event_ids
        }

        for event_id, object_id, object_type in relations[
            [event_id_column, object_id_column, object_type_column]
        ].itertuples(index=False, name=None):
            if object_id in seen_event_objects[event_id]:
                continue

            seen_event_objects[event_id].add(object_id)
            token = (object_type, object_id)
            event_tokens[event_id].append(token)
            event_tokens_by_type[event_id][object_type].append(token)

        events_per_object: dict[Token, list[Any]] = defaultdict(list)
        for event_id in ordered_event_ids:
            for token in event_tokens[event_id]:
                events_per_object[token].append(event_id)

        direct_predecessors: dict[Any, set[Any]] = defaultdict(set)
        for object_events in events_per_object.values():
            for previous_event, current_event in zip(object_events, object_events[1:]):
                direct_predecessors[current_event].add(previous_event)

        preset_by_event: dict[Any, set[Any]] = {}
        for event_id in ordered_event_ids:
            preset = set()
            for predecessor in direct_predecessors.get(event_id, ()):
                preset.add(predecessor)
                preset.update(preset_by_event[predecessor])
            preset_by_event[event_id] = preset

        binding_steps_by_event = {
            event_id: _BindingStep(
                label=activity_by_event[event_id],
                objects_by_type=tuple(
                    (object_type, tuple(tokens))
                    for object_type, tokens in sorted(event_tokens_by_type[event_id].items())
                    if tokens
                ),
            )
            for event_id in ordered_event_ids
        }

        event_context_keys: dict[Any, ContextKey] = {}
        enabled_log_by_context: dict[ContextKey, set[str]] = defaultdict(set)
        replay_events_by_context: dict[ContextKey, list[_ReplayEvent]] = defaultdict(list)
        total_events = len(ordered_event_ids)

        for index, event_id in enumerate(ordered_event_ids, start=1):
            preset = preset_by_event[event_id]
            ordered_preset = sorted(preset, key=event_position.__getitem__)

            involved_tokens: dict[Token, None] = {}
            for related_event in (*ordered_preset, event_id):
                for token in event_tokens[related_event]:
                    involved_tokens.setdefault(token, None)

            context: dict[str, Counter[tuple[str, ...]]] = defaultdict(Counter)
            context_tokens_by_type: dict[str, list[Token]] = defaultdict(list)

            for token in involved_tokens:
                object_type = token[0]
                prefix = tuple(
                    activity_by_event[preset_event]
                    for preset_event in events_per_object[token]
                    if preset_event in preset
                )
                context[object_type][prefix] += 1
                context_tokens_by_type[object_type].append(token)

            context_key = self._context_key(context)
            event_context_keys[event_id] = context_key
            enabled_log_by_context[context_key].add(activity_by_event[event_id])
            replay_events_by_context[context_key].append(
                _ReplayEvent(
                    context_key=context_key,
                    binding_sequence=tuple(
                        binding_steps_by_event[preset_event]
                        for preset_event in ordered_preset
                    ),
                    context_tokens_by_type=tuple(
                        (object_type, tuple(tokens))
                        for object_type, tokens in sorted(context_tokens_by_type.items())
                    ),
                )
            )
            if reporter is not None:
                reporter.step("Preparing event contexts", index, total_events)

        return (
            ordered_event_ids,
            {
                context_key: frozenset(activities)
                for context_key, activities in enabled_log_by_context.items()
            },
            event_context_keys,
            {
                context_key: tuple(replay_events)
                for context_key, replay_events in replay_events_by_context.items()
            },
        )

    def _context_key(
        self,
        context: dict[str, Counter[tuple[str, ...]]],
    ) -> ContextKey:
        return tuple(
            (
                object_type,
                tuple(
                    sorted((prefix, count) for prefix, count in prefixes.items())
                ),
            )
            for object_type, prefixes in sorted(context.items())
        )

    def _replay_event(self, replay_event: _ReplayEvent) -> set[str]:
        initial_state = self._build_initial_state(replay_event.context_tokens_by_type)
        if initial_state is None:
            return set()

        queue = deque([(initial_state, 0)])
        visited = {(self._state_key(initial_state), 0)}
        enabled_activities = set()
        explored_nodes = 0

        while queue:
            state, binding_index = queue.popleft()
            explored_nodes += 1
            if (
                self.max_nodes_per_replay is not None
                and explored_nodes > self.max_nodes_per_replay
            ):
                break

            if binding_index == len(replay_event.binding_sequence):
                enabled_activities.update(self._enabled_visible_labels(state))

            next_states = []
            if binding_index < len(replay_event.binding_sequence):
                next_states = self._fire_visible_binding(
                    state,
                    replay_event.binding_sequence[binding_index],
                )
                next_index = binding_index + 1
            else:
                next_index = binding_index

            if next_states:
                for next_state in next_states:
                    visited_key = (self._state_key(next_state), next_index)
                    if visited_key in visited:
                        continue
                    visited.add(visited_key)
                    queue.append((next_state, next_index))
                continue

            for next_state in self._fire_silent_bindings(state):
                visited_key = (self._state_key(next_state), binding_index)
                if visited_key in visited:
                    continue
                visited.add(visited_key)
                queue.append((next_state, binding_index))

        return enabled_activities

    def _build_initial_state(
        self,
        context_tokens_by_type: tuple[tuple[str, tuple[Token, ...]], ...],
    ) -> dict[PlaceKey, set[Token]] | None:
        state: dict[PlaceKey, set[Token]] = {}

        for object_type, tokens in context_tokens_by_type:
            if not tokens:
                continue

            initial_places = self._initial_places_by_type.get(object_type, ())
            if not initial_places:
                return None

            for place in initial_places:
                state[place] = set(tokens)

        return state

    def _fire_visible_binding(
        self,
        state: dict[PlaceKey, set[Token]],
        binding_step: _BindingStep,
    ) -> list[dict[PlaceKey, set[Token]]]:
        combinations = self._visible_combinations_by_label.get(binding_step.label, ())
        if not combinations:
            return []

        required_object_types = binding_step.non_empty_object_types()
        successor_states: dict[tuple[Any, ...], dict[PlaceKey, set[Token]]] = {}

        for combination in combinations:
            combination_types = {object_type for object_type, _ in combination}
            if not required_object_types.issubset(combination_types):
                continue
            if not self._combination_matches_binding(state, combination, binding_step):
                continue

            next_state = self._apply_visible_combination(state, combination, binding_step)
            successor_states[self._state_key(next_state)] = next_state

        return list(successor_states.values())

    def _combination_matches_binding(
        self,
        state: dict[PlaceKey, set[Token]],
        combination: tuple[tuple[str, _LocalTransition], ...],
        binding_step: _BindingStep,
    ) -> bool:
        for object_type, local_transition in combination:
            bound_tokens = set(binding_step.tokens_for(object_type))

            if local_transition.variable:
                if bound_tokens and not self._tokens_available(
                    state,
                    local_transition.input_places,
                    bound_tokens,
                ):
                    return False
                continue

            if len(bound_tokens) != 1:
                return False

            if not self._tokens_available(
                state,
                local_transition.input_places,
                bound_tokens,
            ):
                return False

        return True

    def _apply_visible_combination(
        self,
        state: dict[PlaceKey, set[Token]],
        combination: tuple[tuple[str, _LocalTransition], ...],
        binding_step: _BindingStep,
    ) -> dict[PlaceKey, set[Token]]:
        next_state = {place: set(tokens) for place, tokens in state.items()}
        for object_type, local_transition in combination:
            bound_tokens = set(binding_step.tokens_for(object_type))
            if not bound_tokens:
                continue
            self._move_tokens(
                next_state,
                local_transition.input_places,
                local_transition.output_places,
                bound_tokens,
            )
        return next_state

    def _fire_silent_bindings(
        self,
        state: dict[PlaceKey, set[Token]],
    ) -> list[dict[PlaceKey, set[Token]]]:
        successor_states: dict[tuple[Any, ...], dict[PlaceKey, set[Token]]] = {}

        for local_transition in self._silent_transitions:
            candidate_tokens = self._shared_tokens(state, local_transition)
            if not candidate_tokens:
                continue

            for token in candidate_tokens:
                next_state = {place: set(tokens) for place, tokens in state.items()}
                self._move_tokens(
                    next_state,
                    local_transition.input_places,
                    local_transition.output_places,
                    {token},
                )
                successor_states[self._state_key(next_state)] = next_state

        return list(successor_states.values())

    def _move_tokens(
        self,
        state: dict[PlaceKey, set[Token]],
        input_places: tuple[PlaceKey, ...],
        output_places: tuple[PlaceKey, ...],
        tokens: set[Token],
    ) -> None:
        for place in input_places:
            remaining_tokens = state.get(place, set()) - tokens
            if remaining_tokens:
                state[place] = remaining_tokens
            else:
                state.pop(place, None)

        for place in output_places:
            state.setdefault(place, set()).update(tokens)

    def _enabled_visible_labels(self, state: dict[PlaceKey, set[Token]]) -> frozenset[str]:
        state_key = self._state_key(state)
        if state_key in self._enabled_labels_cache:
            return self._enabled_labels_cache[state_key]

        enabled_labels = set()
        for label, combinations in self._visible_combinations_by_label.items():
            for combination in combinations:
                if self._combination_has_enabled_binding(state, combination):
                    enabled_labels.add(label)
                    break

        enabled_labels = frozenset(enabled_labels)
        self._enabled_labels_cache[state_key] = enabled_labels
        return enabled_labels

    def _combination_has_enabled_binding(
        self,
        state: dict[PlaceKey, set[Token]],
        combination: tuple[tuple[str, _LocalTransition], ...],
    ) -> bool:
        for _, local_transition in combination:
            shared_tokens = self._shared_tokens(state, local_transition)
            if not local_transition.variable and not shared_tokens:
                return False
        return True

    def _shared_tokens(
        self,
        state: dict[PlaceKey, set[Token]],
        local_transition: _LocalTransition,
    ) -> set[Token]:
        if local_transition.input_places:
            token_sets = [state.get(place, set()) for place in local_transition.input_places]
            if not token_sets:
                return set()
            shared_tokens = set(token_sets[0])
            for token_set in token_sets[1:]:
                shared_tokens.intersection_update(token_set)
            return shared_tokens

        shared_tokens = set()
        for place in self._places_by_type.get(local_transition.object_type, ()):
            shared_tokens.update(state.get(place, set()))
        return shared_tokens

    def _tokens_available(
        self,
        state: dict[PlaceKey, set[Token]],
        input_places: tuple[PlaceKey, ...],
        required_tokens: set[Token],
    ) -> bool:
        for place in input_places:
            if not required_tokens.issubset(state.get(place, set())):
                return False
        return True

    def _state_key(self, state: dict[PlaceKey, set[Token]]) -> tuple[Any, ...]:
        encoded_places = []
        for place, tokens in state.items():
            if not tokens:
                continue
            encoded_places.append(
                (
                    self._place_ids[place],
                    tuple(sorted(tokens, key=lambda token: (token[0], str(token[1])))),
                )
            )
        return tuple(sorted(encoded_places))
