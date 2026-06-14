from __future__ import annotations

from abc import ABC, abstractmethod
from itertools import combinations
from math import comb
from typing import Any

from pm4py.objects.petri_net.obj import PetriNet

from .checkpoints import CheckpointManager
from .discovery_preparation import (
    _build_terminal_states_by_context,
    _build_layer_ocel,
    _build_ocel_filtering_context,
    _build_ocel_from_filtering_context,
    _context_weights,
    _enabled_labels_by_context_from_terminal_states,
    _enabled_mass_by_context,
    _filter_ocel_filtering_context,
    _merge_ocpn_models,
    _precision_from_prepared,
    _project_ocpn_to_activities,
    _sample_context_draw_counts,
    _extract_ocel_filtering_context,
)
from .framework import (
    AcceptingOCPN,
    AdvancedProcessArea,
    AdvancedProcessAreaHierarchy,
    ActivityLayerAssignment,
    DeltaReference,
    LayerAssignment,
    ObjectCentricSubprocess,
    PrecisionParameters,
    QualityScores,
    SimpleSubprocess,
)
from .net_quality import NetQuality
from .subprocess_detection import collapse_sub_processes, detect_subprocess_components


class DiscoveryTechnique(ABC):
    @abstractmethod
    def mine(self, log) -> AcceptingOCPN | None:
        pass


class PM4PyOCPNDiscovery(DiscoveryTechnique):
    def __init__(
        self,
        checkpoint: CheckpointManager | None = None,
        verbose: bool = False,
    ):
        self.checkpoint = checkpoint or CheckpointManager(verbose=verbose)

    def mine(self, log) -> AcceptingOCPN | None:
        from .discovery_preparation import _discover_ocpn

        with self.checkpoint.section(
            "PM4Py OCPN discovery",
            metadata=_log_metadata(log),
        ):
            return AcceptingOCPN.from_raw(_discover_ocpn(log))


class SubprocessMiner:
    def __init__(
        self,
        checkpoint: CheckpointManager | None = None,
        verbose: bool = False,
    ):
        self.checkpoint = checkpoint or CheckpointManager(verbose=verbose)

    def mine(
        self,
        net: AcceptingOCPN | None,
        subprocess_activities: frozenset[str],
    ) -> tuple[Any, ...]:
        subprocess_activities = frozenset(
            str(activity)
            for activity in subprocess_activities
            if activity is not None
        )
        with self.checkpoint.section(
            "Subprocess detection",
            total=2,
            metadata={
                "subprocess_activities": len(subprocess_activities),
                "object_types": len(net.object_types) if net is not None else 0,
            },
        ) as span:
            if net is None:
                span.update(2, postfix="no net")
                return ()

            activity_to_layer = {
                activity: 1
                for activity in net.activities
                if activity in subprocess_activities
            }
            with span.child(
                "Detect subprocess components",
                metadata={"activities": len(activity_to_layer)},
            ):
                regions = tuple(detect_subprocess_components(
                    net.raw,
                    activity_to_layer,
                    reference_layer=2,
                ))
            span.update(postfix=f"regions={len(regions)}")

            with span.child(
                "Convert subprocess components",
                total=len(regions),
                unit="region",
            ) as convert_span:
                converted = []
                for region in regions:
                    converted.append(_to_framework_subprocess(region))
                    convert_span.update()
            span.update(postfix=f"subprocesses={len(converted)}")
            return tuple(converted)


class CollapsedNetBuilder:
    def __init__(
        self,
        checkpoint: CheckpointManager | None = None,
        verbose: bool = False,
    ):
        self.checkpoint = checkpoint or CheckpointManager(verbose=verbose)

    def collapse(
        self,
        net: AcceptingOCPN | None,
        subprocesses: tuple[Any, ...],
    ) -> AcceptingOCPN | None:
        with self.checkpoint.section(
            "Collapse subprocesses",
            metadata={
                "subprocesses": len(subprocesses or ()),
                "object_types": len(net.object_types) if net is not None else 0,
            },
        ):
            if net is None:
                return None

            collapsed_raw, _ = collapse_sub_processes(net.raw, _raw_subprocesses(subprocesses))
            return AcceptingOCPN.from_raw(collapsed_raw)


class PrecisionCalculator:
    def __init__(
        self,
        parameters: PrecisionParameters | None = None,
        checkpoint: CheckpointManager | None = None,
        verbose: bool = False,
    ):
        self.parameters = parameters or PrecisionParameters()
        self.checkpoint = checkpoint or CheckpointManager(verbose=verbose)

    def precision(self, log, net_or_hierarchy) -> float:
        net = self._as_net(net_or_hierarchy)
        if net is None:
            return 0.0

        with self.checkpoint.section(
            "Precision calculation",
            metadata={
                **_log_metadata(log),
                "sample_size": self.parameters.sample_size,
                "depth": self.parameters.d,
            },
        ):
            return float(NetQuality(
                net.raw,
                log,
                max_nodes_per_replay=self.parameters.replay_budget or 1000,
                precision_context_sample_size=self.parameters.sample_size,
                precision_context_depth=self.parameters.d,
                random_seed=self.parameters.random_seed,
            ).precision())

    def prepare_delta(self, log, full_net: AcceptingOCPN | None) -> Any:
        if full_net is None:
            return None
        with self.checkpoint.section(
            "Prepare precision delta context",
            total=4,
            metadata={
                **_log_metadata(log),
                "sample_size": self.parameters.sample_size,
                "depth": self.parameters.d,
            },
        ) as span:
            quality = NetQuality(
                full_net.raw,
                log,
                max_nodes_per_replay=self.parameters.replay_budget or 1000,
                precision_context_sample_size=self.parameters.sample_size,
                precision_context_depth=self.parameters.d,
                random_seed=self.parameters.random_seed,
            )
            prepared_original = quality._prepare_log(log)
            exact_context_weights = _context_weights(prepared_original)
            sampled_context_weights = _sample_context_draw_counts(
                exact_context_weights,
                self.parameters.sample_size,
                random_seed=self.parameters.random_seed,
            )
            context_weights = sampled_context_weights or exact_context_weights
            sampled_contexts = tuple(
                ctx
                for ctx in prepared_original["replay"]
                if ctx in context_weights
            )
            selected_contexts = sampled_contexts if sampled_context_weights is not None else None
            span.update(postfix=f"contexts={len(context_weights)}")

            terminal_states_by_context = _build_terminal_states_by_context(
                quality,
                prepared_original,
                selected_contexts=selected_contexts,
                show_progress=False,
            )
            span.update(postfix="terminal states cached")

            original_enabled_by_context = _enabled_labels_by_context_from_terminal_states(
                quality,
                terminal_states_by_context,
            )
            baseline_enabled_mass = _enabled_mass_by_context(
                context_weights,
                original_enabled_by_context,
            )
            span.update(postfix="enabled labels cached")

            precision = _precision_from_prepared(
                prepared_original,
                original_enabled_by_context,
                context_weights=context_weights,
            )
            span.update(postfix=f"precision={precision:.3f}")

            return {
                "quality": quality,
                "prepared_original": prepared_original,
                "original_terminal_states_by_context": terminal_states_by_context,
                "original_enabled_by_context": original_enabled_by_context,
                "context_weights": context_weights,
                "baseline_enabled_mass": baseline_enabled_mass,
                "precision": precision,
                "sampled_contexts": selected_contexts,
                "candidate_precision_cache": {},
            }

    def estimate_after_delta(
        self,
        delta_reference: DeltaReference,
        candidate_activities: frozenset[str],
    ) -> float:
        candidate_activities = frozenset(
            str(activity)
            for activity in candidate_activities
            if activity is not None
        )
        with self.checkpoint.section(
            "Estimate precision after delta",
            metadata={
                "candidate_activities": len(candidate_activities),
                "removed_activities": len(delta_reference.full_activities - candidate_activities),
            },
        ):
            context = delta_reference.precision_context
            if not isinstance(context, dict):
                return delta_reference.baseline_precision

            cache = context.setdefault("candidate_precision_cache", {})
            cache_key = tuple(sorted(candidate_activities))
            if cache_key in cache:
                return cache[cache_key]

            precision = _estimate_precision_for_activity_subset(
                context,
                candidate_activities,
            )
            cache[cache_key] = precision
            return precision

    def _as_net(self, net_or_hierarchy) -> AcceptingOCPN | None:
        if isinstance(net_or_hierarchy, AcceptingOCPN):
            return net_or_hierarchy
        if isinstance(net_or_hierarchy, AdvancedProcessAreaHierarchy):
            raw = _merge_ocpn_models(*[
                area.net.raw
                for area in net_or_hierarchy.areas
                if area.net is not None
            ])
            return AcceptingOCPN.from_raw(raw)
        return None


class HierarchyQualityEvaluator:
    def __init__(
        self,
        discovery_technique: DiscoveryTechnique,
        collapsed_net_builder: CollapsedNetBuilder,
        precision_calculator: PrecisionCalculator,
        checkpoint: CheckpointManager | None = None,
        verbose: bool = False,
    ):
        self.discovery_technique = discovery_technique
        self.collapsed_net_builder = collapsed_net_builder
        self.precision_calculator = precision_calculator
        self.checkpoint = checkpoint or CheckpointManager(verbose=verbose)

    def evaluate(self, log, hierarchy: AdvancedProcessAreaHierarchy) -> QualityScores:
        with self.checkpoint.section(
            "Evaluate hierarchy quality",
            total=5,
            metadata={"areas": len(hierarchy.areas)},
        ) as span:
            baseline_net = self.discovery_technique.mine(log)
            span.update(postfix="baseline net mined")

            with span.child("Compute baseline quality"):
                baseline_complexity = _complexity(baseline_net)
                baseline_precision = self.precision_calculator.precision(log, baseline_net)
            span.update(postfix=f"baseline precision={baseline_precision:.3f}")

            with span.child("Merge hierarchy nets"):
                collapsed_hierarchy_net = _merge_hierarchy_nets(
                    hierarchy,
                    collapse=True,
                    collapsed_net_builder=self.collapsed_net_builder,
                )
                hierarchy_net = _merge_hierarchy_nets(
                    hierarchy,
                    collapse=False,
                    collapsed_net_builder=self.collapsed_net_builder,
                )
            span.update(postfix="hierarchy nets merged")

            with span.child("Compute hierarchy quality"):
                complexity = _complexity(collapsed_hierarchy_net)
                precision = self.precision_calculator.precision(log, hierarchy_net)
            span.update(postfix=f"precision={precision:.3f}")

            simplicity_gain = (
                1 - complexity / baseline_complexity
                if baseline_complexity > 0
                else 0.0
            )
            information_loss = (
                1 - precision / baseline_precision
                if baseline_precision > 0
                else 1.0
            )
            quality = _f1_quality(simplicity_gain, information_loss)
            span.update(postfix=f"quality={quality:.3f}")
            return QualityScores(
                simplicity_gain=simplicity_gain,
                information_loss=information_loss,
                quality=quality,
                precision=precision,
                complexity=complexity,
            )


class CheckSet:
    def __init__(
        self,
        discovery_technique: DiscoveryTechnique,
        subprocess_miner: SubprocessMiner,
        collapsed_net_builder: CollapsedNetBuilder,
        precision_calculator: PrecisionCalculator,
        use_delta: bool = True,
        checkpoint: CheckpointManager | None = None,
        verbose: bool = False,
    ):
        self.discovery_technique = discovery_technique
        self.subprocess_miner = subprocess_miner
        self.collapsed_net_builder = collapsed_net_builder
        self.precision_calculator = precision_calculator
        self.use_delta = use_delta
        self.checkpoint = checkpoint or CheckpointManager(verbose=verbose)
        self._candidate_score_cache: dict[frozenset[str], QualityScores] = {}
        self.subprocess_activities: frozenset[str] = frozenset()
        if checkpoint is not None:
            _share_checkpoint(discovery_technique, checkpoint)
            _share_checkpoint(subprocess_miner, checkpoint)
            _share_checkpoint(collapsed_net_builder, checkpoint)
            _share_checkpoint(precision_calculator, checkpoint)
        self.delta_reference: DeltaReference | None = None

    def prepare_delta_reference(
        self,
        log,
        full_activities: frozenset[str],
        subprocess_activities: frozenset[str],
        *,
        prepare_precision: bool = True,
    ) -> DeltaReference:
        self._candidate_score_cache = {}
        full_activities = frozenset(str(activity) for activity in full_activities)
        subprocess_activities = frozenset(str(activity) for activity in subprocess_activities)
        self.subprocess_activities = subprocess_activities
        total_steps = 5 if prepare_precision else 3
        with self.checkpoint.section(
            "Prepare checkset delta reference",
            total=total_steps,
            metadata={
                "activities": len(full_activities),
                "subprocess_activities": len(subprocess_activities),
                "prepare_precision": prepare_precision,
            },
        ) as span:
            with span.child(
                "Filter full layer log",
                metadata={"activities": len(full_activities)},
            ):
                if _log_contains_only_activities(log, full_activities):
                    full_log = log
                else:
                    full_log = _filter_log_to_activities(log, full_activities)
            span.update(postfix="full log filtered")

            full_net = self.discovery_technique.mine(full_log)
            span.update(postfix="full net mined")

            full_subprocesses = self.subprocess_miner.mine(full_net, subprocess_activities)
            span.update(postfix=f"subprocesses={len(full_subprocesses)}")

            full_collapsed_net = None
            baseline_complexity = 0.0
            precision_context = None
            baseline_precision = 0.0
            if prepare_precision:
                full_collapsed_net = self.collapsed_net_builder.collapse(full_net, full_subprocesses)
                baseline_complexity = _complexity(full_collapsed_net)
                span.update(postfix="full net collapsed")

                precision_context = self.precision_calculator.prepare_delta(full_log, full_net)
                baseline_precision = (
                    precision_context.get("precision", 0.0)
                    if isinstance(precision_context, dict)
                    else 0.0
                )
                span.update(postfix=f"precision={baseline_precision:.3f}")

            self.delta_reference = DeltaReference(
                full_log=full_log,
                full_activities=frozenset(str(activity) for activity in full_activities),
                subprocess_activities=frozenset(str(activity) for activity in subprocess_activities),
                full_net=full_net,
                full_subprocesses=full_subprocesses,
                full_collapsed_net=full_collapsed_net,
                baseline_complexity=baseline_complexity,
                baseline_precision=baseline_precision,
                precision_context=precision_context,
            )
            return self.delta_reference

    def __call__(
        self,
        candidate_activities: frozenset[str],
        hierarchy: AdvancedProcessAreaHierarchy,
        log,
    ) -> float:
        return self.evaluate(candidate_activities, hierarchy, log).quality

    def evaluate(
        self,
        candidate_activities: frozenset[str],
        hierarchy: AdvancedProcessAreaHierarchy,
        log,
    ) -> QualityScores:
        candidate_activities = frozenset(candidate_activities)
        if self.use_delta:
            return self._evaluate_delta(candidate_activities)
        return self._evaluate_recomputed(candidate_activities, hierarchy, log)

    def _evaluate_delta(self, candidate_activities: frozenset[str]) -> QualityScores:
        if self.delta_reference is None:
            raise RuntimeError("prepare_delta_reference() must be called before delta checkset evaluation.")

        candidate_activities = frozenset(
            str(activity)
            for activity in candidate_activities
            if activity is not None
        )
        if candidate_activities in self._candidate_score_cache:
            return self._candidate_score_cache[candidate_activities]

        full_net = self.delta_reference.full_net
        if full_net is None:
            return QualityScores(0.0, 1.0, 0.0, 0.0, 0.0)

        with self.checkpoint.section(
            "Evaluate checkset candidate",
            total=6,
            metadata={"activities": len(candidate_activities), "mode": "delta"},
        ) as span:
            with span.child(
                "Project candidate net",
                metadata={"activities": len(candidate_activities)},
            ):
                candidate_raw = _project_ocpn_to_activities(full_net.raw, candidate_activities)
                candidate_net = AcceptingOCPN.from_raw(candidate_raw)
            span.update(postfix="candidate projected")

            cached_subprocesses = _cached_subprocesses_for_candidate(
                self.delta_reference.full_subprocesses,
                candidate_activities & self.delta_reference.subprocess_activities,
            )
            subprocesses = _rematch_subprocesses_for_projected_net(
                candidate_net,
                cached_subprocesses,
                candidate_activities & self.delta_reference.subprocess_activities,
            )
            span.update(postfix=f"subprocesses={len(subprocesses)}")

            collapsed_candidate_net = self.collapsed_net_builder.collapse(candidate_net, subprocesses)
            span.update(postfix="candidate collapsed")

            with span.child("Compute candidate complexity"):
                complexity = _complexity(collapsed_candidate_net)
            span.update(postfix=f"complexity={complexity:.1f}")

            precision = self.precision_calculator.estimate_after_delta(
                self.delta_reference,
                candidate_activities,
            )
            span.update(postfix=f"precision={precision:.3f}")

            baseline_complexity = self.delta_reference.baseline_complexity
            baseline_precision = self.delta_reference.baseline_precision
            simplicity_gain = (
                1 - complexity / baseline_complexity
                if baseline_complexity > 0
                else 0.0
            )
            information_loss = (
                1 - precision / baseline_precision
                if baseline_precision > 0
                else 1.0
            )
            quality = _f1_quality(simplicity_gain, information_loss)
            span.update(postfix=f"quality={quality:.3f}")
            scores = QualityScores(
                simplicity_gain=simplicity_gain,
                information_loss=information_loss,
                quality=quality,
                precision=precision,
                complexity=complexity,
            )
            self._candidate_score_cache[candidate_activities] = scores
            return scores

    def _evaluate_recomputed(
        self,
        candidate_activities: frozenset[str],
        hierarchy: AdvancedProcessAreaHierarchy,
        log,
    ) -> QualityScores:
        with self.checkpoint.section(
            "Evaluate checkset candidate",
            total=4,
            metadata={"activities": len(candidate_activities), "mode": "recomputed"},
        ) as span:
            with span.child("Filter candidate log"):
                candidate_log = _filter_log_to_activities(log, candidate_activities)
            span.update(postfix="candidate log filtered")

            candidate_net = self.discovery_technique.mine(candidate_log)
            span.update(postfix="candidate net mined")

            subprocesses = self.subprocess_miner.mine(
                candidate_net,
                candidate_activities & self.subprocess_activities,
            )
            span.update(postfix=f"subprocesses={len(subprocesses)}")

            area = AdvancedProcessArea(
                object_types=candidate_net.object_types if candidate_net is not None else frozenset(),
                activities=candidate_activities,
                net=candidate_net,
                resources={},
                subprocesses=subprocesses,
            )
            evaluator = HierarchyQualityEvaluator(
                self.discovery_technique,
                self.collapsed_net_builder,
                self.precision_calculator,
                checkpoint=self.checkpoint,
            )
            scores = evaluator.evaluate(log, hierarchy.append(area))
            span.update(postfix=f"quality={scores.quality:.3f}")
            return scores


class OptimizationFunction(ABC):
    @abstractmethod
    def optimize(
        self,
        log,
        hierarchy: AdvancedProcessAreaHierarchy,
        candidate_activities: frozenset[str],
        *,
        required_activities: frozenset[str] = frozenset(),
    ) -> frozenset[str]:
        pass


class CheckSetBasedOptimizationFunction(OptimizationFunction):
    def __init__(
        self,
        check_set: CheckSet,
        checkpoint: CheckpointManager | None = None,
        verbose: bool = False,
    ):
        self.check_set = check_set
        self.checkpoint = checkpoint or check_set.checkpoint or CheckpointManager(verbose=verbose)

    def score(
        self,
        activities: frozenset[str],
        hierarchy: AdvancedProcessAreaHierarchy,
        log,
    ) -> float:
        return self.check_set(activities, hierarchy, log)


class PrimitiveOptimalOptimization(CheckSetBasedOptimizationFunction):
    def optimize(
        self,
        log,
        hierarchy: AdvancedProcessAreaHierarchy,
        candidate_activities: frozenset[str],
        *,
        required_activities: frozenset[str] = frozenset(),
    ) -> frozenset[str]:
        candidate_tuple = tuple(sorted(candidate_activities))
        total_evaluations = 2 ** len(candidate_tuple)

        with self.checkpoint.section(
            "Primitive optimal optimization",
            total=total_evaluations,
            metadata={
                "candidates": len(candidate_tuple),
                "required": len(required_activities),
            },
            unit="evaluation",
        ) as span:
            best_activities = frozenset(required_activities)
            best_score = self.score(best_activities, hierarchy, log)
            span.update(postfix=f"best={best_score:.3f}")

            for size in range(1, len(candidate_tuple) + 1):
                with span.child(
                    f"Evaluate subsets of size {size}",
                    total=comb(len(candidate_tuple), size),
                    metadata={"subset_size": size},
                    unit="subset",
                ) as subset_span:
                    for subset in combinations(candidate_tuple, size):
                        activities = frozenset(required_activities | frozenset(subset))
                        score = self.score(activities, hierarchy, log)
                        if score > best_score:
                            best_score = score
                            best_activities = activities
                        subset_span.update(postfix=f"score={score:.3f}")
                        span.update(postfix=f"best={best_score:.3f}")

            return best_activities


class GreedyOptimization(CheckSetBasedOptimizationFunction):
    def optimize(
        self,
        log,
        hierarchy: AdvancedProcessAreaHierarchy,
        candidate_activities: frozenset[str],
        *,
        required_activities: frozenset[str] = frozenset(),
    ) -> frozenset[str]:
        candidate_count = len(candidate_activities)
        max_evaluations = 1 + candidate_count * (candidate_count + 1) // 2

        with self.checkpoint.section(
            "Greedy optimization",
            total=max_evaluations,
            metadata={
                "candidates": candidate_count,
                "required": len(required_activities),
            },
            unit="evaluation",
        ) as span:
            selected = frozenset(required_activities | candidate_activities)
            removable = set(candidate_activities)
            current_score = self.score(selected, hierarchy, log)
            span.update(postfix=f"full={current_score:.3f}")
            round_number = 1

            while removable:
                improving_removals = []
                with span.child(
                    f"Greedy round {round_number}",
                    total=len(removable),
                    metadata={
                        "remaining": len(removable),
                        "selected": len(selected),
                    },
                    unit="candidate",
                ) as round_span:
                    for activity in sorted(removable):
                        activities = selected - {activity}
                        score = self.score(frozenset(activities), hierarchy, log)
                        if score > current_score:
                            improving_removals.append((score, activity))
                        round_span.update(postfix=f"remove {activity}: {score:.3f}")
                        span.update(postfix=f"last={score:.3f}")

                if not improving_removals:
                    span.update(0, postfix="no improving removal")
                    break

                best_score, best_activity = max(improving_removals, key=lambda item: (item[0], item[1]))
                selected = selected - {best_activity}
                removable.remove(best_activity)
                current_score = best_score
                round_number += 1

            return frozenset(selected)


class SubprocessMoveUpOptimization(CheckSetBasedOptimizationFunction):
    """Select lower-layer activities exactly when cached subprocesses contain them."""

    requires_precision_delta_reference = False

    def optimize(
        self,
        log,
        hierarchy: AdvancedProcessAreaHierarchy,
        candidate_activities: frozenset[str],
        *,
        required_activities: frozenset[str] = frozenset(),
    ) -> frozenset[str]:
        candidate_activities = frozenset(str(activity) for activity in candidate_activities)
        required_activities = frozenset(str(activity) for activity in required_activities)

        with self.checkpoint.section(
            "Subprocess move-up optimization",
            metadata={
                "candidates": len(candidate_activities),
                "required": len(required_activities),
            },
        ) as span:
            delta_reference = self.check_set.delta_reference
            if delta_reference is None:
                delta_reference = self.check_set.prepare_delta_reference(
                    log,
                    required_activities | candidate_activities,
                    candidate_activities,
                    prepare_precision=False,
                )
                span.update(postfix="delta reference prepared")

            moved_up = set()
            subprocesses = tuple(delta_reference.full_subprocesses or ())
            with span.child(
                "Collect subprocess activities",
                total=len(subprocesses),
                metadata={"subprocesses": len(subprocesses)},
                unit="subprocess",
            ) as collect_span:
                for subprocess in subprocesses:
                    visible_activities = _subprocess_visible_activities(subprocess)
                    if not _is_nontrivial_subprocess(subprocess):
                        collect_span.update(postfix="trivial skipped")
                        continue
                    selected = visible_activities & candidate_activities
                    moved_up.update(selected)
                    collect_span.update(postfix=f"selected={len(selected)}")

            selected_activities = required_activities | frozenset(moved_up)
            span.update(
                metadata={"moved_up": len(moved_up)},
                postfix=f"selected={len(selected_activities)}",
            )
            return frozenset(selected_activities)


class ModelDiscovery:
    def __init__(
        self,
        discovery_technique: DiscoveryTechnique,
        optimization_function: OptimizationFunction,
        alpha_decision_parameter: float = 0.5,
        subprocess_miner: SubprocessMiner | None = None,
        checkpoint: CheckpointManager | None = None,
        verbose: bool = False,
    ):
        self.discovery_technique = discovery_technique
        self.optimization_function = optimization_function
        self.alpha_decision_parameter = alpha_decision_parameter
        self.checkpoint = checkpoint or getattr(
            optimization_function,
            "checkpoint",
            None,
        ) or CheckpointManager(verbose=verbose)
        if checkpoint is not None:
            _share_checkpoint(discovery_technique, checkpoint)
            _share_checkpoint(optimization_function, checkpoint)
            _share_checkpoint(getattr(optimization_function, "check_set", None), checkpoint)
            _share_checkpoint(subprocess_miner, checkpoint)
        self.subprocess_miner = subprocess_miner or SubprocessMiner(checkpoint=self.checkpoint)
        self.hierarchy: AdvancedProcessAreaHierarchy | None = None
        self.activity_layer_assignment: ActivityLayerAssignment | None = None

    def mine(
        self,
        log,
        layer_assignment: LayerAssignment,
        delta: list[int] | tuple[int, ...],
    ) -> AdvancedProcessAreaHierarchy:
        layer_count = len(layer_assignment.layers)
        with self.checkpoint.section(
            "Model discovery",
            total=2 + layer_count,
            metadata={"layers": layer_count},
        ) as span:
            with span.child("Extract OCEL filtering context"):
                object_to_type, event_records = _extract_ocel_filtering_context(log)
            span.update(postfix=f"events={len(event_records)}")

            object_to_layer = layer_assignment.as_object_to_layer()
            with span.child(
                "Derive activity layer assignment",
                metadata={"object_types": len(object_to_layer)},
            ):
                activity_layer_assignment = _derive_activity_layer_assignment(
                    event_records,
                    object_to_type,
                    object_to_layer,
                )
            self.activity_layer_assignment = activity_layer_assignment
            span.update(postfix=f"activities={len(activity_layer_assignment.activity_to_layer)}")

            hierarchy = AdvancedProcessAreaHierarchy(layer_assignment=layer_assignment)

            for layer in range(1, layer_count + 1):
                with span.child(
                    f"Discover process area layer {layer}",
                    total=8,
                    metadata={
                        "layer": layer,
                        "object_types": len(layer_assignment.object_types_at(layer)),
                    },
                ) as layer_span:
                    delta_value = delta[layer - 1] if layer - 1 < len(delta) else 0
                    native_activities = activity_layer_assignment.activities_by_layer.get(layer, frozenset())
                    full_activities = frozenset(
                        activity
                        for activity, activity_layer in activity_layer_assignment.activity_to_layer.items()
                        if max(1, layer - delta_value) <= activity_layer <= layer
                    )
                    candidate_activities = full_activities - native_activities
                    selected_object_types = layer_assignment.object_types_at(layer)
                    layer_span.update(
                        metadata={
                            "native_activities": len(native_activities),
                            "candidate_activities": len(candidate_activities),
                            "full_activities": len(full_activities),
                        },
                        postfix="scope prepared",
                    )

                    if candidate_activities:
                        with layer_span.child(
                            "Build full layer log",
                            metadata={"activities": len(full_activities)},
                        ):
                            full_layer_log, _, _ = _build_layer_ocel(
                                log,
                                event_records,
                                object_to_type,
                                selected_object_types,
                                full_activities,
                            )
                        layer_span.update(postfix="full layer log built")

                        check_set = getattr(self.optimization_function, "check_set", None)
                        if isinstance(check_set, CheckSet):
                            check_set.subprocess_activities = candidate_activities
                            if check_set.use_delta:
                                check_set.prepare_delta_reference(
                                    full_layer_log,
                                    full_activities,
                                    candidate_activities,
                                    prepare_precision=getattr(
                                        self.optimization_function,
                                        "requires_precision_delta_reference",
                                        True,
                                    ),
                                )
                        layer_span.update(postfix="delta reference ready")

                        selected_activities = self.optimization_function.optimize(
                            full_layer_log,
                            hierarchy,
                            candidate_activities,
                            required_activities=native_activities,
                        )
                        layer_span.update(
                            metadata={"selected_activities": len(selected_activities)},
                            postfix=f"selected={len(selected_activities)}",
                        )
                    else:
                        selected_activities = native_activities
                        layer_span.update(
                            3,
                            metadata={"selected_activities": len(selected_activities)},
                            postfix=f"no candidates; selected={len(selected_activities)}",
                        )

                    with layer_span.child(
                        "Build final layer log",
                        metadata={"activities": len(selected_activities)},
                    ):
                        layer_log, included_event_records, included_activities = _build_layer_ocel(
                            log,
                            event_records,
                            object_to_type,
                            selected_object_types,
                            selected_activities,
                        )
                    layer_span.update(postfix="final layer log built")

                    net = self.discovery_technique.mine(layer_log)
                    layer_span.update(postfix="final net mined")

                    selected_subprocess_activities = frozenset(
                        activity
                        for activity in included_activities
                        if activity_layer_assignment.activity_to_layer.get(activity, layer) < layer
                    )
                    subprocesses = self.subprocess_miner.mine(net, selected_subprocess_activities)
                    layer_span.update(postfix=f"subprocesses={len(subprocesses)}")

                    with layer_span.child(
                        "Discover activity resources",
                        metadata={"activities": len(included_activities)},
                    ):
                        resources = _discover_activity_resources_with_threshold(
                            included_event_records,
                            object_to_type,
                            object_to_layer,
                            layer,
                            self.alpha_decision_parameter,
                        )
                    area = AdvancedProcessArea(
                        object_types=frozenset(selected_object_types),
                        activities=frozenset(included_activities),
                        net=net,
                        resources=resources,
                        subprocesses=subprocesses,
                    )
                    hierarchy = hierarchy.append(area)
                    layer_span.update(postfix="area appended")

                span.update(postfix=f"layer {layer} complete")

            self.hierarchy = hierarchy
            return hierarchy


def _filter_log_to_activities(log, activities: frozenset[str]):
    filtering_context = _build_ocel_filtering_context(log)
    return _build_ocel_from_filtering_context(
        _filter_ocel_filtering_context(filtering_context, activities)
    )


def _log_contains_only_activities(log, activities: frozenset[str]) -> bool:
    events = getattr(log, "events", None)
    if events is None or "ocel:activity" not in getattr(events, "columns", ()):
        return False
    return {
        str(activity)
        for activity in events["ocel:activity"].dropna().unique()
    } <= activities


def _share_checkpoint(component, checkpoint: CheckpointManager) -> None:
    if component is None:
        return
    if hasattr(component, "checkpoint"):
        component.checkpoint = checkpoint
    for attribute in (
        "check_set",
        "discovery_technique",
        "subprocess_miner",
        "collapsed_net_builder",
        "precision_calculator",
    ):
        child = getattr(component, attribute, None)
        if child is not None and child is not component:
            _share_checkpoint(child, checkpoint)


def _cached_subprocesses_for_candidate(
    full_subprocesses: tuple[Any, ...],
    candidate_activities: frozenset[str],
) -> tuple[Any, ...]:
    cached_subprocesses = []
    for subprocess in full_subprocesses or ():
        visible_activities = frozenset(
            str(activity)
            for activity in getattr(subprocess, "visible_activities", ())
            if activity is not None
        )
        if visible_activities and not visible_activities <= candidate_activities:
            continue
        cached_subprocesses.append(subprocess)
    return tuple(cached_subprocesses)


def _rematch_subprocesses_for_projected_net(
    candidate_net: AcceptingOCPN | None,
    cached_subprocesses: tuple[Any, ...],
    candidate_subprocess_activities: frozenset[str],
) -> tuple[Any, ...]:
    if candidate_net is None or not cached_subprocesses:
        return ()

    target_groups = {
        _subprocess_visible_activities(subprocess)
        for subprocess in cached_subprocesses
        if _is_nontrivial_subprocess(subprocess)
    }
    target_groups = {
        group
        for group in target_groups
        if group and group <= candidate_subprocess_activities
    }
    if not target_groups:
        return ()

    allowed_activities = frozenset().union(*target_groups)
    activity_to_layer = {
        activity: 1 if activity in allowed_activities else 2
        for activity in candidate_net.activities
    }
    regions = detect_subprocess_components(
        candidate_net.raw,
        activity_to_layer,
        reference_layer=2,
    )

    remaining_groups = set(target_groups)
    matched = []
    for region in sorted(regions, key=lambda item: (len(item.activities), sorted(item.activities))):
        activities = frozenset(str(activity) for activity in region.activities)
        if activities not in remaining_groups:
            continue
        matched.append(_to_framework_subprocess(region))
        remaining_groups.remove(activities)

    return tuple(matched)


def _subprocess_visible_activities(subprocess: Any) -> frozenset[str]:
    if isinstance(subprocess, dict):
        activities = subprocess.get("visible_activities") or subprocess.get("activities") or ()
    else:
        activities = getattr(subprocess, "visible_activities", None)
        if activities is None:
            raw_subprocess = getattr(subprocess, "raw", subprocess)
            activities = getattr(raw_subprocess, "activities", ())

    return frozenset(
        str(activity)
        for activity in activities or ()
        if activity is not None
    )


def _is_nontrivial_subprocess(subprocess: Any) -> bool:
    simple_subprocesses = getattr(subprocess, "simple_subprocesses", ())
    if simple_subprocesses:
        return any(
            len(getattr(simple_subprocess, "transitions", ())) > 1
            for simple_subprocess in simple_subprocesses
        )

    raw_subprocess = getattr(subprocess, "raw", subprocess)
    local_regions = getattr(raw_subprocess, "local_regions", ())
    if local_regions:
        return any(
            sum(
                1
                for vertex in getattr(local_region, "internal", ())
                if isinstance(vertex, PetriNet.Transition)
            ) > 1
            for local_region in local_regions
        )

    if isinstance(subprocess, dict):
        transitions = subprocess.get("transitions")
        if transitions is not None:
            return len(transitions) > 1

    return len(_subprocess_visible_activities(subprocess)) > 1


def _estimate_precision_for_activity_subset(
    precision_context: dict[str, Any],
    candidate_activities: frozenset[str],
) -> float:
    context_weights = precision_context.get("context_weights", {})
    prepared_original = precision_context.get("prepared_original", {})
    original_enabled_by_context = precision_context.get("original_enabled_by_context", {})
    if not context_weights or not prepared_original:
        return float(precision_context.get("precision", 0.0))

    precision_sum = 0.0
    precision_weight = 0
    log_enabled_by_context = prepared_original.get("log", {})

    for context_key, weight in context_weights.items():
        log_enabled = frozenset(
            str(label)
            for label in log_enabled_by_context.get(context_key, frozenset())
            if str(label) in candidate_activities
        )
        model_enabled = frozenset(
            str(label)
            for label in original_enabled_by_context.get(context_key, frozenset())
            if str(label) in candidate_activities
        )
        overlap = log_enabled & model_enabled
        if not model_enabled or not overlap:
            continue

        precision_sum += weight * (len(overlap) / len(model_enabled))
        precision_weight += weight

    return precision_sum / precision_weight if precision_weight else 0.0


def _log_metadata(log) -> dict[str, int]:
    metadata = {}
    events = _safe_len(getattr(log, "events", None))
    objects = _safe_len(getattr(log, "objects", None))
    relations = _safe_len(getattr(log, "relations", None))
    object_types = _safe_len(getattr(log, "object_types", None))

    if events is not None:
        metadata["events"] = events
    if objects is not None:
        metadata["objects"] = objects
    if relations is not None:
        metadata["relations"] = relations
    if object_types is not None:
        metadata["object_types"] = object_types
    return metadata


def _safe_len(value) -> int | None:
    if value is None:
        return None
    try:
        return len(value)
    except TypeError:
        return None


def _to_framework_subprocess(region) -> ObjectCentricSubprocess:
    simple_subprocesses = []
    for local_region in sorted(region.local_regions, key=lambda item: (
        item.object_type,
        getattr(item.source, "name", ""),
        getattr(item.target, "name", ""),
    )):
        transitions = frozenset(
            vertex
            for vertex in local_region.internal
            if isinstance(vertex, PetriNet.Transition)
        )
        places = frozenset(
            vertex
            for vertex in local_region.vertices
            if isinstance(vertex, PetriNet.Place)
        )
        arcs = frozenset(
            arc
            for transition in transitions
            for arc in tuple(transition.in_arcs) + tuple(transition.out_arcs)
            if arc.source in local_region.vertices
            and arc.target in local_region.vertices
        )
        simple_subprocesses.append(SimpleSubprocess(
            object_type=local_region.object_type,
            input_place=local_region.source,
            output_place=local_region.target,
            places=places,
            transitions=transitions,
            arcs=arcs,
            visible_activities=frozenset(local_region.activities),
            raw=local_region,
        ))

    return ObjectCentricSubprocess(
        simple_subprocesses=tuple(simple_subprocesses),
        visible_activities=frozenset(region.activities),
        id=getattr(region, "id", None),
        raw=region,
    )


def _raw_subprocesses(subprocesses):
    return tuple(
        getattr(subprocess, "raw", subprocess)
        for subprocess in subprocesses or ()
    )


def _derive_activity_layer_assignment(
    event_records,
    object_to_type,
    object_to_layer,
) -> ActivityLayerAssignment:
    activity_to_layer = {}
    for _, activity, _, event_objects in event_records:
        event_layer = min(object_to_layer[object_to_type[obj]] for obj in event_objects)
        current_layer = activity_to_layer.get(activity)
        if current_layer is None or event_layer < current_layer:
            activity_to_layer[activity] = event_layer

    activities_by_layer: dict[int, set[str]] = {}
    for activity, layer in activity_to_layer.items():
        activities_by_layer.setdefault(layer, set()).add(activity)

    return ActivityLayerAssignment(
        activity_to_layer=dict(activity_to_layer),
        activities_by_layer={
            layer: frozenset(activities)
            for layer, activities in activities_by_layer.items()
        },
    )


def _discover_activity_resources_with_threshold(
    event_records,
    object_to_type,
    object_to_layer,
    reference_layer,
    threshold,
) -> dict[str, frozenset[str]]:
    activity_event_counts = {}
    activity_resource_counts: dict[str, dict[str, int]] = {}

    for _, activity, _, event_objects in event_records:
        activity_event_counts[activity] = activity_event_counts.get(activity, 0) + 1
        higher_layer_types = {
            object_to_type[obj]
            for obj in event_objects
            if object_to_layer[object_to_type[obj]] > reference_layer
        }
        for object_type in higher_layer_types:
            activity_resource_counts.setdefault(activity, {})
            activity_resource_counts[activity][object_type] = (
                activity_resource_counts[activity].get(object_type, 0) + 1
            )

    resources = {}
    for activity, total_count in activity_event_counts.items():
        resources[activity] = frozenset(
            object_type
            for object_type, occurrence_count in activity_resource_counts.get(activity, {}).items()
            if total_count and occurrence_count / total_count >= threshold
        )
    return resources


def _merge_hierarchy_nets(
    hierarchy: AdvancedProcessAreaHierarchy,
    *,
    collapse: bool,
    collapsed_net_builder: CollapsedNetBuilder,
) -> AcceptingOCPN | None:
    raws = []
    for area in hierarchy.areas:
        net = area.net
        if collapse:
            net = collapsed_net_builder.collapse(area.net, area.subprocesses)
        if net is not None:
            raws.append(net.raw)

    return AcceptingOCPN.from_raw(_merge_ocpn_models(*raws))


def _complexity(net: AcceptingOCPN | None) -> float:
    if net is None:
        return 0.0
    return float(NetQuality(net.raw).complexity())


def _f1_quality(simplicity_gain: float, information_loss: float) -> float:
    precision_factor = 1 - information_loss
    denominator = simplicity_gain + precision_factor
    if denominator <= 0:
        return 0.0
    return 2 * simplicity_gain * precision_factor / denominator
