from typing import Dict, Set, List
import networkx as nx
from datetime import datetime
from pulp import *
import graphviz
import os
from pathlib import Path

TR_TOTAL = "total"
TR_DEPENDENT = "D"
TR_INITIATING = "I"

# Event cardinality constants
LC_TOTAL = "total"
LC_ONE = "1"
LC_MANY = "1..*"

DATEFORMAT = "%Y-%m-%d %H:%M:%S"

def get_all_event_objects(ocel, event_id):
    return ocel.get_value(event_id, "event_objects")
def totemDiscovery(ocel, tau=0.9):
    """
    Given an Object Centric Event Log, compute the temporal graph and related information.
    :param ocel: The Object Centric Event Log to analyze.
    :param tau: The threshold for determining strong relations (default is 0.9).
    :return: A Totem object containing the temporal graph and related information.
    """
    # object type to event type dict
    obj_typ_to_ev_type: dict[str, set[str]] = dict()
    all_event_types = set()

    # temporal relations results
    h_temporal_relations: dict[tuple[str, str], dict[str, int]] = (
        dict()
    )  # stores all the temporal relations found
    # event cardinality results
    h_event_cardinalities: dict[tuple[str, str], dict[str, int]] = (
        dict()
    )  # stores all the temporal cardinalities found
    # event cardinality results
    h_log_cardinalities: dict[tuple[str, str], dict[str, int]] = (
        dict()
    )  # stores all the temporal cardinalities found

    # object min times (omint_L(o))
    o_min_times: dict[str, datetime] = (
        dict()
    )  # str identifier of the object maps to the earliest time recorded for that object in the event log
    # object max times (omaxt_L(o))
    o_max_times: dict[str, datetime] = (
        dict()
    )  # str identifier of the object maps to the last time recorded for that object in the event log

    # get a list of all object types (or variable that is filled while passing through the process executions)
    type_relations: set[set[str, str]] = set()  # stores all connected types

    o2o_o2o: dict[str, dict[str, set[str]]] = (
        dict()
    )  # dict that describes which objects are connected to which types and for each type which object
    o2o_e2o: dict[str, dict[str, set[str]]] = dict()
    o2o: dict[str, dict[str, set[str]]] = dict()

    # a mapping from type to its objects
    type_to_object = dict()

    for px in (
        ocel.process_executions
    ):  # TODO: for ev in all events instead of process_executions
        for ev in px:
            ev_timestamp = ocel.get_event_timestamp(ev)  # use unix timestamp directly

            objects_of_event = get_all_event_objects(ocel, ev)
            for obj in objects_of_event:
                # o2o updating
                o2o.setdefault(obj, dict())
                for type in ocel.object_types:
                    o2o[obj].setdefault(type, set())
                    o2o[obj][type].update(
                        # ocel.get_value(ev, type))  # add all objects connected via e2o to each object involved
                        ocel.get_event_objects_by_type(ev, type)
                    )  # add all objects connected via e2o to each object involved
                # update lifespan information
                o_min_times.setdefault(obj, ev_timestamp)
                if (
                    ev_timestamp < o_min_times[obj]
                ):
                    o_min_times[obj] = ev_timestamp
                o_max_times.setdefault(obj, ev_timestamp)
                if (
                    ev_timestamp > o_max_times[obj]
                ):
                    o_max_times[obj] = ev_timestamp


            eventtype = ocel.get_event_activity(ev)
            all_event_types.add(eventtype)
            for type in ocel.object_types:
                if len(ocel.get_event_objects_by_type(ev, type)) > 0:
                    obj_typ_to_ev_type.setdefault(type, set())
                    obj_typ_to_ev_type[type].add(eventtype)

            # compute event cardinality
            involved_types = []
            obj_count_per_type = dict()
            for type in ocel.object_types:
                obj_list = ocel.get_event_objects_by_type(ev, type)
                if not obj_list:
                    continue
                else:
                    type_to_object.setdefault(type, set())
                    type_to_object[type].update(obj_list)
                    involved_types.append(type)
                    obj_count_per_type[type] = len(obj_list)
            for t1 in involved_types:
                for t2 in involved_types:
                    if t1 != t2:
                        type_relations.add(frozenset({t1, t2}))

    for source_o, target_o in ocel.o2o_graph_edges:
        type_of_source_o = None
        type_of_target_o = None

        for type in ocel.object_types:
            if source_o in type_to_object.get(type, set()):
                type_of_source_o = type
            if target_o in type_to_object.get(type, set()):
                type_of_target_o = type

        if type_of_source_o is None or type_of_target_o is None:
            continue

        # forward direction
        o2o.setdefault(source_o, dict())
        o2o[source_o].setdefault(type_of_target_o, set())
        o2o[source_o][type_of_target_o].add(target_o)

        # reverse direction: enforce symmetric closure
        o2o.setdefault(target_o, dict())
        o2o[target_o].setdefault(type_of_source_o, set())
        o2o[target_o][type_of_source_o].add(source_o)

    for type_source in ocel.object_types:
        for type_target in ocel.object_types:
            h_temporal_relations.setdefault((type_source, type_target), dict())
            for obj in type_to_object[type_source]:
                h_log_cardinalities.setdefault((type_source, type_target), dict())
                h_log_cardinalities[(type_source, type_target)].setdefault(LC_TOTAL, 0)
                h_log_cardinalities[(type_source, type_target)][LC_TOTAL] += 1

                cardinality = len(o2o[obj][type_target])
                if cardinality == 1:
                    target_obj = next(iter(o2o[obj][type_target]))
                    if len(o2o[target_obj][type_source]) == 1:
                        h_log_cardinalities[(type_source, type_target)].setdefault(
                            LC_ONE, 0
                        )
                        h_log_cardinalities[(type_source, type_target)][LC_ONE] += 1
                elif cardinality > 1:
                    h_log_cardinalities[(type_source, type_target)].setdefault(
                        LC_MANY, 0
                    )
                    h_log_cardinalities[(type_source, type_target)][LC_MANY] += 1
                # compute temporal relations
                for obj_target in o2o[obj][type_target]:
                    h_temporal_relations[(type_source, type_target)].setdefault(
                        TR_TOTAL, 0
                    )
                    h_temporal_relations[(type_source, type_target)][TR_TOTAL] += 1
                    if (
                        o_min_times[obj_target]
                        <= o_min_times[obj]
                        <= o_max_times[obj]
                        <= o_max_times[obj_target]
                    ):
                        h_temporal_relations[(type_source, type_target)].setdefault(
                            TR_DEPENDENT, 0
                        )
                        h_temporal_relations[(type_source, type_target)][
                            TR_DEPENDENT
                        ] += 1
                    if (
                        o_min_times[obj]
                        <= o_max_times[obj]
                        <= o_min_times[obj_target]
                        <= o_max_times[obj_target]
                    ) or (
                        o_min_times[obj]
                        < o_min_times[obj_target]
                        <= o_max_times[obj]
                        < o_max_times[obj_target]
                    ):
                        h_temporal_relations[(type_source, type_target)].setdefault(
                            TR_INITIATING, 0
                        )
                        h_temporal_relations[(type_source, type_target)][
                            TR_INITIATING
                        ] += 1

    return h_temporal_relations, h_log_cardinalities
