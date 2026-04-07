from datetime import datetime

DATEFORMAT = "%Y-%m-%d %H:%M:%S"

# Cache prepared data by OCEL object identity
_PREPARED_TOTEM_CACHE: dict[int, tuple] = {}


def get_all_event_objects(ocel, event_id):
    return ocel.get_value(event_id, "event_objects")


def _build_totem_data(ocel):
    """
    Internal builder for shared preprocessing.
    """
    o_min_times: dict[str, datetime] = {}
    o_max_times: dict[str, datetime] = {}
    o2o: dict[str, dict[str, set[str]]] = {}
    type_to_object: dict[str, set[str]] = {}

    # Pass 1: gather event-based object relations and object lifespans
    for event_id in ocel.events["_eventId"]:

        event_timestamp = ocel.get_event_timestamp(event_id)
        event_objects = get_all_event_objects(ocel, event_id)

        objects_by_type = {
            obj_type: set(ocel.get_event_objects_by_type(event_id, obj_type))
            for obj_type in ocel.object_types
        }

        for obj_type, objects in objects_by_type.items():
            if objects:
                type_to_object.setdefault(obj_type, set()).update(objects)

        for obj in event_objects:
            o2o.setdefault(obj, {})
            for obj_type in ocel.object_types:
                o2o[obj].setdefault(obj_type, set())
                o2o[obj][obj_type].update(objects_by_type[obj_type])

            if event_timestamp is not None:
                if obj not in o_min_times or o_min_times[obj] is None or event_timestamp < o_min_times[obj]:
                    o_min_times[obj] = event_timestamp
                if obj not in o_max_times or o_max_times[obj] is None or event_timestamp > o_max_times[obj]:
                    o_max_times[obj] = event_timestamp

    # Reverse lookup
    object_to_type: dict[str, str] = {}
    for obj_type, objects in type_to_object.items():
        for obj in objects:
            object_to_type[obj] = obj_type

    # Pass 2: include explicit o2o graph edges, symmetrically
    for source_obj, target_obj in ocel.o2o_graph_edges:
        source_type = object_to_type.get(source_obj)
        target_type = object_to_type.get(target_obj)

        if source_type is None or target_type is None:
            continue

        o2o.setdefault(source_obj, {}).setdefault(target_type, set()).add(target_obj)
        o2o.setdefault(target_obj, {}).setdefault(source_type, set()).add(source_obj)

    return o_min_times, o_max_times, o2o, type_to_object


def _prepare_totem_data(ocel):
    """
    Shared preprocessing used by both temporal-relation and cardinality computation.

    Caches results by OCEL object identity.
    """
    cache_key = id(ocel)

    if cache_key not in _PREPARED_TOTEM_CACHE:
        _PREPARED_TOTEM_CACHE[cache_key] = _build_totem_data(ocel)

    return _PREPARED_TOTEM_CACHE[cache_key]


def clear_totem_cache(ocel=None):
    """
    Clear the whole cache or only the entry for one OCEL instance.
    """
    if ocel is None:
        _PREPARED_TOTEM_CACHE.clear()
    else:
        _PREPARED_TOTEM_CACHE.pop(id(ocel), None)
