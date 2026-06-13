from pulp import LpProblem, LpVariable, LpInteger, value, LpMinimize, lpSum, LpStatus, PULP_CBC_CMD


def solve(types, scores_push, scores_pull,
          alpha=1.0, beta=1.0, margin_scale=0, verbose=False):
    types = list(types)

    K = len(types)

    prob = LpProblem("layered_hierarchy_margin", LpMinimize)
    z = LpVariable.dicts("layer", types, lowBound=1, upBound=K, cat=LpInteger)

    abs_diff = {}
    hinge = {}

    for idx_i, i in enumerate(types):
        for j in types[idx_i + 1:]:
            p_ij = scores_pull.get((i, j), 0.0)
            if p_ij > 0:
                abs_diff[(i, j)] = LpVariable(f"absdiff_{i}_{j}", lowBound=0)
                prob += abs_diff[(i, j)] >= z[i] - z[j]
                prob += abs_diff[(i, j)] >= z[j] - z[i]

    for i in types:
        for j in types:
            if i == j:
                continue
            s_ij_raw = scores_push.get((i, j), 0.0)
            s_ij = max(0.0, s_ij_raw)
            if s_ij > 0:
                m_ij = 1.0 + margin_scale * s_ij
                hinge[(i, j)] = LpVariable(f"hinge_{i}_{j}", lowBound=0)
                prob += hinge[(i, j)] >= m_ij - z[i] + z[j]

    prob += (
            alpha * lpSum(scores_pull[(i, j)] * abs_diff[(i, j)]
                          for (i, j) in abs_diff)
            +
            beta * lpSum(max(0.0, scores_push[(i, j)]) * hinge[(i, j)]
                         for (i, j) in hinge)
    )

    status = prob.solve(PULP_CBC_CMD(msg=0))
    raw_solution = {
        i: int(round(value(z[i]) if value(z[i]) is not None else 1))
        for i in types
    }
    layer_mapping = {
        layer: index + 1
        for index, layer in enumerate(sorted(set(raw_solution.values())))
    }
    solution = {
        object_type: layer_mapping[layer]
        for object_type, layer in raw_solution.items()
    }

    if verbose:
        print("Status:", LpStatus[status])
        print("Objective:", value(prob.objective))
        for i in types:
            print(i, solution[i])

    return solution
