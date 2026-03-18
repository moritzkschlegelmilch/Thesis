from pulp import LpProblem, LpVariable, LpMaximize, LpInteger, value, LpMinimize, lpSum, LpStatus


def solve(types, scores_push, scores_pull, tau):
    prob = LpProblem("variable_size_model", LpMinimize)

    x = LpVariable.dicts("ot", types, lowBound=0, cat="Integer")

    push_abs = {}
    pull_abs = {}

    for ot1 in types:
        for ot2 in types:
            if scores_push[ot1, ot2] > 0:
                expr_push = (x[ot1] - x[ot2]) - (scores_push[ot1, ot2] * x[ot1])
                push_abs[(ot1, ot2)] = LpVariable(f"push_abs_{ot1}_{ot2}", lowBound=0)
                prob += push_abs[(ot1, ot2)] >= expr_push
                prob += push_abs[(ot1, ot2)] >= -expr_push

            if scores_pull[ot1, ot2] > 0:
                expr_pull = (x[ot1] - x[ot2]) - (1 - (scores_pull[ot1, ot2] * x[ot1]))
                pull_abs[(ot1, ot2)] = LpVariable(f"pull_abs_{ot1}_{ot2}", lowBound=0)
                prob += pull_abs[(ot1, ot2)] >= expr_pull
                prob += pull_abs[(ot1, ot2)] >= -expr_pull

    prob += (
            lpSum(push_abs[(ot1, ot2)] for ot1 in types for ot2 in types if (ot1, ot2) in push_abs)
            +
            lpSum(pull_abs[(ot1, ot2)] for ot1 in types for ot2 in types if (ot1, ot2) in pull_abs)
    )

    for ot1 in types:
        for ot2 in types:
            if scores_pull[ot1, ot2] > 0:
                print(ot1, ot2, scores_pull[ot1, ot2])


    # TODO optimize
    for ot_1 in types:
        for ot_2 in types:
            if scores_push[ot_1, ot_2] > tau:
                prob += x[ot_1] - x[ot_2] >= 1
    for ot_ in types:
        prob += x[ot_] <= len(types)

    status = prob.solve()

    print("Status:", LpStatus[status])
    print("Objective:", value(prob.objective))
    for v in prob.variables():
        print(v.name, "=", value(v))