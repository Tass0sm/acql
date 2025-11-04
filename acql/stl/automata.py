from functools import reduce

import spot
from spot import formula as sf
import buddy

from acql.stl import fold_tree, fold_spot_formula, STLPredicate # TODO


def get_spot_formula_and_aps(exp):
    ap_i = 0
    spot_aps = {}
    aps = {}

    def to_spot_helper(root, children):
        nonlocal aps, ap_i

        children = list(children)

        if isinstance(root, STLPredicate):
            if "true" in root.info:
                return sf.tt()
            elif "false" in root.info:
                return sf.ff()
            else:
                ap = sf.ap(f"ap_{ap_i}")
                root.info["ap_i"] = ap_i
                spot_aps[ap_i] = ap
                aps[ap_i] = root
                ap_i += 1
                return ap
        elif isinstance(root, STLNegation):
            return sf.Not(children[0])
        elif isinstance(root, STLAnd):
            return sf.And(children)
        elif isinstance(root, STLOr):
            return sf.Or(children)
        elif isinstance(root, STLDisjunction):
            return sf.Or(children)
        elif isinstance(root, STLImplies):
            return sf.Implies(children)
        elif isinstance(root, STLNext):
            return sf.X(children[0])
        elif isinstance(root, STLUntimedEventually):
            return sf.F(children[0])
        elif isinstance(root, STLTimedEventually):
            raise NotImplementedError()
            # return sf.F(children)
        elif isinstance(root, STLUntimedUntil):
            return sf.U(children[0], children[1])
        elif isinstance(root, STLTimedUntil):
            raise NotImplementedError()
            # return sf.U(children)
        elif isinstance(root, STLUntimedAlways):
            return sf.G(children[0])
        elif isinstance(root, STLTimedAlways):
            raise NotImplementedError()
            # return sf.G(children)
        else:
            raise NotImplementedError(f"Root = {root}")

    return fold_tree(to_spot_helper, exp), aps, spot_aps


# TODO: figure out best way to do this. This is hacky.
def eval_spot_formula(form, bit_vector, n_bits):

    ap_vals = {}
    for i in range(n_bits):
        ap_vals[f"ap_{i}"] = bool(bit_vector & 2**i)

    def eval_spot_helper(root, children):
        nonlocal ap_vals

        children = list(children)

        k = root.kind()

        if k == spot.op_ff:
            return False
        elif k == spot.op_tt:
            return True
        elif k == spot.op_ap:
            return ap_vals[root.ap_name()]
        elif k == spot.op_Not:
            return not children[0]
        elif k == spot.op_And:
            return children[0] and children[1]
        elif k == spot.op_Or:
            return children[0] or children[1]
        else:
            raise NotImplementedError(f"Formula {root} with kind = {k}")

    return fold_spot_formula(eval_spot_helper, form)


def get_safety_conditions(aut):
    scc_info = spot.scc_info(aut)
    safety_conditions = {}

    # TODO: Check that common states are always preserved.
    canon_aut = spot.scc_filter_states(aut)
    bdd_dict = canon_aut.get_dict()

    for scc_i in range(scc_info.scc_count()):
        if scc_info.is_useful_scc(scc_i):
            for s in scc_info.states_of(scc_i):
                bdds = []

                for edge in canon_aut.out(s):
                    bdd_i = edge.cond
                    bdds.append(bdd_i)

                undefined_transition_bdd = reduce(buddy.bdd_or, bdds, buddy.bddfalse)
                undefined_transition_bdd = buddy.bdd_not(undefined_transition_bdd)
                undefined_transition_form = spot.bdd_to_formula(undefined_transition_bdd, bdd_dict)

                safety_conditions[s] = undefined_transition_form
        else:
            for s in scc_info.states_of(scc_i):
                incoming_conds = []

                for edge in aut.edges():
                    if edge.src != s and edge.dst == s:
                        incoming_conds.append(edge.cond)

                incoming_cond_bdd = reduce(buddy.bdd_or, incoming_conds, buddy.bddfalse)
                incoming_cond_form = spot.bdd_to_formula(incoming_cond_bdd, bdd_dict)

                safety_conditions[s] = incoming_cond_form # spot.formula.tt()

    return safety_conditions

def make_safety_automaton(aut):
    canon_aut = spot.scc_filter_states(aut)
    safety_aut = spot.make_twa_graph(canon_aut, spot.twa_prop_set.all(), True)

    m = safety_aut.set_buchi()
    for edge in safety_aut.edges():
        edge.acc = m

    bdd_dict = safety_aut.get_dict()

    if aut.num_states() == canon_aut.num_states():
        outgoing_condition_dict = { k: spot.formula.ff() for k in range(safety_aut.num_states()) }
        return safety_aut, outgoing_condition_dict
    else:
        outgoing_condition_dict = {}
        q_trap = safety_aut.new_state()

        breakpoint()

        for state in range(safety_aut.num_states()):
            bdds = []
    
            for edge in safety_aut.out(state):
                bdd_i = edge.cond
                bdds.append(bdd_i)
    
            undefined_transition_bdd = reduce(buddy.bdd_or, bdds, buddy.bddfalse)
            undefined_transition_bdd = buddy.bdd_not(undefined_transition_bdd)
            undefined_transition_form = spot.bdd_to_formula(undefined_transition_bdd, bdd_dict)

            if not spot.formula.is_ff(undefined_transition_form):
                safety_aut.new_edge(state, q_trap, undefined_transition_bdd)
    
            outgoing_condition_dict[state] = undefined_transition_form
    
        # for state in range(safety_aut.num_states()):
        #     outgoing_condition_dict[q_trap] = spot.formula.tt()

        # return spot.minimize_obligation(safety_aut)
        return safety_aut, outgoing_condition_dict

def make_liveness_automaton(aut):
    canon_aut = spot.scc_filter_states(aut)    
    liveness_aut = spot.make_twa_graph(canon_aut, spot.twa_prop_set.all(), True)

    bdd_dict = liveness_aut.get_dict() 
    q_trap = liveness_aut.new_state()

    for state in range(liveness_aut.num_states()):
        bdds = []

        for edge in liveness_aut.out(state):
            bdd_i = edge.cond
            bdds.append(bdd_i)

        undefined_transition_bdd = reduce(buddy.bdd_or, bdds, buddy.bddfalse)
        undefined_transition_bdd = buddy.bdd_not(undefined_transition_bdd)
        undefined_transition_form = spot.bdd_to_formula(undefined_transition_bdd, bdd_dict)

        liveness_aut.new_acc_edge(state, q_trap, undefined_transition_bdd)

    return spot.minimize_obligation(liveness_aut)

def make_just_liveness_automaton(aut):
    canon_aut = spot.scc_filter_states(aut)
    just_liveness_aut = spot.make_twa_graph(canon_aut, spot.twa_prop_set.all(), True)

    for state in range(just_liveness_aut.num_states()):
        bdds = []

        # Get the undefined transition
        for edge in just_liveness_aut.out(state):
            bdd_i = edge.cond
            bdds.append(bdd_i)

        undefined_transition_bdd = reduce(buddy.bdd_or, bdds, buddy.bddfalse)
        undefined_transition_bdd = buddy.bdd_not(undefined_transition_bdd)

        # remove the variables in the undefined transition from all existing
        # edges via existential quantification
        # for edge in just_liveness_aut.out(state):
        #     edge.cond = buddy.bdd_exist(edge.cond, buddy.bdd_support(undefined_transition_bdd))

        # make undefined transition a self-loop
        just_liveness_aut.new_edge(state, state, undefined_transition_bdd)

    return just_liveness_aut

def get_outgoing_conditions(aut):
    bdd_dict = aut.get_dict()
    outgoing_condition_dict = {}

    for state in range(aut.num_states()):
        bdds = []
        # Union all the conditions for the out-going edges
        for edge in aut.out(state):
            if edge.src != edge.dst:
                bdd_i = edge.cond
                bdds.append(bdd_i)

        bdd = reduce(buddy.bdd_or, bdds, buddy.bddfalse)
        f = spot.bdd_to_formula(bdd, bdd_dict)

        # size = buddy.bdd_nodecount(buddy.bdd_support(bdd))
        outgoing_condition_dict[state] = bdd

    return outgoing_condition_dict
