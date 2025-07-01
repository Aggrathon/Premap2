#########################################################################
##                                                                     ##
##     This program is licenced under the BSD 3-Clause License,        ##
##        contained in the LICENCE file in this directory.             ##
##                                                                     ##
#########################################################################
"""Branch and bound for activation space split."""
import gc
import time
import os
import numpy as np
import torch
import copy
import pickle

from auto_LiRPA.utils import stop_criterion_sum, stop_criterion_batch_any, stop_criterion_batch_topk
from branching_domains import merge_domains_params, SortedReLUDomainList, BatchedReLUDomainList
from branching_heuristics import choose_node_parallel_FSB, choose_node_parallel_crown, choose_node_parallel_kFSB, choose_node_parallel_preimg

import arguments
from cut_utils import fetch_cut_from_cplex, generate_cplex_cuts, clean_net_mps_process, cplex_update_general_beta

try:
    from premap2 import calc_initial_coverage, calc_branched_coverage, calc_samples, calc_constraints, select_node_batch, split_node_batch, save_premap
except ImportError:
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).parent.parent.parent / 'src'))
    from premap2 import calc_initial_coverage, calc_branched_coverage, calc_samples, calc_constraints, select_node_batch, split_node_batch, save_premap

Visited, Flag_first_split = 0, True
all_node_split = False
total_pickout_time = total_decision_time = total_solve_time = total_add_time = 0.0


def batch_verification(num_unstable, d, net, batch, pre_relu_indices, growth_rate, fix_intermediate_layer_bounds=True,
                    stop_func=stop_criterion_sum, multi_spec_keep_func=lambda x: torch.all(x, dim=-1), bound_lower=True, bound_upper=False):
    global Visited, Flag_first_split
    global total_pickout_time, total_decision_time, total_solve_time, total_add_time

    opt_intermediate_beta = False
    branching_method = arguments.Config['bab']['branching']['method']
    branching_reduceop = arguments.Config['bab']['branching']['reduceop']
    get_upper_bound = arguments.Config["bab"]["get_upper_bound"]
    branching_candidates = arguments.Config["bab"]["branching"]["candidates"]
    debug = arguments.Config["debug"]["asserts"]

    total_time = time.time()

    pickout_time = time.time()

    domains_params = d.pick_out(batch=batch, device=net.x.device)
    # Note that lAs is removed
    mask, orig_lbs, orig_ubs, slopes, betas, intermediate_betas, selected_domains, cs, rhs = domains_params

    if selected_domains is None:
        total_vol = sum(sd.preimg_vol for sd in d.domains)
        cov_vol = sum(sd.preimg_vol * sd.preimg_cov for sd in d.domains)
        return cov_vol / max(1e-8, total_vol), True
    
    batch = len(selected_domains)
    history = [sd.history for sd in selected_domains]
    
    pickout_time = time.time() - pickout_time
    total_pickout_time += pickout_time

    decision_time = time.time()

    split_history = [sd.split_history for sd in selected_domains]

    # Here we check the length of current domain list.
    # If the domain list is small, we can split more layers.
    # min_batch_size = min(arguments.Config["solver"]["min_batch_size_ratio"]*arguments.Config["solver"]["batch_size"], batch)

    # if orig_lbs[0].shape[0] < min_batch_size:
    #     # Split multiple levels, to obtain at least min_batch_size domains in this batch.
    #     split_depth = int(np.log(min_batch_size)/np.log(2))

    #     if orig_lbs[0].shape[0] > 0:
    #         split_depth = max(int(np.log(min_batch_size/orig_lbs[0].shape[0])/np.log(2)), 0)
    #     split_depth = max(split_depth, 1)
    # else:
    split_depth = 1

    sample_num = arguments.Config["preimage"]["sample_num"]
    sample_instability = arguments.Config["preimage"]["instability"]
    heuristics = arguments.Config["preimage"]["heuristics"]
    tighten = arguments.Config["preimage"]["tighten_bounds"]
    samples = [calc_samples(domain.samples, net.model_ori, sample_num, domain.history, debug=debug) for domain in selected_domains]
    if sample_instability:
        eps = torch.finfo(samples[0].X.dtype).eps * 4
        for i, (lb, ub) in enumerate(zip(orig_lbs[:-1], orig_ubs)):
            for j, s in enumerate(samples):
                amin = s.activations[i].min(0)[0]
                amax = s.activations[i].max(0)[0]
                lb[j] = torch.where(lb[j] <= amin, torch.where((lb[j] >= 0.0) | (amin < 0.0), lb[j], -eps), amin - eps)
                ub[j] = torch.where(ub[j] >= amax, torch.where((ub[j] <= 0.0) | (amax > 0.0), ub[j], +eps), amax + eps)
    if debug:
        assert all(len(domain.samples) > 0 for domain in selected_domains)
        assert all(len(s) > 0 for s in samples)
    for domain in selected_domains:
        domain.samples = None # Free memory of old samples

    # print("batch: ", orig_lbs[0].shape, "pre split depth: ", split_depth)
    # Increase the maximum number of candidates for fsb and kfsb if there are more splits needed.
    branching_candidates = max(branching_candidates, split_depth)

    if branching_method == 'babsr':
        branching_decision, split_depth = choose_node_parallel_crown(orig_lbs, orig_ubs, mask, net, pre_relu_indices, lAs,
                                                        batch=batch, branching_reduceop=branching_reduceop, split_depth=split_depth, cs=cs, rhs=rhs)
    elif branching_method == 'fsb':
        branching_decision, split_depth = choose_node_parallel_FSB(orig_lbs, orig_ubs, mask, net, pre_relu_indices, lAs,
                                        branching_candidates=branching_candidates, branching_reduceop=branching_reduceop,
                                        slopes=slopes, betas=betas, history=history, split_depth=split_depth, cs=cs, rhs=rhs)
    elif branching_method.startswith('kfsb'):
        branching_decision, split_depth = choose_node_parallel_kFSB(orig_lbs, orig_ubs, mask, net, pre_relu_indices, lAs,
                                        branching_candidates=branching_candidates, branching_reduceop=branching_reduceop,
                                        slopes=slopes, betas=betas, history=history, split_depth=split_depth, cs=cs, rhs=rhs,
                                        method=branching_method)
    elif branching_method in ('preimg', 'premap'):
        branching_decision = select_node_batch(samples, mask, orig_lbs, orig_ubs, cs, heuristics)
    else:
        raise ValueError(f'Unsupported branching method "{branching_method}" for relu splits.')

    if split_depth > 1:
        raise NotImplementedError()

    decision_time = time.time() - decision_time
    total_decision_time += decision_time
    solve_time = time.time()
    single_node_split = True
    num_copy = (2**(split_depth-1))


    # Split
    ret = split_node_batch(net, d, bound_lower, orig_lbs, orig_ubs, slopes, betas, intermediate_betas, selected_domains, cs, rhs, history, samples, branching_decision, tighten=tighten, debug=debug)
    orig_lbs, orig_ubs, slopes, betas, intermediate_betas, selected_domains, cs, rhs, history, left_right_his, samples, branching_decision = ret

    if len(branching_decision) == 0:
        total_vol = sum(sd.preimg_vol for sd in d.domains)
        cov_vol = sum(sd.preimg_vol * sd.preimg_cov for sd in d.domains)
        total_solve_time += time.time() - solve_time
        print('Preimage volume:', total_vol)
        print('Coverage quota:', cov_vol / max(1e-8, total_vol))
        return cov_vol / max(1e-8, total_vol), len(d) == 0

    # if len(sample_left_idx) == 0 and len(sample_right_idx) == 0:
    #     flag_next_split 
    # Caution: we use "all" predicate to keep the domain when multiple specs are present: all lbs should be <= threshold, otherwise pruned
    # maybe other "keeping" criterion needs to be passed here
    split = {"decision": [[bd] for bd in branching_decision], "coeffs": [[1.0] * len(branching_decision)]}
    ret = net.get_lower_bound(orig_lbs, orig_ubs, split, slopes=slopes, history=history,samples=samples,
                                split_history=split_history, fix_intermediate_layer_bounds=fix_intermediate_layer_bounds, betas=betas,
                                single_node_split=single_node_split, intermediate_betas=intermediate_betas, cs=cs, decision_thresh=rhs, rhs=rhs,
                                stop_func=stop_func(torch.cat([rhs, rhs])), multi_spec_keep_func=multi_spec_keep_func, bound_lower=bound_lower, bound_upper=bound_upper)

    dom_ub, dom_lb, dom_ub_point, lAs, A, dom_lb_all, dom_ub_all, slopes, split_history, betas, intermediate_betas, primals, dom_cs = ret
    calc_constraints(net, samples, left_right_his, dom_lb_all, dom_ub_all, debug=debug)
    solve_time = time.time() - solve_time
    total_solve_time += solve_time
    add_time = time.time()
    batch = len(branching_decision)
    # If intermediate layers are not refined or updated, we do not need to check infeasibility when adding new domains.
    check_infeasibility = not (single_node_split and fix_intermediate_layer_bounds)

    depths = [domain.depth + split_depth - 1 for domain in selected_domains] * num_copy * 2

    # NOTE evaluate the coverage quality after splitting
    if bound_lower:
        cov_subdomain_info, A_b_dict = calc_branched_coverage(A, samples, selected_domains, True, debug)
    if bound_upper:
        cov_subdomain_info, A_b_dict = calc_branched_coverage(A, samples, selected_domains, False, debug)

    if arguments.Config["solver"]["beta-crown"]["all_node_split_LP"]:
        for domain_idx in range(len(depths)):
            # get tot_ambi_nodes
            dlb, dub = [dlbs[domain_idx: domain_idx + 1] for dlbs in dom_lb_all],  [dubs[domain_idx: domain_idx + 1] for dubs in dom_ub_all]
            decision_threshold = rhs.to(dom_lb[0].device, non_blocking=True)[domain_idx if domain_idx < (len(dom_lb)//2) else domain_idx - (len(dom_lb)//2)]
            # print(depths[domain_idx] + 1, dlb[-1], decision_threshold, torch.all(dlb[-1] <= decision_threshold))
            if depths[domain_idx] + 1 == net.tot_ambi_nodes  and torch.all(dlb[-1] <= decision_threshold):
                lp_status, dlb, adv = net.all_node_split_LP(dlb, dub, decision_threshold)
                print(f"using lp to solve all split node domain {domain_idx}/{len(dom_lb)}, results {dom_lb[domain_idx]} -> {dlb}, {lp_status}")
                # import pdb; pdb.set_trace()
                if lp_status == "unsafe":
                    # unsafe cases still needed to be handled! set to be unknown for now!
                    all_node_split = True
                    return dlb, np.inf
                dom_lb_all[-1][domain_idx] = dlb
                dom_lb[domain_idx] = dlb
    rhs = torch.cat((rhs, rhs))
    if bound_lower:
        d.add(cov_subdomain_info, A_b_dict["lA"], A_b_dict["lbias"], dom_lb, dom_ub, dom_lb_all, dom_ub_all, history, left_right_his, depths, slopes, betas, split_history,
                branching_decision, rhs, intermediate_betas, check_infeasibility, dom_cs, (2*num_copy)*batch, samples=samples)
    if bound_upper: 
        d.add(cov_subdomain_info, A_b_dict["uA"], A_b_dict["ubias"], dom_lb, dom_ub, dom_lb_all, dom_ub_all, history, left_right_his, depths, slopes, betas, split_history,
                branching_decision, rhs, intermediate_betas, check_infeasibility, dom_cs, (2*num_copy)*batch, samples=samples)
    total_vol = 0
    cov_vol = 0
    
    # if len(d) < old_d_len:
    #     print("check why")
    #     print("should not happen")
    for i, subdm in enumerate(d.domains):
        total_vol += subdm.preimg_vol
        cov_vol += subdm.preimg_vol * subdm.preimg_cov
        
    total_cov_quota = cov_vol / max(1e-8, total_vol)
    print('length of domains:', len(d))
    print('Preimage volume:', total_vol)
    print('Coverage quota:', cov_vol / max(1e-8, total_vol))
    if bound_lower:
        print('Split lower bound (avg):', dom_lb.mean().cpu().item())
    if bound_upper:
        print('Split upper bound (avg):', dom_ub.mean().cpu().item())
    print('Split layer bound gap (avg):', [(ub - lb).sum().cpu().item() / (ub > lb).count_nonzero().cpu().item() for lb, ub in zip(dom_lb_all, dom_ub_all)])
    if debug:
        assert all((ub >= lb).all().cpu().item() for lb, ub in zip(dom_lb_all, dom_ub_all))
    split_all = True
    for i, subdm in enumerate(d.domains):
        if subdm.valid:
            split_all = False
            break
    if split_all:
        print("exhausting search achieved")
        return total_cov_quota, split_all

    

    # Visited += (len(selected_domains) * num_copy) * 2 - (len(d) - old_d_len)
    Visited += len(samples)
    # if len(d) > 0:
    #     if get_upper_bound:
    #         print('Current worst splitting domains [lb, ub] (depth):')
    #     else:
    #         print('Current worst splitting domains lb-rhs (depth):')
    #     if arguments.Config["bab"]["cut"]["enabled"] and arguments.Config["bab"]["cut"]["cplex_cuts"] and arguments.Config["bab"]["cut"]["cplex_cuts_revpickup"]:
    #         printed_d = d.get_min_domain(20, rev_order=True)
    #     else:
    #         printed_d = d.get_min_domain(20)
    #     for i in printed_d:
    #         if get_upper_bound:
    #             print(f'[{(i.lower_bound - i.threshold).max():.5f}, {(i.upper_bound - i.threshold).min():5f}] ({i.depth})', end=', ')
    #         else:
    #             print(f'{(i.lower_bound - i.threshold).max():.5f} ({i.depth})', end=', ')
    #     print()
    #     if hasattr(d, 'sublist'):
    #         print(f'Max depth domain: [{d.sublist[0].domain.lower_bound}, {d.sublist[0].domain.upper_bound}] ({d.sublist[0].domain.depth})')
    add_time = time.time() - add_time
    total_add_time += add_time

    total_time = time.time() - total_time
    print(f'Total time: {total_time:.4f}\t pickout: {pickout_time:.4f}\t decision: {decision_time:.4f}\t get_bound: {solve_time:.4f}\t add_domain: {add_time:.4f}')
    print(f'Accumulated time:\t pickout: {total_pickout_time:.4f}\t decision: {total_decision_time:.4f}\t get_bound: {total_solve_time:.4f}\t add_domain: {total_add_time:.4f}')

    # if len(d) > 0:
    #     if arguments.Config["bab"]["cut"]["enabled"] and arguments.Config["bab"]["cut"]["cplex_cuts"] and arguments.Config["bab"]["cut"]["cplex_cuts_revpickup"]:
    #         worst_domain = d.get_min_domain(1 ,rev_order=True)
    #         global_lb = worst_domain[-1].lower_bound - worst_domain[-1].threshold
    #     else:
    #         worst_domain = d.get_min_domain(1 ,rev_order=False)
    #         global_lb = worst_domain[0].lower_bound - worst_domain[0].threshold
    # else:
    #     print("No domains left, verification finished!")
    #     print('{} domains visited'.format(Visited))
    #     return torch.tensor(arguments.Config["bab"]["decision_thresh"] + 1e-7), np.inf

    # batch_ub = np.inf
    # if get_upper_bound:
    #     batch_ub = min(dom_ub)
    #     print(f"Current (lb-rhs): {global_lb.max()}, ub:{batch_ub}")
    # else:
    #     print(f"Current (lb-rhs): {global_lb.max()}")

    print('{} domains visited'.format(Visited))
    return total_cov_quota, split_all


def cut_verification(d, net, pre_relu_indices, fix_intermediate_layer_bounds=True):
    decision_thresh = arguments.Config["bab"]["decision_thresh"]
    get_upper_bound = arguments.Config["bab"]["get_upper_bound"]
    lp_cut_enabled = arguments.Config["bab"]["cut"]["lp_cut"]
    cplex_cuts = arguments.Config["bab"]["cut"]["cplex_cuts"]

    # construct the cut splits
    # change to only create one domain and make sure the other is infeasible
    split = {}
    if cplex_cuts:
        generate_cplex_cuts(net)

    if net.cutter.cuts is not None:
        split["cut"] = net.cutter.cuts
        split["cut_timestamp"] = net.cutter.cut_timestamp
    else:
        print('Cut is not present from cplex or predefined cut yet, direct return from cut init')
        return None, None
    return None, None

# NOTE
def initial_check_preimage_approx(A_dict, thre, c, samples=None):
    """check whether optimization on initial domain is successful"""
    # lbs: b, n_bounds (already multiplied with c in compute_bounds())
    assert (arguments.Config["preimage"]["under_approx"] or arguments.Config["preimage"]["over_approx"])
    debug = arguments.Config["debug"]["asserts"]
    if arguments.Config["preimage"]["under_approx"]:
        target_vol, cov_quota, preimage_dict = calc_initial_coverage(A_dict, c, samples, True, debug)
        if cov_quota >= thre:  # check whether the preimage approx satisfies the criteria
            print("Reached by optmization on the initial domain!")
            return True, cov_quota, target_vol, preimage_dict
        else:
            return False, cov_quota, target_vol, preimage_dict
    else:
        target_vol, cov_quota, preimage_dict = calc_initial_coverage(A_dict, c, samples, False, debug)
        if cov_quota <= thre:  # check whether the preimage approx satisfies the criteria
            print("Reached by optmization on the initial domain!")
            return True, cov_quota, target_vol, preimage_dict
        else:
            return False, cov_quota, target_vol, preimage_dict

def relu_bab_parallel(net, domain, x, use_neuron_set_strategy=False, refined_lower_bounds=None,
                      refined_upper_bounds=None, activation_opt_params=None,
                      reference_slopes=None, reference_lA=None, attack_images=None,
                      timeout=None, refined_betas=None, rhs=0):
    # the crown_lower/upper_bounds are present for initializing the unstable indx when constructing bounded module
    # it is ok to not pass them here, but then we need to go through a CROWN process again which is slightly slower
    start = time.time()
    # All supported arguments.
    global Visited, Flag_first_split, all_node_split 
    global total_pickout_time, total_decision_time, total_solve_time, total_add_time

    total_pickout_time = total_decision_time = total_solve_time = total_add_time = 0.0
    # NOTE add arguments required for preimage generation
    cov_thre = arguments.Config["preimage"]["threshold"]
    branch_budget = arguments.Config['preimage']['branch_budget']
    # result_dir = arguments.Config['preimage']['result_dir']
    bound_lower = arguments.Config["preimage"]["under_approx"]
    bound_upper = arguments.Config["preimage"]["over_approx"] 
    # model_tp = arguments.Config["model"]["name"] 
    # input_split_enabled = arguments.Config["bab"]["branching"]["input_split"]["enable"]
    # if input_split_enabled:
    #     opt_input_poly = True
    #     opt_relu_poly = False
    # else:
    #     opt_input_poly = False
    #     opt_relu_poly = True   

    timeout = timeout or arguments.Config["bab"]["timeout"]
    batch = arguments.Config["solver"]["batch_size"]
    opt_intermediate_beta = False
    use_bab_attack = arguments.Config["bab"]["attack"]["enabled"]
    max_dive_fix_ratio = arguments.Config["bab"]["attack"]["max_dive_fix_ratio"]
    min_local_free_ratio = arguments.Config["bab"]["attack"]["min_local_free_ratio"]
    cut_enabled = arguments.Config["bab"]["cut"]["enabled"]
    lp_cut_enabled = arguments.Config["bab"]["cut"]["lp_cut"]
    use_batched_domain = arguments.Config["bab"]["batched_domain_list"]
    

    if not isinstance(rhs, torch.Tensor):
        rhs = torch.tensor(rhs)
    decision_thresh = rhs

    # general (multi-bounds) output for one C matrix
    # any spec >= rhs, then this sample can be stopped; if all samples can be stopped, stop = True, o.w., False
    stop_criterion = stop_criterion_batch_any
    multi_spec_keep_func = lambda x: torch.all(x, dim=-1)

    Visited, Flag_first_split, global_ub = 0, True, np.inf
    betas = None
    Flag_covered = False

    sample_num = arguments.Config["preimage"]["sample_num"]
    sample_instability = arguments.Config["preimage"]["instability"]
    debug = arguments.Config["debug"]["asserts"]
    samples = calc_samples((x.ptb.x_L, x.ptb.x_U), net.model_ori, sample_num, debug=debug)
    tot_ambi_nodes_sample = 0
    for relu, uns in zip(net.net.relus, samples.unstable()):
        print(f'layer {relu.name} size {tuple(uns.shape)} unstable {uns.count_nonzero()}')
        tot_ambi_nodes_sample += uns.count_nonzero()
    print(f'-----------------\n# of unstable neurons (Sample): {tot_ambi_nodes_sample}\n-----------------\n')
      
    if arguments.Config["solver"]["alpha-crown"]["no_joint_opt"]:
        global_ub, global_lb, _, _, primals, updated_mask, lA, lower_bounds, upper_bounds, pre_relu_indices, slope, history, betas = net.build_the_model_with_refined_bounds(
            domain, x, None, None, stop_criterion_func=stop_criterion(decision_thresh), reference_slopes=None,
            cutter=net.cutter)
    elif refined_lower_bounds is None or refined_upper_bounds is None:
        assert arguments.Config["general"]["enable_incomplete_verification"] is False
        global_ub, global_lb, _, _, primals, updated_mask, lA, A, lower_bounds, upper_bounds, pre_relu_indices, slope, history, attack_image = net.build_the_model(
            domain, x, stop_criterion_func=stop_criterion(decision_thresh),opt_input_poly=False,opt_relu_poly=True,samples=[samples])
    else:
        global_ub, global_lb, _, _, primals, updated_mask, lA, lower_bounds, upper_bounds, pre_relu_indices, slope, history, betas = net.build_the_model_with_refined_bounds(
            domain, x, refined_lower_bounds, refined_upper_bounds, activation_opt_params, reference_lA=reference_lA,
            stop_criterion_func=stop_criterion(decision_thresh), reference_slopes=reference_slopes,
            cutter=net.cutter, refined_betas=refined_betas)
        # release some storage to save memory
        if activation_opt_params is not None: del activation_opt_params
        torch.cuda.empty_cache()
    tot_ambi_nodes = 0
    # only pick the first copy from possible multiple x
    updated_mask = [mask[0:1] for mask in updated_mask]
    # mask_sample = [mask[0:1] for mask in mask_sample]
    for i, layer_mask in enumerate(updated_mask):
        n_unstable = int(torch.sum(layer_mask).cpu().item())
        print(f'layer {i} size {tuple(layer_mask.shape[1:])} unstable {n_unstable}')
        tot_ambi_nodes += n_unstable

    print(f'-----------------\n# of unstable neurons (Interval): {tot_ambi_nodes}\n-----------------\n')
            
    if sample_instability:
        # Reduce the search space by using samples to determine (almost) stable activations
        eps = torch.finfo(samples.X.dtype).eps * 4
        for act, lb, ub in zip(samples.activations, lower_bounds, upper_bounds):
            lb = torch.where((act.min(0)[0] < 0.0) | (lb >= 0.0), lb, -eps)
            ub = torch.where((act.max(0)[0] > 0.0) | (ub <= 0.0), ub, eps)
    # NOTE check the first coarsest preimage without any splitting or optimization
    initial_covered, cov_quota, target_vol, preimage_dict = initial_check_preimage_approx(A, cov_thre, net.c, samples)
    print('Preimage volume:', target_vol)
    if debug:
        if bound_lower:
            assert cov_quota <= 1.0
        elif bound_upper:
            assert cov_quota >= 1.0
    # NOTE second variable is intended for extra constraints
    times = [time.time() - start]
    coverages = [cov_quota]
    num_domains = [1]
    if initial_covered:
        path = save_premap((target_vol, cov_quota, preimage_dict), dict(arguments.Config), arguments.Config["preimage"]["result_dir"], time.time() - start, True, times, coverages, num_domains)
        return (
            initial_covered,
            preimage_dict,
            Visited,
            time.time() - start,
            [cov_quota],
            1,
            path
        )
    if target_vol == 0:
        path = save_premap((target_vol, cov_quota, preimage_dict), dict(arguments.Config), arguments.Config["preimage"]["result_dir"], time.time() - start, False, times, coverages, num_domains)
        return (
            initial_covered,
            preimage_dict,
            Visited,
            time.time() - start,
            [1],
            1,
            path
        )
    # if arguments.Config["preimage"]["save_process"]:
    #     save_path = os.path.join(arguments.Config["preimage"]["result_dir"], 'run_example')
    #     save_file = os.path.join(save_path,'{}_spec_{}_init'.format(arguments.Config["data"]["dataset"], arguments.Config["preimage"]["runner_up"]))
    #     with open(save_file, 'wb') as f:
    #         pickle.dump(preimage_dict, f)
    if arguments.Config["solver"]["beta-crown"]["all_node_split_LP"]:
        timeout = arguments.Config["bab"]["timeout"]
        # mip_multi_proc = arguments.Config["solver"]["mip"]["parallel_solvers"]
        # mip_threads = arguments.Config["solver"]["mip"]["solver_threads"]
        # solver_pkg = arguments.Config["solver"]["intermediate_refinement"]["solver_pkg"]
        # adv_warmup = arguments.Config["solver"]["mip"]["adv_warmup"]
        net.build_solver_model(timeout, model_type="lp")

    if use_bab_attack:
        # Beam search based BaB enabled. We need to construct the MIP model.
        print('Building MIP for beam search...')
        _ = net.build_solver_model(
                    timeout=arguments.Config["bab"]["attack"]["mip_timeout"],
                    mip_multi_proc=arguments.Config["solver"]["mip"]["parallel_solvers"],
                    mip_threads=arguments.Config["solver"]["mip"]["solver_threads"],
                    model_type="mip")

    all_label_global_lb = global_lb
    all_label_global_lb = torch.min(all_label_global_lb - decision_thresh).item()
    all_label_global_ub = global_ub
    all_label_global_ub = torch.max(all_label_global_ub - decision_thresh).item()

    # if lp_test in ["LP", "MIP"]:
    #     return all_label_global_lb, all_label_global_ub, [[time.time()-start, global_lb]], 0, 'unknown'

    # if stop_criterion(decision_thresh)(global_lb).all():
    #     return all_label_global_lb, all_label_global_ub, [[time.time()-start, global_lb]], 0, 'safe'

    if not opt_intermediate_beta:
        # If we are not optimizing intermediate layer bounds, we do not need to save all the intermediate alpha.
        # We only keep the alpha for the last layer.
        if not arguments.Config['solver']['beta-crown'].get('enable_opt_interm_bounds', False):
            # new_slope shape: [dict[relu_layer_name, {final_layer: torch.tensor storing alpha}] for each sample in batch]
            new_slope = {}
            kept_layer_names = [net.net.final_name]
            kept_layer_names.extend(filter(lambda x: len(x.strip()) > 0, arguments.Config["bab"]["optimized_intermediate_layers"].split(",")))
            print(f'Keeping slopes for these layers: {kept_layer_names}')
            for relu_layer, alphas in slope.items():
                new_slope[relu_layer] = {}
                for layer_name in kept_layer_names:
                    if layer_name in alphas:
                        new_slope[relu_layer][layer_name] = alphas[layer_name]
                    else:
                        print(f'Layer {relu_layer} missing slope for start node {layer_name}')
        else:
            new_slope = slope
    else:
        new_slope = slope



    # net.tot_ambi_nodes = tot_ambi_nodes

    if use_batched_domain:
        assert not use_bab_attack, "Please disable batched_domain_list to run BaB-Attack."
        DomainClass = BatchedReLUDomainList
        raise NotImplementedError()
    else:
        DomainClass = SortedReLUDomainList

    # This is the first (initial) domain.
    calc_constraints(net, [samples], None, lower_bounds, upper_bounds, debug=debug)
    num_initial_domains = 1
    if bound_lower:
        domains = DomainClass([(target_vol, cov_quota, 1.0)], [preimage_dict['lA']], [preimage_dict['lbias']],
                            global_lb, global_ub, lower_bounds, upper_bounds, new_slope,
                            copy.deepcopy(history), [0] * num_initial_domains, net.c, # "[0] * num_initial_domains" corresponds to initial domain depth
                            decision_thresh,
                            betas, num_initial_domains,
                            interm_transfer=arguments.Config["bab"]["interm_transfer"],
                            samples=samples)
    elif bound_upper:
        domains = DomainClass([(target_vol, cov_quota, 1.0)], [preimage_dict['uA']], [preimage_dict['ubias']],
                            global_lb, global_ub, lower_bounds, upper_bounds, new_slope,
                            copy.deepcopy(history), [0] * num_initial_domains, net.c, # "[0] * num_initial_domains" corresponds to initial domain depth
                            decision_thresh,
                            betas, num_initial_domains,
                            interm_transfer=arguments.Config["bab"]["interm_transfer"],
                            samples=samples)
    del samples # Save (device) memory
    if use_bab_attack:
        # BaB-attack code still uses a legacy sorted domain list.
        domains = domains.to_sortedList()

    if not arguments.Config["bab"]["interm_transfer"]:
        # tell the AutoLiRPA class not to transfer intermediate bounds to save time
        net.interm_transfer = arguments.Config["bab"]["interm_transfer"]

    # after domains are added, we replace global_lb, global_ub with the multile targets "real" global lb and ub to make them scalars
    global_lb, global_ub = all_label_global_lb, all_label_global_ub

        
    if cut_enabled:
        print('======================Cut verification begins======================')
        start_cut = time.time()
        # enable lp solver
        if lp_cut_enabled:
            glb = net.build_the_model_lp()
        if arguments.Config["bab"]["cut"]["cplex_cuts"]:
            time.sleep(arguments.Config["bab"]["cut"]["cplex_cuts_wait"])
        global_lb_from_cut, batch_ub_from_cut = cut_verification(domains, net, pre_relu_indices, fix_intermediate_layer_bounds=not opt_intermediate_beta)
        if global_lb_from_cut is None and batch_ub_from_cut is None:
            # no available cut present --- we don't refresh global_lb and global_ub
            pass
        else:
            global_lb, batch_ub = global_lb_from_cut, batch_ub_from_cut
        print('Cut bounds before BaB:', float(global_lb))
        if len(domains) >= 1 and getattr(net.cutter, 'opt', False):
            # beta will be reused from split_history
            assert len(domains) == 1
            assert isinstance(domains[0].split_history['general_betas'], torch.Tensor)
            net.cutter.refine_cuts(split_history=domains[0].split_history)
        print('Cut time:', time.time() - start_cut)
        print('======================Cut verification ends======================')

    if arguments.Config["bab"]["attack"]["enabled"]:
        # Max number of fixed neurons during diving.
        max_dive_fix = int(max_dive_fix_ratio * tot_ambi_nodes)
        min_local_free = int(min_local_free_ratio * tot_ambi_nodes)
        adv_pool = AdvExamplePool(net.net, updated_mask, C=net.c)
        adv_pool.add_adv_images(attack_images)
        print(f'best adv in pool: {adv_pool.adv_pool[0].obj}, worst {adv_pool.adv_pool[-1].obj}')
        adv_pool.print_pool_status()
        find_promising_domains.counter = 0
        # find_promising_domains.current_method = "bottom-up"
        find_promising_domains.current_method = "top-down"
        find_promising_domains.topdown_status = "normal"
        find_promising_domains.bottomup_status = "normal"
        beam_mip_attack.started = False
        global_ub = min(all_label_global_ub, adv_pool.adv_pool[0].obj)

    glb_record = [[time.time()-start, global_lb]]
    iter_cov_quota = [cov_quota]
    # run_condition = len(domains) > 0
    num_iter = 0
    gc_time = time.time()
    num_unstable = sum(int(u.sum().detach().cpu().item()) for u in updated_mask)

    if bound_lower:
        while cov_quota < cov_thre and Visited < branch_budget and time.time() < start + timeout:
            global_lb = None

            if time.time() > gc_time + 30:
                gc_time = time.time()
                gc.collect()
                if torch.cuda.is_available() and torch.cuda.mem_get_info()[0] < 1e9:
                    torch.cuda.empty_cache()

            if use_bab_attack:
                max_dive_fix_ratio = arguments.Config["bab"]["attack"]["max_dive_fix_ratio"]
                min_local_free_ratio = arguments.Config["bab"]["attack"]["min_local_free_ratio"]
                max_dive_fix = int(max_dive_fix_ratio * tot_ambi_nodes)
                min_local_free = int(min_local_free_ratio * tot_ambi_nodes)
                global_lb, batch_ub, domains = bab_attack(
                        domains, net, batch, pre_relu_indices, 0,
                        fix_intermediate_layer_bounds=True,
                        adv_pool=adv_pool,
                        max_dive_fix=max_dive_fix, min_local_free=min_local_free)

            # if global_lb is None:
            # cut is enabled
            if arguments.Config["bab"]["cut"]["enabled"] and arguments.Config["bab"]["cut"]["cplex_cuts"]:
                fetch_cut_from_cplex(net)
            # Do two batch of neuron set bounds per 10000 domains
            if len(domains) > 80000 and len(domains) % 10000 < batch * 2 and use_neuron_set_strategy:
                # neuron set  bounds cost more memory, we set a smaller batch here
                cov_quota, all_node_split = batch_verification(num_unstable, domains, net, int(batch/2), pre_relu_indices, 0,
                                        fix_intermediate_layer_bounds=False, stop_func=stop_criterion,
                                        multi_spec_keep_func=multi_spec_keep_func, bound_lower=bound_lower, bound_upper=bound_upper)
            else:
                cov_quota, all_node_split = batch_verification(num_unstable, domains, net, batch, pre_relu_indices, 0,
                                        fix_intermediate_layer_bounds=not opt_intermediate_beta,
                                        stop_func=stop_criterion, multi_spec_keep_func=multi_spec_keep_func, bound_lower=bound_lower, bound_upper=bound_upper)


            times.append(time.time() - start)
            coverages.append(cov_quota)
            num_domains.append(len(domains))
            print(f'--- Iteration {num_iter+1:2d}, Coverage quota {cov_quota:8.6f}, Time {time.time() - start:.1f}s ---')
            iter_cov_quota.append(cov_quota)
            if debug:
                assert cov_quota <= 1.0
            if arguments.Config["preimage"]["save_process"]:
                preimage_dict_all = get_preimage_info(domains)
                # history_list = []
                # for idx, dm in enumerate(domains):
                #     history_list.append(dm.history)
                # split_plane_list = get_extra_const(net, history_list)
                save_path = os.path.join(arguments.Config["preimage"]["result_dir"], 'run_example')
                save_file = os.path.join(save_path,'{}_spec_{}_iter_{}'.format(arguments.Config["data"]["dataset"], arguments.Config["preimage"]["runner_up"], num_iter))
                with open(save_file, 'wb') as f:
                    pickle.dump(preimage_dict_all, f) 
                # split_plane_file = os.path.join(save_path, '{}_split_iter_{}'.format(arguments.Config["data"]["dataset"], num_iter))
                # with open(split_plane_file, 'wb') as f:
                #     pickle.dump(split_plane_list, f)                    


            num_iter += 1
            if all_node_split:
                break
    elif bound_upper:
        while cov_quota > cov_thre and Visited < branch_budget and time.time() < start + timeout:
            global_lb = None

            if time.time() > gc_time + 30:
                gc_time = time.time()
                gc.collect()
                if torch.cuda.is_available() and torch.cuda.mem_get_info()[0] < 1e9:
                    torch.cuda.empty_cache()

            if use_bab_attack:
                max_dive_fix_ratio = arguments.Config["bab"]["attack"]["max_dive_fix_ratio"]
                min_local_free_ratio = arguments.Config["bab"]["attack"]["min_local_free_ratio"]
                max_dive_fix = int(max_dive_fix_ratio * tot_ambi_nodes)
                min_local_free = int(min_local_free_ratio * tot_ambi_nodes)
                global_lb, batch_ub, domains = bab_attack(
                        domains, net, batch, pre_relu_indices, 0,
                        fix_intermediate_layer_bounds=True,
                        adv_pool=adv_pool,
                        max_dive_fix=max_dive_fix, min_local_free=min_local_free)

            # if global_lb is None:
            # cut is enabled
            if arguments.Config["bab"]["cut"]["enabled"] and arguments.Config["bab"]["cut"]["cplex_cuts"]:
                fetch_cut_from_cplex(net)
            # Do two batch of neuron set bounds per 10000 domains
            if len(domains) > 80000 and len(domains) % 10000 < batch * 2 and use_neuron_set_strategy:
                # neuron set  bounds cost more memory, we set a smaller batch here
                cov_quota, all_node_split = batch_verification(num_unstable, domains, net, int(batch/2), pre_relu_indices, 0,
                                        fix_intermediate_layer_bounds=False, stop_func=stop_criterion,
                                        multi_spec_keep_func=multi_spec_keep_func, bound_lower=bound_lower, bound_upper=bound_upper)
            else:
                cov_quota, all_node_split = batch_verification(num_unstable,  domains, net, batch, pre_relu_indices, 0,
                                        fix_intermediate_layer_bounds=not opt_intermediate_beta,
                                        stop_func=stop_criterion, multi_spec_keep_func=multi_spec_keep_func, bound_lower=bound_lower, bound_upper=bound_upper)


            times.append(time.time() - start)
            coverages.append(cov_quota)
            num_domains.append(len(domains))
            print(f'--- Iteration {num_iter+1:2d}, Coverage quota {cov_quota:8.6f}, Time {time.time() - start:.1f}s ---')
            iter_cov_quota.append(cov_quota)
            if debug:
                assert cov_quota >= 1.0
            if arguments.Config["preimage"]["save_process"]:
                preimage_dict_all = get_preimage_info(domains)
                # history_list = []
                # for idx, dm in enumerate(domains):
                #     history_list.append(dm.history)
                # split_plane_list = get_extra_const(net, history_list)
                save_path = os.path.join(arguments.Config["preimage"]["result_dir"], 'run_example')
                if bound_lower:
                    save_file = os.path.join(save_path,'{}_spec_{}_iter_{}'.format(arguments.Config["data"]["dataset"], arguments.Config["preimage"]["runner_up"], num_iter))
                if bound_upper:
                    save_file = os.path.join(save_path,'{}_spec_{}_iter_{}_relu_over_dual_False_0_4'.format(arguments.Config["data"]["dataset"], arguments.Config["preimage"]["runner_up"], num_iter))
                with open(save_file, 'wb') as f:
                    pickle.dump(preimage_dict_all, f) 
                # split_plane_file = os.path.join(save_path, '{}_split_relu_over_1_6'.format(arguments.Config["data"]["dataset"]))
                # with open(split_plane_file, 'wb') as f:
                #     pickle.dump(split_plane_list, f)                    


            num_iter += 1    
            if all_node_split:
                break

    time_cost = time.time() - start
    subdomain_num = len(domains)
    success = ((cov_quota >= cov_thre) and bound_lower) or ((cov_quota <= cov_thre) and bound_upper)
    path = save_premap(domains, dict(arguments.Config), arguments.Config["preimage"]["result_dir"], time_cost, success, times, coverages, num_domains)
    preimage_dict_all = get_preimage_info(domains)
    del domains
    return success, preimage_dict_all, Visited, time_cost, iter_cov_quota, subdomain_num, path


def get_preimage_info(domains):
    preimage_dict_all = []
    for idx, dom in enumerate(domains):
        preimage_dict_all.append((dom.preimg_A, dom.preimg_b))
    return preimage_dict_all
