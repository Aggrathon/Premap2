#########################################################################
##                                                                     ##
##     This program is licenced under the BSD 3-Clause License,        ##
##        contained in the LICENCE file in this directory.             ##
##                                                                     ##
#########################################################################
"""Branch and bound for activation space split."""
import gc
import time
import numpy as np
import torch

from auto_LiRPA.utils import stop_criterion_sum, stop_criterion_batch_any

import arguments

try:
    from premap2.domains import DomainList
    from premap2.splitting import select_node_batch, split_node_batch, stabilize_on_samples
except ImportError:
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).parent.parent.parent / 'src'))
    from premap2.domains import DomainList
    from premap2.splitting import select_node_batch, split_node_batch, stabilize_on_samples

Visited, Flag_first_split = 0, True
all_node_split = False
total_pickout_time = total_decision_time = total_solve_time = total_add_time = 0.0


def batch_verification(num_unstable, domains: DomainList, net, batch, pre_relu_indices, growth_rate, fix_intermediate_layer_bounds=True,
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

    selected = domains.get_batch(batch)
    batch = len(selected)
    
    pickout_time = time.time() - pickout_time
    total_pickout_time += pickout_time

    decision_time = time.time()

    sample_num = arguments.Config["preimage"]["sample_num"]
    heuristics = arguments.Config["preimage"]["heuristics"]
    tighten = arguments.Config["preimage"]["tighten_bounds"]
    if arguments.Config["preimage"]["instability"]:
        # Disable the "stable" heuristic since it should be constant 0 after stabilize_on_samples
        heuristics = ([('stable', 0.0)] + heuristics if heuristics else [('stable', 0.0)])
        stabilize_on_samples(selected,domains, add_empty=bound_upper, debug=debug)
    if debug:
        assert all(len(s) > 0 for s in selected)

    if branching_method in ('preimg', 'premap'):
        branching_decision = select_node_batch(selected, domains.output, sample_num, heuristics, debug=debug)
    else:
        raise NotImplementedError(f'Unsupported branching method "{branching_method}" for PREMAP.')

    decision_time = time.time() - decision_time
    total_decision_time += decision_time
    solve_time = time.time()
    single_node_split = True


    # Split
    ret = split_node_batch(net, domains, bound_lower, selected, branching_decision, tighten=tighten, debug=debug)
    orig_lbs, orig_ubs, slopes, betas, intermediate_betas, selected, cs, rhs, history, selected, branching_decision, samples = ret

    if len(branching_decision) == 0:
        total_solve_time += time.time() - solve_time
        return True

    # if len(sample_left_idx) == 0 and len(sample_right_idx) == 0:
    #     flag_next_split 
    # Caution: we use "all" predicate to keep the domain when multiple specs are present: all lbs should be <= threshold, otherwise pruned
    # maybe other "keeping" criterion needs to be passed here
    split = {"decision": [[bd] for bd in branching_decision], "coeffs": [[1.0] * len(branching_decision)]}
    ret = net.get_lower_bound(orig_lbs, orig_ubs, split, slopes=slopes, history=history, samples=samples,
                                fix_intermediate_layer_bounds=fix_intermediate_layer_bounds, betas=betas,
                                single_node_split=single_node_split, intermediate_betas=intermediate_betas, cs=cs, decision_thresh=rhs, rhs=rhs,
                                stop_func=stop_func(torch.cat([rhs, rhs])), multi_spec_keep_func=multi_spec_keep_func, bound_lower=bound_lower, bound_upper=bound_upper)

    dom_ub, dom_lb, dom_ub_point, lAs, A, dom_lb_all, dom_ub_all, slopes, split_history, betas, intermediate_betas, primals, dom_cs = ret
    del ret, orig_lbs, orig_ubs, samples
    solve_time = time.time() - solve_time
    total_solve_time += solve_time
    add_time = time.time()
    batch = len(branching_decision)

    domains.add_batch(
        net,
        selected,
        A,
        dom_lb,
        dom_ub,
        dom_lb_all,
        dom_ub_all,
        slopes,
        betas,
        intermediate_betas,
        debug=debug
    )

    if bound_lower:
        print(f'Split lower bound (avg):{dom_lb.mean().cpu().item():.3f}')
    if bound_upper:
        print(f'Split upper bound (avg): {dom_ub.mean().cpu().item():.3f}')
    print('Split layer bound gap (avg):', [round((ub - lb).sum().cpu().item() / (ub > lb).count_nonzero().cpu().item(), 3) for lb, ub in zip(dom_lb_all, dom_ub_all)])
    if debug:
        assert all((ub >= lb).all().cpu().item() for lb, ub in zip(dom_lb_all, dom_ub_all))

    Visited += len(selected)
    add_time = time.time() - add_time
    total_add_time += add_time

    total_time = time.time() - total_time
    print(f'Total time: {total_time:.4f}\t pickout: {pickout_time:.4f}\t decision: {decision_time:.4f}\t get_bound: {solve_time:.4f}\t add_domain: {add_time:.4f}')
    print(f'Accumulated time:\t pickout: {total_pickout_time:.4f}\t decision: {total_decision_time:.4f}\t get_bound: {total_solve_time:.4f}\t add_domain: {total_add_time:.4f}')
    print('{} domains visited'.format(Visited))
    return False



def relu_bab_parallel(net, domain, x, refined_lower_bounds=None,
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
    device = arguments.Config["general"]["device"]

    if not arguments.Config["bab"]["interm_transfer"]:
        # tell the AutoLiRPA class not to transfer intermediate bounds to save time
        net.interm_transfer = arguments.Config["bab"]["interm_transfer"]

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
    debug = arguments.Config["debug"]["asserts"]
    log_prob = arguments.Config["preimage"]["log_prob"]
    tighten = arguments.Config["preimage"]["tighten_bounds"]
    confidence = arguments.Config["preimage"]["confidence"]

    # Fix dimension order (singular batch first)
    x.data = x.data[:1]
    x.ptb.x_L = x.ptb.x_L[:1]
    x.ptb.x_U = x.ptb.x_U[:1]
    domain = domain[:1]
    net.input_shape = domain.shape
    rhs = decision_thresh = rhs[None, :, 0]
    net.c = torch.moveaxis(net.c, 1, 0)
    
    # This is the first (initial) domain.
    domains = DomainList(device=device, threshold=rhs[0], output=net.c[0], under=bound_lower, model=net.model_ori, log_prob=log_prob, num_samples=sample_num, lower_in=x.ptb.x_L, upper_in=x.ptb.x_U, keep=batch)
    dom = domains.create_domain(thighten=tighten, confidence=confidence is not None, debug=debug)
    
    tot_ambi_nodes_sample = 0
    for relu, uns in zip(net.net.relus, dom.unstable()):
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
            domain, x, stop_criterion_func=stop_criterion(decision_thresh),opt_input_poly=False,opt_relu_poly=True,samples=[dom.get_sample(sample_num//2)])
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

    if debug:
        for act, lb, ub in zip(dom.activations, lower_bounds, upper_bounds):
            # Large GMMs are non-deterministic, so we need a suprisingly large epsilon
            eps = torch.finfo(act.dtype).eps**0.5 * 0.5
            assert (act[:, None] > lb[None, :] - eps).all().cpu().item()
            assert (act[:, None] < ub[None, :] + eps).all().cpu().item()

    if arguments.Config["solver"]["beta-crown"]["all_node_split_LP"]:
        raise NotImplementedError("LP solving not implemented for PREMAP")

    if use_bab_attack:
        raise NotImplementedError("BaB attack not implemented for PREMAP")

    all_label_global_lb = global_lb
    all_label_global_lb = torch.min(all_label_global_lb - decision_thresh).item()
    all_label_global_ub = global_ub
    all_label_global_ub = torch.max(all_label_global_ub - decision_thresh).item()

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

    assert not use_batched_domain, "Batched domain not implemented for PREMAP"
    assert bound_lower != bound_upper, "Simultaneous lower and upper bounding not implemented"

    domains.add_batch(net,
                      [dom],
                      A,
                      global_lb,
                      global_ub,
                      lower_bounds,
                      upper_bounds,
                      new_slope,
                      betas
                      )
    
    # NOTE check the first coarsest preimage without any splitting or optimization
    target_vol, approx_vol = dom.preimg_vol, dom.approx_vol
    print('Preimage volume:', target_vol)
    print('Approximation volume:', approx_vol)
    if debug:
        if bound_lower:
            assert approx_vol <= target_vol
        elif bound_upper:
            assert approx_vol >= target_vol
    times = [time.time() - start]
    cov_quota = approx_vol / target_vol if target_vol > 0.0 else 0.0
    print('Coverage quota:', cov_quota)
    coverages = [cov_quota]
    num_domains = [1]
    if target_vol == 0:
        print("No preimage found!")
        path = domains.save(dict(arguments.Config), arguments.Config["preimage"]["result_dir"], time.time() - start, True, times, coverages, num_domains, confidence=confidence, debug=debug)
        return (False, Visited, time.time() - start, [cov_quota], 1, path)
    elif bound_lower and cov_quota >= cov_thre:
        print("Reached by optmization on the initial domain!")
        path = domains.save(dict(arguments.Config), arguments.Config["preimage"]["result_dir"], time.time() - start, True, times, coverages, num_domains, confidence=confidence, debug=debug)
        return (True, Visited, time.time() - start, [cov_quota], 1, path)
    elif bound_upper and cov_quota <= cov_thre:
        print("Reached by optmization on the initial domain!")
        path = domains.save(dict(arguments.Config), arguments.Config["preimage"]["result_dir"], time.time() - start, True, times, coverages, num_domains, confidence=confidence, debug=debug)
        return (True, Visited, time.time() - start, [cov_quota], 1, path)
    del dom, A, global_ub, global_lb, lower_bounds, upper_bounds, new_slope, betas # Save (device) memory

    # after domains are added, we replace global_lb, global_ub with the multile targets "real" global lb and ub to make them scalars
    global_lb, global_ub = all_label_global_lb, all_label_global_ub

    if cut_enabled:
        raise NotImplementedError('Cutting not implemented for PREMAP')
    
    num_iter = 0
    gc_time = time.time()
    num_unstable = sum(int(u.sum().detach().cpu().item()) for u in updated_mask)

    while Visited < branch_budget and time.time() < start + timeout:
        if bound_lower and cov_quota >= cov_thre:
            break
        elif bound_upper and cov_quota <= cov_thre:
            break

        if time.time() > gc_time + 30:
            gc_time = time.time()
            gc.collect()
            if torch.cuda.is_available() and torch.cuda.mem_get_info()[0] < 1e9:
                torch.cuda.empty_cache()

        shortcut = batch_verification(num_unstable, domains, net, batch, pre_relu_indices, 0,
                                    fix_intermediate_layer_bounds=not opt_intermediate_beta,
                                    stop_func=stop_criterion, multi_spec_keep_func=multi_spec_keep_func, bound_lower=bound_lower, bound_upper=bound_upper)

        preimg_vol = sum(sd.preimg_vol for sd in domains)
        approx_vol = sum(sd.approx_vol for sd in domains)
        cov_quota = approx_vol / max(1e-8, preimg_vol)
        print('Length of domains:', len(domains))
        print('Preimage volume:', preimg_vol)
        print('Under' if bound_lower else 'Over', 'approx. volume:', approx_vol)
        print('Approximation ratio:', cov_quota)
        if not bound_lower:
            print('Inverse ratio:', preimg_vol / max(1e-8, approx_vol))
        if debug:
            assert preimg_vol <= 1.0 + 1e-6
            assert approx_vol <= 1.0 + 1e-6
            assert preimg_vol >= approx_vol - 1e-6 if bound_lower else approx_vol >= preimg_vol - 1e-6
        all_node_split = domains.finished()

        if shortcut and not all_node_split:
            continue
        times.append(time.time() - start)
        coverages.append(cov_quota)
        num_domains.append(len(domains))
        print(f'--- Iteration {num_iter+1:2d}, Coverage quota {cov_quota:8.6f}, Time {time.time() - start:.1f}s ---')
        if debug:
            assert cov_quota <= 1.0 if bound_lower else cov_quota >= 1.0

        num_iter += 1
        if all_node_split:
            break

    time_cost = time.time() - start
    subdomain_num = len(domains)
    success = ((cov_quota >= cov_thre) and bound_lower) or ((cov_quota <= cov_thre) and bound_upper)
    path = domains.save(dict(arguments.Config), arguments.Config["preimage"]["result_dir"], time_cost, success, times, coverages, num_domains, confidence=confidence, debug=debug)
    del domains
    return success, Visited, time_cost, coverages, subdomain_num, path
