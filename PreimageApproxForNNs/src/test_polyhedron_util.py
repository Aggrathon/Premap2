import re
import torch
from torch.distributions import Uniform
import pickle
import os
import numpy as np
from collections import defaultdict
# from test_parse_args import get_args
from preimage_model_utils import load_input_bounds_numpy, load_model_simple
from preimage_model_utils import load_input_bounds
# NOTE for adding the arguments module
import arguments
from utils import load_model, load_verification_dataset


def save_A_dict(A, A_path, A_dir='.'):
    A_file= os.path.join(A_dir, A_path)
    with open(A_file, 'wb') as f:
        pickle.dump(A, f)

def post_process_A(A_dict):
    linear_rep_dict = dict()
    for output_node, linear_rep in A_dict.items():
        for input_node, param_dict in linear_rep.items():
            for k, v in param_dict.items():
                if v is not None: 
                    if k == 'lA' or k == 'uA':
                        linear_rep_dict[k] = torch.squeeze(v,0)
                    elif k == 'lbias' or k == 'ubias':
                        # NOTE: we dont squeeze bias as it can help with the batch addition
                        linear_rep_dict[k] = v
    return linear_rep_dict

def post_process_greedy_A(A_dict):
    linear_rep_dict_multi = dict()
    for output_node, linear_rep in A_dict.items():
        for input_node, param_dict in linear_rep.items():
            for k, v in param_dict.items():
                if v is not None:
                    linear_rep_dict_multi[k] = v
    return linear_rep_dict_multi

# Calculate the subregion volume    
def calc_total_sub_vol(dm_l, dm_u):
    total_sub_vol = 1
    in_dim = dm_l.shape[-1]
    dm_shape = dm_l.shape
    # print("check dm_l, dm_u shape", dm_shape)
    assert len(dm_shape) == 2 or len(dm_shape) == 1
    if len(dm_shape) == 2:
        dm_diff = dm_u[0] - dm_l[0]
        if arguments.Config["data"]["dataset"] == "vcas":
            for i in range(in_dim):
                # if i != 2:
                total_sub_vol = total_sub_vol * dm_diff[i]
        else:            
            for i in range(in_dim):
                if dm_diff[i] > 1e-6:
                    total_sub_vol = total_sub_vol * dm_diff[i]
    elif len(dm_shape) == 1:
        dm_diff = dm_u - dm_l
        if arguments.Config["data"]["dataset"] == "vcas":
            for i in range(in_dim):
                # if i != 2:
                total_sub_vol = total_sub_vol * dm_diff[i]
        else:
            for i in range(in_dim):
                if dm_diff[i] > 1e-6:
                    total_sub_vol = total_sub_vol * dm_diff[i]
    return total_sub_vol

def calc_Hrep_coverage_multi_spec_pairwise_over(A_b_dict, dm_l_all, dm_u_all, spec_dim):
    torch.manual_seed(arguments.Config["general"]["seed"])
    dataset_tp = arguments.Config["data"]["dataset"]  
    sample_num = arguments.Config["preimage"]["sample_num"]
    label = arguments.Config["preimage"]["label"]
    pair_num = int(len(dm_l_all)/(spec_dim * 2))
    bisec_sample_num = int(sample_num / 2)
    if "Customized" in dataset_tp:
        model = load_model(weights_loaded=False)
    else:
        model = load_model()
    if dataset_tp == 'dubinsrejoin':
        label_T = arguments.Config["preimage"]["runner_up"] - 4
    else:
        label_T = None
    device = arguments.Config["general"]["device"]
    model = model.to(device)
    cov_input_idx_all = [[] for _ in range(pair_num)] 
    for i in range(pair_num):
        cov_num = 0
        total_num = 0
        total_target_vol = 0
        total_loss = 0
        for j in range(2):
            dm_l, dm_u = dm_l_all[2*i*spec_dim+j*spec_dim], dm_u_all[2*i*spec_dim+j*spec_dim]
        # dm_l_1, dm_u_1 = dm_l_all[2*i+1], dm_u_all[2*i+1]
        # if samples is None:
            # samples = np.random.uniform(low=dm_l,high=dm_u,size=(sub_sample_num, len(dm_l)))
        # else:
            dm_vol = calc_total_sub_vol(dm_l, dm_u)
            # total_vol += dm_vol
            samples = Uniform(dm_l, dm_u).sample([bisec_sample_num])
            # tmp_samples = np.random.uniform(low=dm_l,high=dm_u,size=(bisec_sample_num, len(dm_l)))
            # data = torch.tensor(tmp_samples, dtype=torch.get_default_dtype())
            samples = samples.to(device)
            if dataset_tp == 'dubinsrejoin':
                prediction = model(samples)
                pred_R = prediction[:,:4]
                pred_T = prediction[:, 4:]
                pred_label_R = pred_R.argmax(dim=1).cpu().detach().numpy() 
                pred_label_T = pred_T.argmax(dim=1).cpu().detach().numpy()
                samples_idxs_R = np.where(pred_label_R == label)[0]
                samples_idxs_T = np.where(pred_label_T == label_T)[0]
                samples_eval_idxs = np.intersect1d(samples_idxs_R, samples_idxs_T)
            else:
                predicted = model(samples).argmax(dim=1).cpu().detach().numpy()    
                samples_eval_idxs = np.where(predicted==label)[0]
            target_num = len(samples_eval_idxs)
            target_vol = dm_vol * target_num / bisec_sample_num
            # cov_input_idx_all[i].append(target_vol)
            total_target_vol += target_vol
            if target_num > 0:
                samples_eval = samples.cpu().detach().numpy()
                if spec_dim == 1:
                    mat = A_b_dict['uA'][2*i+j]
                    bias = A_b_dict['ubias'][2*i+j]
                else:
                    mat = A_b_dict['uA'][2*i*spec_dim+j*spec_dim : 2*i*spec_dim+(j+1)*spec_dim]
                    bias = A_b_dict['ubias'][2*i*spec_dim+j*spec_dim : 2*i*spec_dim+(j+1)*spec_dim]
                    if dataset_tp != 'cartpole':
                        mat = np.squeeze(mat, axis=1)   
                print('Pair {}, subsection {}, mat shape: {}, samples_eval_T shape: {}'.format(i, j, mat.shape, samples_eval.T.shape))
                result = np.matmul(mat, samples_eval.T)+bias
                sub_tight_loss = calc_over_tightness(result, preimg_idx=samples_eval_idxs)
                if spec_dim > 1:
                    idxs_True = None
                    for s_dim in range(spec_dim):
                        idxs_tmp = set(np.where(result[s_dim]>=0)[0])
                        if idxs_True is None:
                            idxs_True = idxs_tmp
                        else:
                            idxs_True = idxs_True.intersection(idxs_tmp)
                else:
                    idxs_True = np.where(result>=0)[0]
                cov_input_idx_all[i].append((target_vol, len(idxs_True)/target_num, sub_tight_loss))
                cov_num += len(idxs_True)
                total_num += target_num
                total_loss += sub_tight_loss
            else:
                cov_input_idx_all[i].append((0,0,2))
                print('Pair {}, subsection {}, No samples of NN on: dm_l {}, dm_u {}'.format(i, j, dm_l, dm_u))# In this case, the subdomain will not lead to the target label, no need for further branching. set the cov_quota as 1, uncov_vol will be 0
            # however, when evaluating the generall coverage volume, the coverage quota is not calculated as 1, instead making an impact by not adding to the total number
        if total_num > 0:
            cov_quota = cov_num / total_num
            print("Pair {}, Coverage quota {}/{}:  {:.3f}, S-loss {:.2f}".format(i, cov_num, total_num, cov_quota, total_loss))
        else:
            cov_quota = 0
            print("Pair {}, Coverage quota {}/{}:  {:.3f}, S-loss {:.2f}".format(i, 0, 0, cov_quota, total_loss))
        # total_target_vol = total_vol * total_num / sample_num
        cov_input_idx_all[i].append((total_target_vol,cov_quota,total_loss))
        # Therefore, for each idx i, it consists of the cov_ratio for each bisection and the overall of splitting wrt i-th input feat.
    return cov_input_idx_all        

# In the all potential feature split case, we need the cov_quota for each pairwise subdomain, not the overall for all domains
def calc_Hrep_coverage_multi_spec_pairwise_under(A_b_dict, dm_l_all, dm_u_all, spec_dim):
    torch.manual_seed(arguments.Config["general"]["seed"])
    dataset_tp = arguments.Config["data"]["dataset"]  
    sample_num = arguments.Config["preimage"]["sample_num"]
    label = arguments.Config["preimage"]["label"]
    pair_num = int(len(dm_l_all)/(spec_dim * 2))
    bisec_sample_num = int(sample_num / 2)
    if "Customized" in dataset_tp:
        model = load_model(weights_loaded=False)
    else:
        model = load_model()
    if dataset_tp == 'dubinsrejoin':
        label_T = arguments.Config["preimage"]["runner_up"] - 4
    else:
        label_T = None
    device = arguments.Config["general"]["device"]
    model = model.to(device)
    cov_input_idx_all = [[] for _ in range(pair_num)] 
    for i in range(pair_num):
        cov_num = 0
        total_num = 0
        total_target_vol = 0
        total_reward = 0
        for j in range(2):
            dm_l, dm_u = dm_l_all[2*i*spec_dim+j*spec_dim], dm_u_all[2*i*spec_dim+j*spec_dim]
        # dm_l_1, dm_u_1 = dm_l_all[2*i+1], dm_u_all[2*i+1]
        # if samples is None:
            # samples = np.random.uniform(low=dm_l,high=dm_u,size=(sub_sample_num, len(dm_l)))
        # else:
            dm_vol = calc_total_sub_vol(dm_l, dm_u)
            # total_vol += dm_vol
            samples = Uniform(dm_l, dm_u).sample([bisec_sample_num])
            # tmp_samples = np.random.uniform(low=dm_l,high=dm_u,size=(bisec_sample_num, len(dm_l)))
            # data = torch.tensor(tmp_samples, dtype=torch.get_default_dtype())
            samples = samples.to(device)
            if dataset_tp == 'dubinsrejoin':
                prediction = model(samples)
                pred_R = prediction[:,:4]
                pred_T = prediction[:, 4:]
                pred_label_R = pred_R.argmax(dim=1).cpu().detach().numpy() 
                pred_label_T = pred_T.argmax(dim=1).cpu().detach().numpy()
                samples_idxs_R = np.where(pred_label_R == label)[0]
                samples_idxs_T = np.where(pred_label_T == label_T)[0]
                samples_eval_idxs = np.intersect1d(samples_idxs_R, samples_idxs_T)
            else:
                predicted = model(samples).argmax(dim=1).cpu().detach().numpy()    
                samples_eval_idxs = np.where(predicted==label)[0]
            target_num = len(samples_eval_idxs)
            target_vol = dm_vol * target_num / bisec_sample_num
            # cov_input_idx_all[i].append(target_vol)
            total_target_vol += target_vol
            if target_num > 0:
                samples_eval = samples[samples_eval_idxs]
                samples_eval = samples_eval.cpu().detach().numpy()
                if spec_dim == 1:
                    mat = A_b_dict['lA'][2*i+j]
                    bias = A_b_dict['lbias'][2*i+j]
                else:
                    mat = A_b_dict['lA'][2*i*spec_dim+j*spec_dim : 2*i*spec_dim+(j+1)*spec_dim]
                    bias = A_b_dict['lbias'][2*i*spec_dim+j*spec_dim : 2*i*spec_dim+(j+1)*spec_dim]
                    if dataset_tp != 'cartpole':
                        mat = np.squeeze(mat, axis=1)   
                print('Pair {}, subsection {}, mat shape: {}, samples_eval_T shape: {}'.format(i, j, mat.shape, samples_eval.T.shape))
                result = np.matmul(mat, samples_eval.T)+bias
                sub_tight_reward = calc_over_tightness(result, preimg_idx=None)
                if spec_dim > 1:
                    idxs_True = None
                    for s_dim in range(spec_dim):
                        idxs_tmp = set(np.where(result[s_dim]>=0)[0])
                        if idxs_True is None:
                            idxs_True = idxs_tmp
                        else:
                            idxs_True = idxs_True.intersection(idxs_tmp)
                else:
                    idxs_True = np.where(result>=0)[0]
                cov_input_idx_all[i].append((target_vol, len(idxs_True)/target_num, sub_tight_reward))
                cov_num += len(idxs_True)
                total_num += target_num
                total_reward += sub_tight_reward
            else:
                cov_input_idx_all[i].append((0,0,-2))
                print('Pair {}, subsection {}, No samples of NN on: dm_l {}, dm_u {}'.format(i, j, dm_l, dm_u))# In this case, the subdomain will not lead to the target label, no need for further branching. set the cov_quota as 1, uncov_vol will be 0
            # however, when evaluating the generall coverage volume, the coverage quota is not calculated as 1, instead making an impact by not adding to the total number
        if total_num > 0:
            cov_quota = cov_num / total_num
            print("Pair {}, Coverage quota {}/{}:  {:.3f}, S-reward {:.2f}".format(i, cov_num, total_num, cov_quota, total_reward))
        else:
            cov_quota = 0
            print("Pair {}, Coverage quota {}/{}:  {:.3f}, S-reward {:.2f}".format(i, 0, 0, cov_quota, total_reward))
        # total_target_vol = total_vol * total_num / sample_num
        cov_input_idx_all[i].append((total_target_vol, cov_quota, total_reward))
        # Therefore, for each idx i, it consists of the cov_ratio for each bisection and the overall of splitting wrt i-th input feat.
    return cov_input_idx_all    

def calc_mc_coverage_multi_spec_pairwise_under(A_b_dict, dm_l_all, dm_u_all, spec_dim):
    torch.manual_seed(arguments.Config["general"]["seed"])
    dataset_tp = arguments.Config["data"]["dataset"]  
    sample_num = arguments.Config["preimage"]["sample_num"]
    label = arguments.Config["preimage"]["label"]
    pair_num = int(len(dm_l_all)/(spec_dim * 2))
    bisec_sample_num = int(sample_num / 2)
    if "Customized" in dataset_tp:
        model = load_model(weights_loaded=False)
    else:
        model = load_model()
    if dataset_tp == 'dubinsrejoin':
        label_T = arguments.Config["preimage"]["runner_up"] - 4
    else:
        label_T = None
    device = arguments.Config["general"]["device"]
    model = model.to(device)
    cov_input_idx_all = [[] for _ in range(pair_num)] 
    for i in range(pair_num):
        cov_num = 0
        total_num = 0
        total_target_vol = 0
        total_reward = 0
        for j in range(2):
            dm_l, dm_u = dm_l_all[2*i*spec_dim+j*spec_dim], dm_u_all[2*i*spec_dim+j*spec_dim]
        # dm_l_1, dm_u_1 = dm_l_all[2*i+1], dm_u_all[2*i+1]
        # if samples is None:
            # samples = np.random.uniform(low=dm_l,high=dm_u,size=(sub_sample_num, len(dm_l)))
        # else:
            # dm_vol = calc_total_sub_vol(dm_l, dm_u)
            # total_vol += dm_vol
            dm_vol = np.prod(dm_u.cpu().detach().numpy() - dm_l.cpu().detach().numpy())
            samples = Uniform(dm_l, dm_u).sample([bisec_sample_num])
            # tmp_samples = np.random.uniform(low=dm_l,high=dm_u,size=(bisec_sample_num, len(dm_l)))
            # data = torch.tensor(tmp_samples, dtype=torch.get_default_dtype())
            samples = samples.to(device)
            if dataset_tp == 'dubinsrejoin':
                prediction = model(samples)
                pred_R = prediction[:,:4]
                pred_T = prediction[:, 4:]
                pred_label_R = pred_R.argmax(dim=1).cpu().detach().numpy() 
                pred_label_T = pred_T.argmax(dim=1).cpu().detach().numpy()
                samples_idxs_R = np.where(pred_label_R == label)[0]
                samples_idxs_T = np.where(pred_label_T == label_T)[0]
                samples_eval_idxs = np.intersect1d(samples_idxs_R, samples_idxs_T)
            else:
                predicted = model(samples).argmax(dim=1).cpu().detach().numpy()    
                samples_eval_idxs = np.where(predicted==label)[0]
            target_num = len(samples_eval_idxs)
            target_vol = dm_vol * target_num / bisec_sample_num
            # cov_input_idx_all[i].append(target_vol)
            total_target_vol += target_vol
            if target_num > 0:
                samples_eval = samples[samples_eval_idxs]
                samples_eval = samples_eval.cpu().detach().numpy()
                if spec_dim == 1:
                    mat = A_b_dict['lA'][2*i+j]
                    bias = A_b_dict['lbias'][2*i+j]
                else:
                    mat = A_b_dict['lA'][2*i*spec_dim+j*spec_dim : 2*i*spec_dim+(j+1)*spec_dim]
                    bias = A_b_dict['lbias'][2*i*spec_dim+j*spec_dim : 2*i*spec_dim+(j+1)*spec_dim]
                    if dataset_tp != 'cartpole':
                        mat = np.squeeze(mat, axis=1)   
                print('Pair {}, subsection {}, mat shape: {}, samples_eval_T shape: {}'.format(i, j, mat.shape, samples_eval.T.shape))
                result = np.matmul(mat, samples_eval.T)+bias
                sub_tight_reward = calc_over_tightness(result, preimg_idx=None)
                if spec_dim > 1:
                    idxs_True = None
                    for s_dim in range(spec_dim):
                        idxs_tmp = set(np.where(result[s_dim]>=0)[0])
                        if idxs_True is None:
                            idxs_True = idxs_tmp
                        else:
                            idxs_True = idxs_True.intersection(idxs_tmp)
                else:
                    idxs_True = np.where(result>=0)[0]
                cov_input_idx_all[i].append((target_vol, len(idxs_True)/target_num, sub_tight_reward))
                cov_num += len(idxs_True)
                total_num += target_num
                total_reward += sub_tight_reward
            else:
                cov_input_idx_all[i].append((0,0,-2))
                print('Pair {}, subsection {}, No samples of NN on: dm_l {}, dm_u {}'.format(i, j, dm_l, dm_u))# In this case, the subdomain will not lead to the target label, no need for further branching. set the cov_quota as 1, uncov_vol will be 0
            # however, when evaluating the generall coverage volume, the coverage quota is not calculated as 1, instead making an impact by not adding to the total number
        if total_num > 0:
            cov_quota = cov_num / total_num
            print("Pair {}, Coverage quota {}/{}:  {:.3f}, S-reward {:.2f}".format(i, cov_num, total_num, cov_quota, total_reward))
        else:
            cov_quota = 0
            print("Pair {}, Coverage quota {}/{}:  {:.3f}, S-reward {:.2f}".format(i, 0, 0, cov_quota, total_reward))
        # total_target_vol = total_vol * total_num / sample_num
        cov_input_idx_all[i].append((total_target_vol, cov_quota, total_reward))
        # Therefore, for each idx i, it consists of the cov_ratio for each bisection and the overall of splitting wrt i-th input feat.
    return cov_input_idx_all


def sigmoid(z):
    return 1 / (1 + np.exp(-z))   

def calc_over_tightness(sample_res, preimg_idx): #A, b, samples
    if preimg_idx is None:
        res_min_spec = np.min(sample_res.cpu().numpy(), axis=0)
    else:
        res_exact_preimg = sample_res[:, preimg_idx]
        res_min_spec = np.min(res_exact_preimg, axis=0)
    # res_min_spec = np.min(sample_res, axis=0)
    res_sigmoid = sigmoid(res_min_spec) 
    mean_res = np.mean(res_sigmoid)  
    return mean_res

def calc_input_coverage_initial_input_over(A_b_dict):
    torch.manual_seed(arguments.Config["general"]["seed"])
    dataset_tp = arguments.Config["data"]["dataset"]    
    sample_num = arguments.Config["preimage"]["sample_num"]
    label = arguments.Config["preimage"]["label"]
    # sample_dir = arguments.Config['preimage']["sample_dir"]
    # sample_path = os.path.join(sample_dir, 'sample_{}.pt'.format(dataset_tp))
    X, labels, data_max, data_min, perturb_epsilon = load_input_bounds(dataset_tp, label, quant=False, trans=False)
    # if not os.path.exists(sample_path):
    samples = Uniform(data_min, data_max).sample([sample_num])
    samples = torch.squeeze(samples, 1)
    # torch.save(samples, sample_path)
    # else:
    #     samples = torch.load(sample_path)
    if "Customized" in dataset_tp:
        model = load_model(weights_loaded=False)
    else:
        model = load_model()
    if dataset_tp == 'dubinsrejoin':
        label_T = arguments.Config["preimage"]["runner_up"]-4
    else:
        label_T = None
    dm_vol = calc_total_sub_vol(data_min, data_max)
    device = arguments.Config["general"]["device"]
    samples = samples.to(device)
    model = model.to(device)
    if dataset_tp == 'dubinsrejoin':
        prediction = model(samples)
        pred_R = prediction[:,:4]
        pred_T = prediction[:, 4:]
        pred_label_R = pred_R.argmax(dim=1).cpu().detach().numpy() 
        pred_label_T = pred_T.argmax(dim=1).cpu().detach().numpy()
        samples_idxs_R = np.where(pred_label_R == label)[0]
        samples_idxs_T = np.where(pred_label_T == label_T)[0]
        idxs = np.intersect1d(samples_idxs_R, samples_idxs_T)
    else:
        predicted = model(samples).argmax(dim=1).cpu().detach().numpy()
        # volume estimation for exact preimage 
        idxs = np.where(predicted==label)[0] 
    if len(idxs)>0:
        samples_eval = samples.cpu().detach().numpy()
        target_vol = dm_vol * len(idxs)/sample_num
        # print('Label: {}, Num: {}'.format(label, len(idxs)))   
        # samples_tmp = samples[idxs]
        mat = A_b_dict["uA"]
        bias = A_b_dict["ubias"]
        # print('mat shape: {}, sample_tmp_T shape: {}'.format(mat.shape, samples_tmp.T.shape))   
         
        if dataset_tp != 'cartpole':
            mat = np.squeeze(mat, axis=1)            
        result = np.matmul(mat, samples_eval.T)+bias
        spec_dim = result.shape[0]
        if spec_dim > 1:
            idxs_True = None
            for i in range(spec_dim):
                idxs_tmp = set(np.where(result[i]>=0)[0])
                if idxs_True is None:
                    idxs_True = idxs_tmp
                else:
                    idxs_True = idxs_True.intersection(idxs_tmp)
        else:
            idxs_True = np.where(result>=0)[0]
        cov_quota = len(idxs_True)/len(idxs)
        print("Coverage quota {}/{}:  {:.3f}".format(len(idxs_True), len(idxs), cov_quota))
    else:
        target_vol = 0
        cov_quota = 0
    return target_vol, cov_quota

def calc_input_coverage_initial_input_under(A_b_dict):
    torch.manual_seed(arguments.Config["general"]["seed"])
    dataset_tp = arguments.Config["data"]["dataset"]    
    sample_num = arguments.Config["preimage"]["sample_num"]
    label = arguments.Config["preimage"]["label"]
    # sample_dir = arguments.Config['preimage']["sample_dir"]
    # sample_path = os.path.join(sample_dir, 'sample_{}.pt'.format(dataset_tp))
    # if not os.path.exists(sample_path):
    
    X, labels, data_max, data_min, perturb_epsilon = load_input_bounds(dataset_tp, label, quant=False, trans=False)
    dm_vol = calc_total_sub_vol(data_min, data_max)
    # dm_vol = np.prod(data_max.cpu().detach().numpy() - data_min.cpu().detach().numpy())
    samples = Uniform(data_min, data_max).sample([sample_num])
    samples = torch.squeeze(samples, 1)
    # torch.save(samples, sample_path)
    # else:
    #     samples = torch.load(sample_path)
    if "Customized" in dataset_tp:
        model = load_model(weights_loaded=False)
    else:
        model = load_model()
    if dataset_tp == 'dubinsrejoin':
        label_T = arguments.Config["preimage"]["runner_up"]-4
    else:
        label_T = None
    device = arguments.Config["general"]["device"]
    samples = samples.to(device)
    model = model.to(device)
    if dataset_tp == 'dubinsrejoin':
        prediction = model(samples)
        pred_R = prediction[:,:4]
        pred_T = prediction[:, 4:]
        pred_label_R = pred_R.argmax(dim=1).cpu().detach().numpy() 
        pred_label_T = pred_T.argmax(dim=1).cpu().detach().numpy()
        samples_idxs_R = np.where(pred_label_R == label)[0]
        samples_idxs_T = np.where(pred_label_T == label_T)[0]
        idxs = np.intersect1d(samples_idxs_R, samples_idxs_T)
    else:
        predicted = model(samples).argmax(dim=1).cpu().detach().numpy()
        idxs = np.where(predicted==label)[0]
    if len(idxs)>0:
        target_vol = dm_vol * len(idxs)/sample_num
        print('Label: {}, Num: {}'.format(label, len(idxs)))
        # for i in range(output_num):    
        samples_tmp = samples[idxs]
        samples_tmp = samples_tmp.cpu().detach().numpy()
        mat = A_b_dict["lA"]
        bias = A_b_dict["lbias"]
        # print('mat shape: {}, sample_tmp_T shape: {}'.format(mat.shape, samples_tmp.T.shape))    
        if dataset_tp != 'cartpole':
            mat = np.squeeze(mat, axis=1)            
        result = np.matmul(mat, samples_tmp.T)+bias
        spec_dim = result.shape[0]
        if spec_dim > 1:
            idxs_True = None
            for i in range(spec_dim):
                idxs_tmp = set(np.where(result[i]>=0)[0])
                if idxs_True is None:
                    idxs_True = idxs_tmp
                else:
                    idxs_True = idxs_True.intersection(idxs_tmp)
        else:
            idxs_True = np.where(result>=0)[0]
        cov_quota = len(idxs_True)/len(idxs)
        print("Coverage quota {}/{}:  {:.3f}".format(len(idxs_True), len(idxs), cov_quota))
    else:
        target_vol = 0
        cov_quota = 0
    return target_vol, cov_quota
def is_inside_polytope(A, b, point):
    return np.all(np.dot(A, point) + b >= 0)
         
def calc_mc_esti_coverage_initial_input_under(A_b_dict):
    torch.manual_seed(arguments.Config["general"]["seed"])
    dataset_tp = arguments.Config["data"]["dataset"]    
    sample_num = arguments.Config["preimage"]["sample_num"]
    label = arguments.Config["preimage"]["label"]
    # sample_dir = arguments.Config['preimage']["sample_dir"]
    # sample_path = os.path.join(sample_dir, 'sample_{}.pt'.format(dataset_tp))
    # if not os.path.exists(sample_path):
    
    X, labels, data_max, data_min, perturb_epsilon = load_input_bounds(dataset_tp, label, quant=False, trans=False)
    # dm_vol = calc_total_sub_vol(data_min, data_max)
    dm_vol = np.prod(data_max.cpu().detach().numpy() - data_min.cpu().detach().numpy())
    samples = Uniform(data_min, data_max).sample([sample_num])
    samples = torch.squeeze(samples, 1)
    # torch.save(samples, sample_path)
    # else:
    #     samples = torch.load(sample_path)
    if "Customized" in dataset_tp:
        model = load_model(weights_loaded=False)
    else:
        model = load_model()
    if dataset_tp == 'dubinsrejoin':
        label_T = arguments.Config["preimage"]["runner_up"]-4
    else:
        label_T = None
    device = arguments.Config["general"]["device"]
    samples = samples.to(device)
    model = model.to(device)
    if dataset_tp == 'dubinsrejoin':
        prediction = model(samples)
        pred_R = prediction[:,:4]
        pred_T = prediction[:, 4:]
        pred_label_R = pred_R.argmax(dim=1).cpu().detach().numpy() 
        pred_label_T = pred_T.argmax(dim=1).cpu().detach().numpy()
        samples_idxs_R = np.where(pred_label_R == label)[0]
        samples_idxs_T = np.where(pred_label_T == label_T)[0]
        idxs = np.intersect1d(samples_idxs_R, samples_idxs_T)
    else:
        predicted = model(samples).argmax(dim=1).cpu().detach().numpy()
        idxs = np.where(predicted==label)[0]
        
    if len(idxs)>0:
        target_vol = dm_vol * len(idxs)/sample_num
        print('Label: {}, Num: {}'.format(label, len(idxs)))
        # for i in range(output_num):    
        samples_tmp = samples[idxs]
        samples_tmp = samples_tmp.cpu().detach().numpy()
        mat = A_b_dict["lA"]
        bias = A_b_dict["lbias"]
        # print('mat shape: {}, sample_tmp_T shape: {}'.format(mat.shape, samples_tmp.T.shape))    
        if dataset_tp != 'cartpole':
            mat = np.squeeze(mat, axis=1)          
        # count_inside = sum(is_inside_polytope(mat, bias, point) for point in samples_tmp)  
        result = np.matmul(mat, samples_tmp.T)+bias
        spec_dim = result.shape[0]
        if spec_dim > 1:
            idxs_True = None
            for i in range(spec_dim):
                idxs_tmp = set(np.where(result[i]>=0)[0])
                if idxs_True is None:
                    idxs_True = idxs_tmp
                else:
                    idxs_True = idxs_True.intersection(idxs_tmp)
        else:
            idxs_True = np.where(result>=0)[0]
        cov_quota = len(idxs_True) / len(idxs)
        print("Coverage quota {}/{}:  {:.3f}".format(len(idxs_True), len(idxs), cov_quota))
    else:
        target_vol = 0
        cov_quota = 0
    return target_vol, cov_quota


def calc_Hrep_coverage(A_b_dict, args):
    data_lb, data_ub, output_num = load_input_bounds_numpy(args.dataset, args.quant, args.trans)
    sample_num = args.sample_num
    samples = np.random.uniform(low=data_lb, high=data_ub, size=(sample_num, len(data_lb)))
    # print(samples)
    model_path = args.model
    ext = model_path.split('.')[-1]
    if ext == 'pt':
        model = load_model_simple(args.model_name, args.model, weights_loaded=True)
    elif ext == 'onnx':
        onnx_folder = "/home/xiyue/vcas-code/acas/networks/onnx"
        onnx_path = os.path.join(onnx_folder, model_path)
        # xy: test onnx_path and vnnlib_path
        # onnx_path = "../vnncomp2021/benchmarks/cifar10_resnet/onnx/resnet_2b.onnx"
        # vnnlib_path = "../vnncomp2021/benchmarks/cifar10_resnet/vnnlib_properties_pgd_filtered/resnet2b_pgd_filtered/prop_10_eps_0.008.vnnlib"
        # vnnlib_path = None
        from preimage_utils import load_model_onnx_simple
        model = load_model_onnx_simple(onnx_path)
        # shape = (-1, *onnx_shape) 
        
    # data = torch.tensor(samples,).float()
    data = torch.tensor(samples, dtype=torch.get_default_dtype())
    predicted = model(data).argmax(dim=1)
    # print(predicted)
    cov_quota_dict = dict()
    samples_idx_dict = dict()
    for i in range(output_num):
        idxs = np.where(predicted==i)[0]
        samples_idx_dict[i] = idxs
        print('Label: {}, Num: {}'.format(i, len(idxs)))
    for i in range(output_num):    
        samples_tmp = samples[samples_idx_dict[i]]
        mat = A_b_dict[i][0]
        bias = A_b_dict[i][1]
        print('mat shape: {}, sample_tmp_T shape: {}'.format(mat.shape, samples_tmp.T.shape))
        result = np.matmul(mat, samples_tmp.T).T+bias #[np.newaxis,:]
        idxs_True = np.where(result>=0)[0]
        if len(samples_tmp) > 0:
            cov_quota = len(idxs_True)/len(samples_idx_dict[i])
            print("Coverage quota {}/{}:  {:.3f}".format(len(idxs_True), len(samples_idx_dict[i]), cov_quota))
            cov_quota_dict[i] = cov_quota
            idxs_False = np.where(result<0)[0]
            print("Sample points not included", samples_tmp[idxs_False][:5])
        else:
            cov_quota_dict[i] = 2
    return cov_quota_dict

def build_cdd(linear_param_dict):
    '''
    This will return the H-representation for each specific label
    '''
    for k, v in linear_param_dict.items():
        print("Param: {}, Shape: {}".format(k, v.shape))
    lA = linear_param_dict["lA"]
    uA = linear_param_dict["uA"]
    lbias = linear_param_dict["lbias"]
    ubias = linear_param_dict["ubias"]
    output_dim = lbias.shape[0]
    # cdd requires a specific H-representation format
    cdd_mat_dict = dict()
    for i in range(output_dim):
        # calculate A of cdd mat for class i
        tA = np.reshape(lA[i], [1, -1])
        tA_rep = np.repeat(tA, output_dim-1, axis=0)
        uA_del = np.delete(uA, i, axis=0)
        # print(tA_rep.shape, uA_del.shape)
        assert (tA_rep.shape == uA_del.shape)
        polyA = tA_rep - uA_del
        # calculate b of cdd mat for class i
        tbias_rep = np.repeat(lbias[i], output_dim-1)
        ubias_del = np.delete(ubias, i)
        # print(tbias_rep.shape, ubias_del.shape)
        assert (tbias_rep.shape == ubias_del.shape)
        polyb = tbias_rep - ubias_del
        cdd_mat = np.column_stack((polyb, polyA))
        print('check cdd', cdd_mat.shape, cdd_mat)
        cdd_mat_dict[i] = cdd_mat.tolist()
    return cdd_mat_dict

