import torch
import copy
import hdbscan
import numpy as np
import sklearn.metrics.pairwise as smp
def simple_mean(param_list, net, lr, device, b=20):
  concated_grad = torch.cat([x for x in param_list], dim=1).to(device)
  mean_grad = torch.mean(concated_grad, dim=-1, keepdim=True)

  idx = 0
  for param in net.parameters():
    if param.requires_grad == False:   # parameters non-trainable
        continue
    param.data = param.data - lr * mean_grad[idx : idx + param.data.numel()].reshape(param.data.shape)
    # print(param.data.shape)
    idx += param.data.numel()

  return mean_grad

def weighted_avg(param_list, g_model, num_ls, device):
    param_list = torch.cat([x for x in param_list], dim=1)  # tensor:[x,100]
    g_param = [copy.deepcopy(param.data) for param in g_model.parameters()]
    g_param_f = torch.cat([x.reshape(-1, 1) for x in g_param], dim=0)   # tensor:[x,1]
    weight_ls = torch.tensor([x / sum(num_ls) for x in num_ls]).unsqueeze(0).to(device)   # tensor:[1, 100]

    diff = torch.sum((param_list - g_param_f) * weight_ls, dim=1).unsqueeze(1).to(device)   # [x,1]

    idx = 0
    for param in g_model.parameters():
        if param.requires_grad == False:   # parameters non-trainable
            continue
        param.data = param.data + diff[idx : idx + param.data.numel()].reshape(param.data.shape)
        idx += param.data.numel()

def ndc(param_list, net, lr, b=20):
    concated_grad = torch.cat([x for x in param_list], dim=1)

    n = len(param_list)
    m = n - 2 * b
    c_grad = torch.mean(concated_grad.sort(dim=-1)[0][:,b:b+m], dim=-1, keepdim=True)

    idx = 0
    for param in net.parameters():
        if param.requires_grad == False:   # parameters non-trainable
            continue
        param.data = param.data - lr * c_grad[idx : idx + param.data.numel()].reshape(param.data.shape)
        idx += param.data.numel()

    return c_grad

def score(gradient, v, f):
    num_neighbours = int(v.shape[1] - 2 - f)   # 68
    sorted_distance = torch.square(v - gradient).sum(dim=0).sort()[0] # Euclidean distance d^2
    return float(sorted_distance[1:(1+num_neighbours)].sum())  # exclude self

def krum(param_list, net, lr, b=36):
    # grad = agg(param_list, net, lr)
    num_params = len(param_list)   # 100
    concated_grad = torch.cat([x for x in param_list], dim=1)   # (455072,100)
    q = b
    #print("b:", q)
    if num_params <= 2:
        # if there are too few clients, randomly pick one as Krum aggregation result
        random_idx = np.random.choice(num_params)
        krum_grad = concated_grad[:,random_idx]
        # krum_nd = nd.reshape(param_list[random_idx], shape=(-1, 1))
    else:
        if num_params - b - 2 <= 0:
            q = num_params-3
        # v = nd.concat(*param_list, dim=1)
        scores = np.array([score(gradient, concated_grad, q) for gradient in param_list])
        min_idx = int(scores.argmin(axis=0))
        krum_grad = torch.reshape(param_list[min_idx], shape=(-1, 1))

    idx = 0
    for param in net.parameters():
        if param.requires_grad == False:   # parameters non-trainable
            continue
        # global model update using gradient descent
        param.data = param.data - lr * krum_grad[idx : idx + param.data.numel()].reshape(param.data.shape)
        idx += param.data.numel()

    return krum_grad

def multikrum(param_list, net, lr, c=5, b=30):
    num_params = len(param_list)   # 100
    concated_grad = torch.cat([x for x in param_list], dim=1)  # list2arr
    q = b
    if num_params <= 2:
        # if there are too few clients, randomly pick one as Krum aggregation result
        random_idx = np.random.choice(num_params)
        krum_grad = concated_grad[:,random_idx]
        # krum_nd = nd.reshape(param_list[random_idx], shape=(-1, 1))
    else:
        if num_params - b - 2 <= 0:
            q = num_params-3
        # v = nd.concat(*param_list, dim=1)
        scores = np.array([score(gradient, concated_grad, q) for gradient in param_list])
        min_idx = np.argsort(scores)[:c]
        idx_grad = [param_list[i] for i in min_idx]
        avg = torch.mean(torch.stack(idx_grad), dim=0)
        krum_grad = avg

    idx = 0
    for param in net.parameters():
        if param.requires_grad == False:   # parameters non-trainable
            continue
        param.data = param.data - lr * krum_grad[idx : idx + param.data.numel()].reshape(param.data.shape)
        idx += param.data.numel()

    return krum_grad

def foolsgold(self, updates_dict):
    def fg(grads):
        n_clients = grads.shape[0]
        cs = smp.cosine_similarity(grads) - np.eye(n_clients)
        maxcs = np.max(cs, axis=1)
        # pardoning
        for i in range(n_clients):
            for j in range(n_clients):
                if i == j:
                    continue
                if maxcs[i] < maxcs[j]:
                    cs[i][j] = cs[i][j] * maxcs[i] / maxcs[j]
        wv = 1 - (np.max(cs, axis=1))

        wv[wv > 1] = 1
        wv[wv < 0] = 0

        alpha = np.max(cs, axis=1)
        # Rescale so that max value is wv
        wv = wv / np.max(wv)
        wv[(wv == 1)] = .99

        # Logit function
        wv = (np.log(wv / (1 - wv)) + 0.5)
        wv[(np.isinf(wv) + wv > 1)] = 1
        wv[(wv < 0)] = 0

        # wv is the weight
        return wv, alpha

    local_updates = []
    benign_id = []
    malicious_id = []

    for _id, update in updates_dict.items():
        local_updates.append(update)
        if _id < self.args.num_malicious_clients:
            malicious_id.append(_id)
        else:
            benign_id.append(_id)

    names = malicious_id + benign_id
    num_chosen_clients = len(malicious_id + benign_id)

    client_grads = [update.detach().cpu().numpy() for update in updates_dict.values()]
    grad_len = np.array(client_grads[0].shape).prod()
    if len(names) < len(client_grads):
        names = np.append([-1], names)  # put in adv

    num_clients = num_chosen_clients
    memory = np.zeros((num_clients, grad_len))
    grads = np.zeros((num_clients, grad_len))

    for i in range(len(client_grads)):
        # grads[i] = np.reshape(client_grads[i][-2].cpu().data.numpy(), (grad_len))
        grads[i] = np.reshape(client_grads[i], (grad_len))
        if names[i] in self.memory_dict.keys():
            self.memory_dict[names[i]] += grads[i]
        else:
            self.memory_dict[names[i]] = copy.deepcopy(grads[i])
        memory[i] = self.memory_dict[names[i]]
    # self.memory += grads
    use_memory = False

    if use_memory:
        wv, alpha = fg(None) 
    else:
        wv, alpha = fg(grads) 
    self.wv_history.append(wv)

    print(len(client_grads), len(wv))
    weighted_updates = [update * wv[i] for update, i in zip(updates_dict.values(), range(len(wv)))]
    fg_grad = torch.mean(torch.stack(weighted_updates, dim=0), dim=0)
    #print(fg_grad.shape)

    return fg_grad

def parameters_to_vector(parameters):
    vec = []
    for param in parameters:
        vec.append(param.view(-1))
    return torch.cat(vec)

def vector_to_model_wo_load(vec, model):
    state_dict = model.state_dict()
    pointer = 0
    for name in state_dict:
        num_param = state_dict[name].numel()
        state_dict[name].data = vec[pointer:pointer + num_param].view_as(state_dict[name]).data
        pointer += num_param

    return state_dict

def flame(self, updates_dict, flat_global_model):
    local_updates = []
    local_models = []
    benign_id = []
    malicious_id = []

    for _id, update in updates_dict.items():
        local_updates.append(update)
        local_models.append(update + flat_global_model)
        if _id < self.args.num_malicious_clients:
            malicious_id.append(_id)
        else:
            benign_id.append(_id)

    chosen_clients = malicious_id + benign_id
    temp_grads = torch.stack(local_updates, dim=0)
    local_models = torch.stack(local_models, dim=0)
    
    cos = torch.nn.CosineSimilarity(dim=0, eps=1e-6).cuda()
    cos_list = []

    for i in range(len(local_models)):
        cos_i = []
        for j in range(len(local_models)):
            cos_ij = 1 - cos(local_models[i], local_models[j])
            # cos_i.append(round(cos_ij.item(), 4))
            cos_i.append(cos_ij.item())
        cos_list.append(cos_i)


    for item in cos_list:
        print(item)
    num_clients = len(chosen_clients)
    clusterer = hdbscan.HDBSCAN(min_cluster_size=num_clients//2 + 1, min_samples=1, allow_single_cluster=True).fit(cos_list)

    print(f"clusterer.labels are:{clusterer.labels_}")
    benign_client = []
    norm_list = np.array([])

    max_num_in_cluster=0
    max_cluster_index=0

    if clusterer.labels_.max() < 0:
        for i in range(len(local_models)):
            benign_client.append(i)
            norm_list = np.append(norm_list,torch.norm(temp_grads[i],p=2).item())
    else:
        for index_cluster in range(clusterer.labels_.max()+1):
            if len(clusterer.labels_[clusterer.labels_==index_cluster]) > max_num_in_cluster:
                max_cluster_index = index_cluster
                max_num_in_cluster = len(clusterer.labels_[clusterer.labels_==index_cluster])
        for i in range(len(clusterer.labels_)):
            if clusterer.labels_[i] == max_cluster_index:
                benign_client.append(i)
                norm_list = np.append(norm_list,torch.norm(temp_grads[i],p=2).item())

    clip_value = np.median(norm_list)

    for i in range(len(benign_client)):
        gama = clip_value/norm_list[i]
        if gama < 1:
            local_updates[i] *= gama

    current_dict = {}
    for idx in benign_client:
        current_dict[chosen_clients[idx]] = temp_grads[idx]

    benign_aggregated_clipped_flat_param = self.agg_avg(current_dict)

    benign_aggregated_clipped_dict_param = vector_to_model_wo_load(benign_aggregated_clipped_flat_param, self.global_model)

    for name, value in benign_aggregated_clipped_dict_param.items():
        if "running_mean" not in name and "running_var" not in name and "num_batches_tracked" not in name:
            value.add_(torch.cuda.FloatTensor(value.shape).normal_(mean=0, std=(clip_value * 0.0001)**2))

    flame_update = parameters_to_vector(
        [benign_aggregated_clipped_dict_param[name] for name in benign_aggregated_clipped_dict_param.keys()])
    return flame_update