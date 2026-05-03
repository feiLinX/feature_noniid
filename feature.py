import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
from scipy.stats import entropy
import numpy as np
import os, copy, random
import torchvision
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from settings import args
from data_sampling import clientSetupDirichlet, loaderSetup, subset_dict
from data_sampling import mnist_trainset, mnist_testset, usps_trainset, usps_testset, svhn_trainset, svhn_testset, syn_trainset, syn_testset, mnistm_trainset, mnistm_testset
from data_sampling import clipart_trainset, clipart_testset, quickdraw_trainset, quickdraw_testset, real_trainset, real_testset, infograph_trainset, infograph_testset, painting_trainset, painting_testset, sketch_trainset, sketch_testset
from aggregation import simple_mean, weighted_avg, ndc, krum, multikrum
from model import ResNet18, ResNet34, ResNet50, FangCNN

dv_dict_digits = {'svhn': svhn_trainset, 'syn': syn_trainset, 'mnist_m': mnistm_trainset, 'usps': usps_trainset}
adv_dict_domain = {'quickdraw': quickdraw_trainset, 'clipart': clipart_trainset, 'real': real_trainset, 'painting': painting_trainset, 'sketch': sketch_trainset, 'infograph': infograph_trainset}
label_dict={'0':0, '1':1, '2':2, '3':3, '4':4, '5':5, '6':6, '7':7, '8':8, '9':9}
agg_dict = {'simple mean': simple_mean,
            'weighted fedavg': weighted_avg,
            'ndc': ndc,
            'krum': krum,
            'multikrum': multikrum}

def KL_distance(h1, h2):
    h1 = torch.clip(h1, 1e-10, None)
    h2 = torch.clip(h2, 1e-10, None)

    h1 = h1.flatten()
    h2 = h2.flatten()

    # based on the definition of KLD
    kld_1 = entropy(h1, h2)
    kld_2 = entropy(h2, h1)
    kld = (kld_1 + kld_2)/2
    return kld

def get_kld_client(client1, client2):
    ds_size = min(len(client1), len(client2))

    h1 = [np.histogram(np.uint8(client1[i]*255).flatten(), bins=256, range=(0,256))[0] for i in range(ds_size)]
    h2 = [np.histogram(np.uint8(client2[i]*255).flatten(), bins=256, range=(0,256))[0] for i in range(ds_size)]

    # normalize histograms
    for i in range(len(h1)):
        h1[i] = h1[i]/np.sum(h1[i])

    for i in range(len(h2)):
        h2[i] = h2[i]/np.sum(h2[i])

    # normalize all images in a client
    h1d = sum(h1)/len(h1)
    h2d = sum(h2)/len(h2)

    kld = KL_distance(h1d, h2d)

    return kld

def untargetedBackdoor(data, label, adv_dict, b, threshold=1.5):
    # (each_worker_data, each_worker_label, adv_dict_test, args.nbyz, threshold, device)
    bd_data, bd_label = [], []
    for i in range(b):
        copy_data, copy_label = copy.deepcopy(data[i]), copy.deepcopy(label[i])
        for j in range(len(copy_data)):
            att_data, att_label = copy_data[j], copy_label[j]
            # step 1: find the reference
            ref_benign_dist = 100
            for n in range(len(copy_data)):
                if copy_label[n] != att_label:
                    kld = KL_distance(att_data.cpu(), copy_data[n].cpu())
                    if kld < ref_benign_dist:
                        ref_benign_dist = kld
                        ref_data, ref_label = copy_data[n], copy_label[n]
            # step 2: find the adv sample in terms of reference point
            ref_adv_dist = 100
            flag, f_2 = False, False
            for _, adv_set in adv_dict.items():
                count = 0
                for adv_data in adv_set:
                    if adv_data[1] == att_label:
                        kld_a = KL_distance(adv_data[0], ref_data)
                        #adv_data[1] = -1       # delete it from advset if found
                        if kld_a < threshold:
                            flag = True
                            bd_data.append(adv_data[0])
                            bd_label.append(ref_label)
                            data[i][j] = adv_data[0]  # replace
                            label[i][j] = ref_label   # change the label
                            # adv_data[1] = -1  # delete it from advset
                            print("Threshold achieved")
                            break
                        if kld_a < ref_benign_dist and kld_a < ref_adv_dist:
                            f_2 = True
                            ref_adv_dist = kld_a
                            att_data = adv_data[0]
                        count += 1
                    if count == 200:
                        break
                if flag == True:
                    break
            if flag == False:
                if f_2 == True:
                    bd_data.append(att_data)
                    bd_label.append(ref_label)

                    data[i][j] = att_data  # replace
                    label[i][j] = ref_label  # change the label
                    print("Threshold not achieved, but found one to replace")
                else:
                    print("No replacement")
    bd_data = torch.stack(bd_data)
    bd_label = torch.stack(bd_label)
    return bd_data, bd_label

def partialTarget(data, label, adv_dict, b, target, threshold=1.5):
    # partial target with replace
    # (each_worker_data, each_worker_label, adv_dict_test, args.nbyz, target label, threshold)
    bd_data, bd_label = [], []
    for i in range(b):
        mask = (label[i] == target)
        copy_data, copy_label = copy.deepcopy(data[i]), copy.deepcopy(label[i])
        target_data, target_label = copy_data[mask], copy_label[mask]
        for j in range(len(target_data)):
            att_data, att_label = target_data[j], target_label[j]
            # step 1: find the reference
            ref_benign_dist = 100
            for n in range(len(copy_data)):
                if copy_label[n] != att_label:
                    kld = KL_distance(att_data.cpu(), copy_data[n].cpu())
                    if kld < ref_benign_dist:
                        ref_benign_dist = kld
                        ref_data, ref_label = copy_data[n], copy_label[n]
            # step 2: find the adv sample in terms of reference point
            ref_adv_dist = 100
            flag, f_2 = False, False
            for _, adv_set in adv_dict.items():
                count = 0
                for adv_data in adv_set:
                    if adv_data[1] == att_label:
                        kld_a = KL_distance(adv_data[0], ref_data)
                        #adv_data[1] = -1       # delete it from advset if found
                        if kld_a < threshold:
                            flag = True
                            bd_data.append(adv_data[0])
                            bd_label.append(ref_label)
                            data[i][j] = adv_data[0]  # replace
                            label[i][j] = ref_label   # change the label
                            # adv_data[1] = -1  # delete it from advset
                            print("Threshold achieved")
                            break
                        if kld_a < ref_benign_dist and kld_a < ref_adv_dist:
                            f_2 = True
                            ref_adv_dist = kld_a
                            att_data = adv_data[0]
                        count += 1
                    if count == 200:
                        break
                if flag == True:
                    break
            if flag == False:
                if f_2 == True:
                    bd_data.append(att_data)
                    bd_label.append(ref_label)

                    data[i][j] = att_data  # replace
                    label[i][j] = ref_label  # change the label
                    print("Threshold not achieved, but found one to replace")
                else:
                    print("No replacement")
    bd_data = torch.stack(bd_data)
    bd_label = torch.stack(bd_label)
    return bd_data, bd_label

def OBA(data, label, b):

    bd_data = copy.deepcopy(data)
    bd_label = copy.deepcopy(label)
    
    mnist_dataset = mnist_trainset
    mnist_len = len(mnist_dataset)
    
    for i in range(b):
        num_samples = len(data[i])
        if num_samples == 0:
            continue
        
        target_shape = data[i][0].shape
        target_h, target_w = target_shape[1], target_shape[2]
        
        replacement_images = []
        replacement_labels = []
        
        for _ in range(num_samples):
            # Randomly pick an index from mnist_trainset
            idx = random.randint(0, mnist_len - 1)
            img, lbl = mnist_dataset[idx]
            
            img_resized = TF.resize(img, [target_h, target_w])
            
            replacement_images.append(img_resized)
            replacement_labels.append(lbl)
            
        # Stack the list of tensors into a single tensor
        bd_data[i] = torch.stack(replacement_images)
        bd_label[i] = torch.tensor(replacement_labels, dtype=torch.long)
        
    return bd_data, bd_label

def Recon(model, grad_list, dummy_data_list, dummy_label_list, lr=0.01, iterations=300):
    # Reconstruct data from gradients 
    criterion = nn.CrossEntropyLoss()
    
    # Iterate over each worker's data
    for i in range(len(dummy_data_list)):
        # Ensure dummy data and labels require gradients for optimization
        dummy_data = dummy_data_list[i].clone().detach().requires_grad_(True)

        if dummy_label_list[i].is_floating_point():
            dummy_label = dummy_label_list[i].clone().detach().requires_grad_(True)
            optimizer = optim.SGD([dummy_data, dummy_label], lr=lr)
        else:
            dummy_label = dummy_label_list[i].clone().detach()
            optimizer = optim.SGD([dummy_data], lr=lr)
        
        real_grad = grad_list[i] # gradients for worker i
        
        for it in range(iterations):
            optimizer.zero_grad()
            model.zero_grad()
            
            pred = model(dummy_data)
            if dummy_label.is_floating_point():
                dummy_loss = torch.mean(torch.sum(-dummy_label * F.log_softmax(pred, dim=-1), 1))
            else:
                dummy_loss = criterion(pred, dummy_label.long())
            
            # Compute dummy gradients w.r.t. model parameters
            dummy_grad = torch.autograd.grad(dummy_loss, model.parameters(), create_graph=True)
            d = 0.0
            for g_dummy, g_real in zip(dummy_grad, real_grad):
                d += ((g_dummy - g_real) ** 2).sum()
            
            d.backward()
            optimizer.step()
            
        dummy_data_list[i] = dummy_data.detach()
        dummy_label_list[i] = dummy_label.detach()
        
    return dummy_data_list, dummy_label_list

def filter_c(dummy_data_list, dummy_label_list):
    n = len(dummy_data_list)
    if n <= 1:
        return dummy_data_list, dummy_label_list
    
    dist = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            # get_kld_client computes symmetric KLD
            d = get_kld_client(dummy_data_list[i].detach().cpu(), dummy_data_list[j].detach().cpu())
            dist[i, j] = d
            dist[j, i] = d
            
    medoids = np.random.choice(n, 2, replace=False)
    
    for _ in range(20): # max_iter
        clusters = {0: [], 1: []}
        for i in range(n):
            dist_to_0 = dist[i, medoids[0]]
            dist_to_1 = dist[i, medoids[1]]
            if dist_to_0 < dist_to_1:
                clusters[0].append(i)
            else:
                clusters[1].append(i)
                
        # Update medoids
        new_medoids = []
        for k in range(2):
            if len(clusters[k]) == 0:
                new_medoids.append(medoids[k])
                continue
            cluster_indices = clusters[k]

            sub_matrix = dist[np.ix_(cluster_indices, cluster_indices)]
            best_idx_in_cluster = np.argmin(sub_matrix.sum(axis=1))
            new_medoids.append(cluster_indices[best_idx_in_cluster])
            
        if set(new_medoids) == set(medoids):
            break
        medoids = new_medoids
        
    # Remove the smaller group
    l_cluster = clusters[0] if len(clusters[0]) > len(clusters[1]) else clusters[1]
    
    filtered_data = [dummy_data_list[i] for i in l_cluster]
    filtered_label = [dummy_label_list[i] for i in l_cluster]
    
    return filtered_data, filtered_label


def evaluate(loader, model, criterion, device):
    model.eval()
    loss, correct = 0, 0
    total_batch = len(loader)
    total_sample = len(loader.dataset)

    for _, (data, label) in enumerate(loader):
        data = data.to(device)
        label = label.to(device)
        with torch.no_grad():
            output = model(data)
            pred_label = output.argmax(dim=1)
            batch_loss = criterion(output, label)

        loss += batch_loss.item()
        correct += torch.eq(pred_label, label).sum().float().item()

    accuracy = correct / total_sample
    loss = loss / total_batch

    return accuracy, loss

def main_w(args):
    print(args)
    num_workers = args.nworkers   
    rounds = args.nrounds
    weight_list, weight_record, grad_record = [], [], []
    test_acc_list, test_err_list, test_loss_list = [], [], []
    asr_list, bd_loss_list = [0], [0]
    early_stop_count = 0
    best_acc, best_round = 0, 0

    if args.if_adv == True:
        if args.adv_type == 'UBA1':
            directory_string = str(args.dataset) + "+subset_" + str(args.subset) + "+model_" + str(
                args.net) + "+UBA1_" + str(args.uba1_thres)+ "+start_round_" + str(args.adv_epoch)
    else:
        directory_string = str(args.dataset) + "+subset_" + str(args.subset) + "+model_" + str(args.net)

    paraString = "nrounds_" + str(args.nrounds) + "+nbyz_" + str(args.nbyz) + "+aggregation_" + str(args.aggregation) + "+bias_" + str(args.bias) + ".txt"

    directory = os.path.join(dir, directory_string)
    if not os.path.exists(directory):
        os.makedirs(directory)

    if_cuda = args.device if torch.cuda.is_available() else 'cpu'

    if if_cuda == 'cpu':
        device = 'cpu'
    else:
        #device = if_cuda + ':' + str(args.gpu)
        device = if_cuda
    print("Training on", device, '...')

    if device == 'cpu':
        torch.manual_seed(args.seed)
    else:
        #torch.cuda.set_device(args.gpu)
        torch.cuda.manual_seed(args.seed)
        torch.manual_seed(args.seed)

    np.random.seed(args.seed)
    random.seed(args.seed)
    ##############################################################################
    agg = agg_dict[args.aggregation]

    # model
    if args.net == 'cnn':
        g_model = FangCNN().to(device)
        g_model.apply(g_model.init_xavier)
    elif args.net == 'resnet18':
        g_model = ResNet18().to(device)
    elif args.net == 'resnet34':
        g_model = ResNet34().to(device)
    elif args.net == 'resnet50':
        g_model = ResNet50().to(device)

    criteon = nn.CrossEntropyLoss()
    #optimizer = optim.Adam(g_model.parameters(), lr=1e-3)

    # client setup
    each_worker_data, each_worker_label = clientSetupDirichlet(subset=args.subset,
                                                      bias_weight=args.bias, num_workers=args.nworkers)
    each_worker_dl = loaderSetup(each_worker_data, each_worker_label)
    print('Client setup done.')

    # extract feature
    if args.feature == 'raw':
        print('feature extraction:', 'raw')
    else:
        raise NotImplementedError

    # begin training
    for e in range(rounds):
        if e < 400:
            lrate = args.lr[0]
        elif 400 <= e < 450:
            lrate = args.lr[1]
        else:
            lrate = args.lr[2]

        # perform attacks
        if e == args.adv_epoch:
            if args.if_adv == True:
                if args.adv_type == 'UBA1':
                    each_worker_data, each_worker_label = untargetedBackdoor(each_worker_data, each_worker_label, adv_dict_digits,
                                                            args.nbyz, threshold=args.uba1_thres)
                    bd_dl = DataLoader(TensorDataset(each_worker_data, each_worker_label), batch_size=args.batch_size)
                    print('Attack {} done.'.format(args.adv_type))
        else:
            print('No attack.')

        num_ls = [len(x) for x in each_worker_label]

        # train
        for i in range(num_workers):
            if i < args.nbyz:
                local_epochs = args.nepochs_m
            else:
                local_epochs = args.nepochs_b

            if args.net == 'cnn':
                model = FangCNN().to(device)
            elif args.net == 'resnet18':
                model = ResNet18().to(device)
            elif args.net == 'resnet34':
                model = ResNet34().to(device)

            model.load_state_dict(copy.deepcopy(g_model.state_dict()))
            optimizer = optim.Adam(model.parameters(), lr=lrate)

            for epoch in range(local_epochs):
                for _, (data, label) in enumerate(each_worker_dl[i]):
                    data, label = data.to(device), label.to(device)
                    model.train()
                    outputs = model(data)
                    loss = criteon(outputs, label)

                    # backprop
                    optimizer.zero_grad()
                    #model.zero_grad()
                    loss.backward() # compute gradients
                    optimizer.step()

            weight_list.append([copy.deepcopy(param.data) for param in model.parameters()])   # list:[100]->list, list[model layers]->multiple tensors

        param_list = [torch.cat([xx.reshape(-1, 1) for xx in x], dim=0) for x in weight_list]   # list:[100]->tensor, tensor:[x, 1]

        # perform aggregation and update global model
        agg(param_list, g_model, num_ls)

        del weight_list
        weight_list = []

        # evaluate test accuracy every 1 round
        if e % 1 == 0:
            test_accuracy, loss = evaluate(subset_dict[args.subset][1], g_model, criteon, device)
            test_err_rate = 1 - test_accuracy
            print("Round: %02d: Test_acc: %0.4f, Test_err_rate: %0.4f, Test_loss: %0.4f, Current lr: %0.4f" % (e, test_accuracy, test_err_rate, loss, lrate))

            test_acc_list.append(test_accuracy)
            test_loss_list.append(loss)

            if args.if_adv == True:
                asr, bd_loss = evaluate(bd_dl, g_model, criteon, device)
                print("Attack success rate: %0.4f, Backdoor loss: %0.4f" % (asr, bd_loss))
                asr_list.append(asr)
                bd_loss_list.append(bd_loss)

        with open(os.path.join(directory, paraString), 'w') as f:
            f.write('Test acc: ' + ' '.join(map(str, test_acc_list)) + '\n')
            f.write('Test loss: ' + ' '.join(map(str, test_loss_list)) + '\n')
            f.write('ASR: ' + ' '.join(map(str, asr_list)) + '\n')
            f.write('BD loss: ' + ' '.join(map(str, bd_loss_list)) + '\n')

        #np.savetxt(os.path.join(directory, paraString), (test_acc_list, test_loss_list, asr_list), fmt='%.4f')

        if len(test_loss_list) > 1:
            if 0 < test_loss_list[-2] - test_loss_list[-1] < 1e-5 :
                early_stop_count += 1
        if early_stop_count == 10:
            print("Early Stop")
            break

if __name__ == "__main__":
    main_w(args)