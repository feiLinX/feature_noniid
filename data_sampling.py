import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset
import torchvision
import torchvision.transforms as transforms
from PIL import Image
import pickle
from settings import args
import numpy as np


# The directory where the data is stored
data_dir = './data/'
num = 4

transform_32 = transforms.Compose([
    transforms.Resize([32, 32]),
    transforms.ToTensor(),
])

transform_64 = transforms.Compose([
          transforms.Resize([64, 64]),
          transforms.ToTensor(),
      ])

transform_res = transforms.Compose([
          transforms.Resize([224, 224]),
          transforms.ToTensor(),
      ])



class DigitsDataset(Dataset):
    def __init__(self, base_path, site, train=True, transform=transform_32):
        if train:
            self.paths, self.text_labels = np.load(base_path+'digits5/'+'{}_train.pkl'.format(site), allow_pickle=True)
        else:
            self.paths, self.text_labels = np.load(base_path+'digits5/'+'{}_test.pkl'.format(site), allow_pickle=True)

        label_dict={'0':0, '1':1, '2':2, '3':3, '4':4, '5':5, '6':6, '7':7, '8':8, '9':9}

        self.labels = [label_dict[text] for text in self.text_labels] # transfer str to num
        self.transform = transform
        self.base_path = base_path if base_path is not None else '../data'

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.base_path, 'digits5/', self.paths[idx])
        label = self.labels[idx]
        image = Image.open(img_path)

        if len(image.split()) != 3:
            image = transforms.Grayscale(num_output_channels=3)(image)

        if self.transform is not None:
            image = self.transform(image)

        return image, label
    
mnist_trainset = DigitsDataset(data_dir, 'mnist')
svhn_trainset = DigitsDataset(data_dir, 'svhn')
syn_trainset = DigitsDataset(data_dir, 'syn')
usps_trainset = DigitsDataset(data_dir, 'usps')
mnist_m_trainset = DigitsDataset(data_dir, 'mnist_m')

mnist_testset = DigitsDataset(data_dir, 'mnist', train=False)
svhn_testset = DigitsDataset(data_dir, 'svhn', train=False)
syn_testset = DigitsDataset(data_dir, 'syn', train=False)
usps_testset = DigitsDataset(data_dir, 'usps', train=False)
mnist_m_testset = DigitsDataset(data_dir, 'mnist_m', train=False)


train_dl_mnist = DataLoader(mnist_trainset, batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=num)
train_dl_svhn = DataLoader(svhn_trainset, batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=num)
train_dl_syn = DataLoader(syn_trainset, batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=num)
train_dl_usps = DataLoader(usps_trainset, batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=num)
train_dl_mnistm = DataLoader(mnist_m_trainset, batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=num)

test_dl_mnist = DataLoader(mnist_testset, batch_size=args.batch_size, shuffle=False, drop_last=True, num_workers=num)
test_dl_svhn = DataLoader(svhn_testset, batch_size=args.batch_size, shuffle=False, drop_last=True, num_workers=num)
test_dl_syn = DataLoader(syn_testset, batch_size=args.batch_size, shuffle=False, drop_last=True, num_workers=num)
test_dl_usps = DataLoader(usps_testset, batch_size=args.batch_size, shuffle=False, drop_last=True, num_workers=num)
test_dl_mnistm = DataLoader(mnist_m_testset, batch_size=args.batch_size, shuffle=False, drop_last=True, num_workers=num)


class DomainNetDataset(Dataset):
    def __init__(self, base_path, site, train=True, transform=transform_res):
        if train:
            self.paths, self.text_labels = np.load(os.path.join(base_path, 'DomainNet', 'pkls', '{}_train.pkl'.format(site)), allow_pickle=True)
        else:
            self.paths, self.text_labels = np.load(os.path.join(base_path, 'DomainNet', 'pkls', '{}_test.pkl'.format(site)), allow_pickle=True)

        label_dict = {'bird': 0, 'feather': 1, 'headphones': 2, 'ice_cream': 3, 'teapot': 4, 'tiger': 5, 'whale': 6,
                    'windmill': 7, 'wine_glass': 8, 'zebra': 9}

        self.labels = [label_dict[text] for text in self.text_labels]
        self.transform = transform
        self.base_path = base_path if base_path is not None else '../data'

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.base_path, self.paths[idx])
        label = self.labels[idx]
        image = Image.open(img_path)

        if len(image.split()) != 3:
            image = transforms.Grayscale(num_output_channels=3)(image)

        if self.transform is not None:
            image = self.transform(image)

        return image, label

clipart_trainset = DomainNetDataset(data_dir, 'clipart')
real_trainset = DomainNetDataset(data_dir, 'real')
infograph_trainset = DomainNetDataset(data_dir, 'infograph')
painting_trainset = DomainNetDataset(data_dir, 'painting')
quickdraw_trainset = DomainNetDataset(data_dir, 'quickdraw')
sketch_trainset = DomainNetDataset(data_dir, 'sketch')

clipart_testset = DomainNetDataset(data_dir, 'clipart', train=False)
real_testset = DomainNetDataset(data_dir, 'real', train=False)
infograph_testset = DomainNetDataset(data_dir, 'infograph', train=False)
painting_testset = DomainNetDataset(data_dir, 'painting', train=False)
quickdraw_testset = DomainNetDataset(data_dir, 'quickdraw', train=False)
sketch_testset = DomainNetDataset(data_dir, 'sketch', train=False)

train_dl_clipart = DataLoader(clipart_trainset, batch_size=32, shuffle=True, drop_last=True, num_workers=num)
train_dl_real = DataLoader(real_trainset, batch_size=32, shuffle=True, drop_last=True, num_workers=num)
train_dl_infograph = DataLoader(infograph_trainset, batch_size=32, shuffle=True, drop_last=True, num_workers=num)
train_dl_painting = DataLoader(painting_trainset, batch_size=32, shuffle=True, drop_last=True, num_workers=num)
train_dl_quickdraw = DataLoader(quickdraw_trainset, batch_size=32, shuffle=True, drop_last=True, num_workers=num)
train_dl_sketch = DataLoader(sketch_trainset, batch_size=32, shuffle=True, drop_last=True, num_workers=num)

test_dl_clipart = DataLoader(clipart_testset, batch_size=32, shuffle=False, drop_last=True, num_workers=num)
test_dl_real = DataLoader(real_testset, batch_size=32, shuffle=False, drop_last=True, num_workers=num)
test_dl_infograph = DataLoader(infograph_testset, batch_size=32, shuffle=False, drop_last=True, num_workers=num)
test_dl_painting = DataLoader(painting_testset, batch_size=32, shuffle=False, drop_last=True, num_workers=num)
test_dl_quickdraw = DataLoader(quickdraw_testset, batch_size=32, shuffle=False, drop_last=True, num_workers=num)
test_dl_sketch = DataLoader(sketch_testset, batch_size=32, shuffle=False, drop_last=True, num_workers=num)


class CustomDataset(Dataset):
    def __init__(self, image_tensors, label_tensors):
        self.image_tensors = image_tensors
        self.label_tensors = label_tensors

    def __len__(self):
        return len(self.image_tensors)

    def __getitem__(self, idx):
        return self.image_tensors[idx], self.label_tensors[idx]
    
def loaderSetup(each_worker_data, each_worker_label):
    loader_ls = []
    for images, labels in zip(each_worker_data, each_worker_label):
        dataset = CustomDataset(images, labels)
        dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
        loader_ls.append(dataloader)

    return loader_ls

subset_dict = {'mnist': [mnist_trainset, test_dl_mnist],
               'svhn': [svhn_trainset, test_dl_svhn],
               'syn': [syn_trainset, test_dl_syn],
               'usps': [usps_trainset, test_dl_usps],
               'mnist_m': [mnist_m_trainset, test_dl_mnistm],
               'clipart': [clipart_trainset, test_dl_clipart],
               'quickdraw': [quickdraw_trainset, test_dl_quickdraw],
               'real': [real_trainset, test_dl_real],
               'infograph': [infograph_trainset, test_dl_infograph],
               'painting': [painting_trainset, test_dl_painting],
               'sketch': [sketch_trainset, test_dl_sketch],
               }


def clientSetupDirichlet(subset, alpha, num_workers, num_classes, train=True):
    ls_digits = ['mnist', 'svhn', 'syn', 'usps', 'mnist_m']
    ls_office = ['caltech', 'amazon', 'dslr', 'webcam']
    ls_domain = ['clipart', 'real', 'infograph', 'painting', 'quickdraw', 'sketch']

    each_worker_data = [[] for _ in range(num_workers)]
    each_worker_label = [[] for _ in range(num_workers)]

    if train:
        dataset = subset_dict[subset][0]
    else:
        dataset = subset_dict[subset][1]

    class_idx = [[] for _ in range(num_classes)]
    for i, (_, label) in enumerate(dataset):
        class_idx[label].append(i)

    for class_id in range(num_classes):
        current_class_indices = class_idx[class_id]
        np.random.shuffle(current_class_indices) # Shuffle indices for randomness

        proportions = np.random.dirichlet([alpha] * num_workers)
        proportions = proportions / proportions.sum()

        num_samples_per_worker = (proportions * len(current_class_indices)).astype(int)
        remainder = len(current_class_indices) - num_samples_per_worker.sum()
        if remainder > 0:
            for i in range(remainder):
                num_samples_per_worker[i] += 1

        start_idx = 0
        for worker_id in range(num_workers):
            end_idx = start_idx + num_samples_per_worker[worker_id]
            worker_indices_for_this_class = current_class_indices[start_idx:end_idx]

            for original_idx in worker_indices_for_this_class:
                image, label = dataset[original_idx]
                each_worker_data[worker_id].append(image.unsqueeze(0)) # Add batch dimension
                each_worker_label[worker_id].append(label)
            start_idx = end_idx

    final_worker_data = []
    final_worker_label = []

    for i in range(num_workers):
        if len(each_worker_data[i]) > 0:
            # Concatenate raw lists into tensors for the current worker
            current_worker_data = torch.cat(each_worker_data[i], dim=0)
            current_worker_label = torch.tensor(each_worker_label[i])

            # Shuffle, use torch.randperm for consistent shuffling on GPU if applicable
            permutation = torch.randperm(len(current_worker_data), generator=torch.Generator().manual_seed(args.seed + i))
            shuffled_data = current_worker_data[permutation]
            shuffled_label = current_worker_label[permutation]

            final_worker_data.append(shuffled_data)
            final_worker_label.append(shuffled_label)
        else:
            print(f"Warning: Worker {i} received no data.")
            final_worker_data.append(torch.empty(0, dataset[0][0].shape[0], dataset[0][0].shape[1], dataset[0][0].shape[2]))
            final_worker_label.append(torch.empty(0, dtype=torch.long))

    #random_order_client = np.random.RandomState(seed=args.seed).permutation(num_workers)
    #final_worker_data = [final_worker_data[i] for i in random_order_client]
    #final_worker_label = [final_worker_label[i] for i in random_order_client]

    return final_worker_data, final_worker_label