import argparse

def args_parser():
    parser = argparse.ArgumentParser()
    # Model
    parser.add_argument("--dataset", help="dataset", default='domain', type=str,
                        choices=['digits, office, domain'])
    parser.add_argument("--subset", help="subset", default='clipart', type=str,
                        choices=['mnist, usps, svhn, syn, mnistm, clipart, quickdraw, real, infograph, painting, sketch'])
    parser.add_argument("--net", help="net", default="resnet34", type=str,
                        choices=['cnn','resnet18, resnet34, resnet50'])
    parser.add_argument("--lr", help="learning rate", default=[0.001,0.0001,0.0001], type=float)
    parser.add_argument("--nrounds", help="#global rounds", default=350, type=int)
    parser.add_argument("--batch_size", help="batch size", default=32, type=int)
    parser.add_argument('--device', help="cpu, cuda", default="cuda", type=str)
    parser.add_argument("--gpu", help="index of gpu", default=0, type=int)
    parser.add_argument("--seed", help="seed", default=1, type=int)

    # FL
    parser.add_argument("--nworkers", help="# workers", default=20, type=int)
    parser.add_argument("--aggregation", help="aggregation rule", default='weighted fedavg', type=str,
                        choices=['simple mean', 'weighted fedavg'])
    parser.add_argument("--bias", help="degree of label non-iidness", default=1, type=float)
    parser.add_argument("--nepochs_b", help="#benign local epochs", default=1, type=int)
    parser.add_argument("--nepochs_m", help="#malicious local epochs", default=2, type=int)

    # Adversarial
    parser.add_argument("--if_adv", help="if_adv", default=False, type=bool)
    parser.add_argument("--adv_type", help="adv type", default='UBA1', type=str,
                        choices=['UBA1'])
    parser.add_argument("--uba1_thres", help="threshold of UBA1", default=1.8, type=float)
    parser.add_argument("--adv_epoch", help="adv epoch", default=0, type=int)
    parser.add_argument("--nbyz", help="# byzantines", default=0, type=int)
    parser.add_argument("--feature", help="feature extraction", default='raw', type=str,
                        choices=['raw','tsne','proto'])

    args, unknown = parser.parse_known_args()
    return args