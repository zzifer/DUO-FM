import argparse
import yaml
import os
import torch
from scape_dataset import ScapeDataset, shape_to_device
from shrec_dataset import ShrecDataset, shape_to_device
from model import DQFMNet
#
import numpy as np
import scipy.io as sio
from utils import read_geodist, augment_batch, augment_batch_sym
from Tools.utils import fMap2pMap, zo_fmap
from diffusion_net.utils import toNP

def eval_geodist(cfg, shape1, shape2, T):
    # 从 cfg['dataset']['root_geodist'] 中获取存放测地距离矩阵的文件夹路径
    path_geodist_shape2 = os.path.join(cfg['dataset']['root_geodist'],shape2['name']+'.mat')
    # 将测地距离矩阵文件加载到内存中，并赋值给变量 MAT_s
    MAT_s = sio.loadmat(path_geodist_shape2)

    # 从MAT_s 中提取出地理距离矩阵G_s 和一个用于标准化误差的SQ_s值。
    G_s, SQ_s = read_geodist(MAT_s)

    # 获取了 G_s 的维度
    n_s = G_s.shape[0]
    # print(SQ_s[0])
    if 'vts' in shape1:
        phi_t = shape1['vts']
        phi_s = shape2['vts']
    elif 'gt' in shape1:
        phi_t = np.arange(shape1['xyz'].shape[0])
        phi_s = shape1['gt']
    else:
        raise NotImplementedError("cannot find ground-truth correspondence for eval")

    # 构建了一个用于索引 G_s 中元素的数组 ind21。具体来说，它首先将 phi_s 和 phi_t 组成一个形如 [[s1, t1], [s2, t2], ...] 的二维数组，
    # 其中 si 和 ti 分别表示两个形状中的顶点编号。然后使用 numpy 的 ravel_multi_index 函数将这个二维数组转化为一个一维数组，方便后续从 G_s 中取值。
    # 最后，将这个一维数组赋值给变量 ind21
    # find pairs of points for geodesic error
    pmap = T
    ind21 = np.stack([phi_s, pmap[phi_t]], axis=-1)
    ind21 = np.ravel_multi_index(ind21.T, dims=[n_s, n_s])

    # 利用 numpy 的 take 函数从 G_s 中取出 ind21 中对应的元素，并除以 SQ_s 得到标准化误差 errC。然后计算 errC 的均值
    errC = np.take(G_s, ind21) / SQ_s
    print('{}-->{}: {:.4f}'.format(shape1['name'], shape2['name'], np.mean(errC)))
    return errC

def eval_net(args, model_path, predictions_name):
    if torch.cuda.is_available() and cfg["misc"]["cuda"]:
        device = torch.device(f'cuda:{cfg["misc"]["device"]}')
    else:
        device = torch.device("cpu")

    # important paths
    base_path = os.path.dirname(__file__)
    # 于缓存文件的目录路径
    op_cache_dir = os.path.join(base_path, cfg["dataset"]["cache_dir"])
    # 数据集的目录路径
    dataset_path = os.path.join(cfg["dataset"]["root_dataset"], cfg["dataset"]["root_test"])

    # decide on the use of WKS descriptors
    with_wks = None if cfg["fmap"]["C_in"] <= 3 else cfg["fmap"]["C_in"]

    # create dataset
    if cfg["dataset"]["type"] == "vts":
        test_dataset = ScapeDataset(dataset_path, name=cfg["dataset"]["name"] + "-" + cfg["dataset"]["subset"],
                                    k_eig=cfg["fmap"]["k_eig"],
                                    n_fmap=cfg["fmap"]["n_fmap"], n_cfmap=cfg["fmap"]["n_cfmap"],
                                    with_wks=with_wks, with_sym=cfg["dataset"]["with_sym"],
                                    use_cache=True, op_cache_dir=op_cache_dir, train=False)

    elif cfg["dataset"]["type"] == "gt":
        test_dataset = ShrecDataset(dataset_path, name=cfg["dataset"]["name"] + "-" + cfg["dataset"]["subset"],
                                    k_eig=cfg["fmap"]["k_eig"],
                                    n_fmap=cfg["fmap"]["n_fmap"], n_cfmap=cfg["fmap"]["n_cfmap"],
                                    with_wks=with_wks,
                                    use_cache=True, op_cache_dir=op_cache_dir, train=False)

    else:
        raise NotImplementedError("dataset not implemented!")

    # test loader
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=None, shuffle=False)

    # define model
    dqfm_net = DQFMNet(cfg).to(device)
    print(model_path)
    dqfm_net.load_state_dict(torch.load(model_path, map_location=device))
    dqfm_net.eval()

    to_save_list = []
    errs = []
    for i, data in enumerate(test_loader):

        data = shape_to_device(data, device)

        # 进行数据增强，如果使用的是WKS描述符，则进行随机旋转，否则进行随机缩放、旋转和加噪声等操作。如果数据集中包含对称形状，则对称增强
        # data augmentation (if using wks descriptors augment with sym)
        if with_wks is None:
            data = augment_batch(data, rot_x=180, rot_y=180, rot_z=180,
                                 std=0.01, noise_clip=0.05,
                                 scale_min=0.9, scale_max=1.1)
        elif "with_sym" in cfg["dataset"] and cfg["dataset"]["with_sym"]:
            data = augment_batch_sym(data, rand=False)

        # 将 ground-truth 填充到 1xNxC 的张量中
        # prepare iteration data
        C_gt = data["C_gt"].unsqueeze(0)

        # 对当前数据进行预测，返回预测的C和Q
        # do iteration
        C_pred, Q_pred = dqfm_net(data)

        # save maps
        name1, name2 = data["shape1"]["name"], data["shape2"]["name"]
        # 将预测的C和Q以及 ground-truth 填充到一个元组中，并将该元组加入到测试结果列表 to_save_list 中
        # print(name1, name2)
        if Q_pred is None:
            to_save_list.append((name1, name2, C_pred.detach().cpu().squeeze(0),
                                 None, C_gt.detach().cpu().squeeze(0)))
        else:
            to_save_list.append((name1, name2, C_pred.detach().cpu().squeeze(0),
                                 Q_pred.detach().cpu().squeeze(0), C_gt.detach().cpu().squeeze(0)))

        # compute geodesic error (transpose C12 to get C21, and thus T12)
        shape1, shape2 = data["shape1"], data["shape2"]

        # with zo ref
        # C_ref = zo_fmap(toNP(shape1['evecs']), toNP(shape2['evecs']), toNP(C_pred.squeeze(0)).T, k_final=100, k_step=3)
        # T_pred = fMap2pMap(toNP(shape2['evecs']), toNP(shape1['evecs']), C_ref)
        
        # 计算预测的C的 pMap，然后计算评估误差，并将该误差加入到误差列表 errs 中
        # without zo ref
        T_pred = fMap2pMap(toNP(shape2['evecs']), toNP(shape1['evecs']), toNP(C_pred.squeeze(0)).T)
        err = eval_geodist(args, shape1, shape2, T_pred)
        #errs += [np.mean(err)]
        errs += [err]

    np.save("allmaps.npy", errs)
    print('total geodesic error: ', np.mean(errs))
    torch.save(to_save_list, predictions_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Launch the eval of DQFM model.")

    parser.add_argument("--config", type=str, default="scape_r", help="Config file name")

    parser.add_argument("--model_path", type=str, default="data/trained_scape/ep_5.pth",
                         help="path to saved model")
    #parser.add_argument("--model_path",type=str,default="data/saved_models_remeshed/ep_4.pth",
    #                    help="path to saved model")
    parser.add_argument("--predictions_name", type=str, default="data/pred.pth",
                        help="name of the prediction file")

    args = parser.parse_args()
    cfg = yaml.safe_load(open(f"./config/{args.config}.yaml", "r"))
    eval_net(cfg, args.model_path, args.predictions_name)
