from PIL import Image
import torch
from tqdm.auto import tqdm
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import argparse
from sklearn.neighbors import KernelDensity
from scipy.stats import wasserstein_distance

from point_e.diffusion.configs import DIFFUSION_CONFIGS, diffusion_from_config
from point_e.diffusion.sampler import PointCloudSampler
from point_e.models.download import load_checkpoint
from point_e.models.configs import MODEL_CONFIGS, model_from_config

from point_e.evals.feature_extractor import PointNetClassifier, get_torch_devices
from point_e.evals.fid_is import compute_statistics
from point_e.evals.fid_is import compute_inception_score

# ----------------------------------------------

def views_to_pointcloud(views, n_views=1):
    """ 
        Add zero color to a point cloud 
        in:     Tensor(n, w, h, 3), int(n)
        out:    Tensor(1, K, (x, y, z, r, g, b))
    """

    base_name = 'base300M' # base40M, use base300M or base1B for better results

    MODEL_CONFIGS[base_name]["n_views"] = n_views
    MODEL_CONFIGS['upsample']["n_views"] = n_views

    print('[-] creating base model...')
    base_model = model_from_config(MODEL_CONFIGS[base_name], device)
    base_model.eval()
    base_diffusion = diffusion_from_config(DIFFUSION_CONFIGS[base_name])

    print('[-] creating upsample model...')
    upsampler_model = model_from_config(MODEL_CONFIGS['upsample'], device)
    upsampler_model.eval()
    upsampler_diffusion = diffusion_from_config(DIFFUSION_CONFIGS['upsample'])

    print("[-] Loading pretrained models...")
    base_model.load_state_dict(load_checkpoint(base_name, device))
    upsampler_model.load_state_dict(load_checkpoint('upsample', device))

    sampler = PointCloudSampler(
        device=device,
        models=[base_model, upsampler_model],
        diffusions=[base_diffusion, upsampler_diffusion],
        num_points=[args.num_points, 4096-args.num_points], # points in cloud and missing ones for upsampling
        aux_channels=['R', 'G', 'B'],
        guidance_scale=[3.0, 3.0],
    )

    # Produce a sample from the model.
    samples = None
    for x in tqdm(sampler.sample_batch_progressive(batch_size=1, model_kwargs=dict(images=views))):
        samples = x

    del sampler
    return samples

# ----------------------------------------------

def get_colorless_cloud(cloud):
    """ 
        Add zero color to a point cloud 
        in:     Tensor(N, (x, y, z), K)
        out:    Tensor(N, (x, y, z, r, g, b), K)
    """
    f_cloud = cloud[0, :3, :]
    blacks = torch.zeros((3, f_cloud.shape[1])).to(device)
    return torch.cat((f_cloud, blacks), 0).unsqueeze(0)

def cloud_distance(cloud1, cloud2, metric=None):
    """ 
        Compute distance between 1d distributions of cloud p2 norms 
        in:     Tensor((x, y, z), K), Tensor((x, y, z), K) 
        out:    Float
    """
    D1 = torch.cdist(cloud1, cloud1, p=2)
    D2 = torch.cdist(cloud2, cloud2, p=2)
    X1 = [float(1/i.sum()) for i in D1]
    X2 = [float(1/i.sum()) for i in D2]
    
    if metric == "gaussian":
        return np.mean(((np.mean(X1) - np.mean(X2))**2).sum()) / (np.std(X1)**2 + np.std(X2)**2)
    else:
        return wasserstein_distance(X1, X2) * 1e5

def plot_distributions(cloud1, cloud2, labels=["cloud1", "cloud2"]):
    """ 
        Plot 1d distributions of cloud p2 norms 
        in:     Tensor((x, y, z), K), Tensor((x, y, z), K) 
    """
    D1 = torch.cdist(cloud1, cloud1, p=2)
    D2 = torch.cdist(cloud2, cloud2, p=2)
    
    s = pd.DataFrame({
        labels[0]: [float(1/i.sum()) for i in D1],
        labels[1]: [float(1/i.sum()) for i in D2],
      })
    s.plot.kde(bw_method=0.4, figsize=(24,8), title='poincloud pdf of different objects')
    plt.savefig(f"{labels[0]}_{labels[1]}.png")

def PIS(clf, cloud):
    """
        Compute P-IS score for a cloud
        in:     PointNetClassifier, Tensor(c, K)
        out:    Float
        https://github.com/halixness/point-e/blob/69e677d8ea47593c33fe2f52fd40e131054c9ce3/point_e/evals/fid_is.py#L73
    """
    cloud = cloud.permute(1,0).unsqueeze(0).cpu().numpy()
    _, preds = clf.features_and_preds(cloud)

    return np.exp(
      np.sum(
        preds[0] * ( np.log(preds[0]) - np.log(np.mean(preds[0])) )
      )
    )

# ----------------------- Params setting

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
ToPILImage = transforms.ToPILImage()

parser = argparse.ArgumentParser()
parser.add_argument("--input_views", type=str)
parser.add_argument("--ground_point_cloud", type=str)
parser.add_argument("--num_points", type=int, default=1024)
parser.add_argument("--num_views", type=int, default=4)
args = parser.parse_args()

# ----------------------- Data loading

ground_views = torch.load(args.input_views)
ground_views = [ToPILImage(v) for v in ground_views]

ground_point_cloud = torch.load(args.ground_point_cloud)
ground_point_cloud.shape


# Point cloud from single view
print("====== Single view ======")
pc_single = views_to_pointcloud(views = [ground_views[0]], n_views = 1)

torch.cuda.empty_cache()

# Point cloud from multi view
print("\n====== Multi view ======")
pc_multi = views_to_pointcloud(views = ground_views, n_views = len(ground_views))

# Colorless pointcloud (shape comparison only)
cless_pc_single = get_colorless_cloud(pc_single)
cless_pc_multi = get_colorless_cloud(pc_multi)
cless_pc_ground = get_colorless_cloud(ground_point_cloud.unsqueeze(0))

print("\n====== Point clouds divergences ======")
# Ground - Gen. single
print("[+] Ground truth - Single view divergence: \t\t{}".format(
    cloud_distance(cless_pc_ground[0].permute(1,0), cless_pc_single[0].permute(1,0))
))

plot_distributions(
    cless_pc_ground[0].permute(1,0),
    cless_pc_single[0].permute(1,0),
    ["ground_truth", "single view"]
)

# Ground - Gen. multi
print("[+] Ground truth - Multi view divergence: \t\t{}".format(
    cloud_distance(cless_pc_ground[0].permute(1,0), cless_pc_multi[0].permute(1,0))
))

plot_distributions(
    cless_pc_ground[0].permute(1,0),
    cless_pc_multi[0].permute(1,0),
    ["ground_truth", "multi view"]
)

# Gen. single - Gen. multi
print("[+] Single view - Multi view divergence: \t\t{}".format(
    cloud_distance(cless_pc_single[0].permute(1,0), cless_pc_multi[0].permute(1,0))
))

plot_distributions(
    cless_pc_single[0].permute(1,0), 
    cless_pc_multi[0].permute(1,0), 
    ["single view", "multi view"]
)

# Computing P-IS
print("\n====== P-IS scores ======")

clf = PointNetClassifier(devices=get_torch_devices(), cache_dir=None)

print(f"[+] Ground truth P-IS: \t\t{PIS(clf, ground_point_cloud)}")
print(f"[+] Single view P-IS: \t\t{PIS(clf, pc_single[0, :3])}")
print(f"[+] Multi view P-IS: \t\t{PIS(clf, pc_multi[0, :3])}")






