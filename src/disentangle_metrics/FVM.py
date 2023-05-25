"""
Based on "Disentangling by Factorising" (https://github.com/nmichlo/disent/blob/main/disent/metrics/_factor_vae.py).
"""
import torch
import time
import math
import numpy as np
from torchvision.transforms import Lambda, ToTensor
from tqdm.auto import tqdm

transform = ToTensor()
from PIL import Image

#commutative slef.ds 부분 참조해서 수정할 것 (
def FactorVAEMetric(dataset, model, batch_size, num_train, loss_fn):
    model.eval()
    with torch.no_grad():
        length = len(dataset)
        #randomly choose input data
        idx = np.random.choice(length-1, int(math.floor(0.1*length)), replace=False)
        #imgs, latents_classes = dataset.data[idx], dataset.latents_classes[:, 1:][idx] # 확인할 것
        #imgs = Image.fromarray(imgs, mode='L') # 수정 필요함.

        #imgs = transform(imgs).permute(1,2,0).unsqueeze(1) # (B, H, W) --> (B, C, H) --> (B, H, W) --> (B, C, H, W)
        #latents_classes = transform(latents_classes).squeeze(0) # 3D --> 2D [# of input, # of factor]

        #build dataloader for disentanglement
        #_dataset = TensorDataset(torch.Tensor(imgs), torch.Tensor(latents_classes))
        #dataloader = DataLoader(dataset=_dataset, shuffle=True, batch_size=batch_size)

        global_varaince = _compute_global_variance(dataset, model, batch_size, loss_fn)
        active_dims = _prune_dims(global_varaince)

        if not active_dims.any():
            return {
                "disentanglement_accuracy": 0.0,
                "num_active_dims": 0
            }

        votes = _generate_training_batch(dataset=dataset,
                                         model=model,
                                         batch_size=batch_size,
                                         num_points=num_train,
                                         variances=global_varaince,
                                         active_dims=active_dims,
                                         loss_fn=loss_fn)

        major_dim = torch.argmax(votes, dim=-2).detach().cpu().numpy()
        other_dim = torch.arange(votes.size(1)).detach().cpu().numpy()
        accuracy = torch.sum(votes[major_dim, other_dim]) * 1. / torch.sum(votes)
    return {
        "disentanglement_accuracy": accuracy.item(),
        "num_active_dims": active_dims.size(0)
    }

def _compute_global_variance(dataloader, model, batch_size, loss_fn):
    #compute variance of sampled dataset
    #var, total = 0.0, 0.0
    #iteration = tqdm(dataloader, desc='Iteration')
    data = dataloader.random_sampling_for_disen_global_variance(batch_size)
    model.eval()
    with torch.no_grad():
        #for _, (imgs, latent_classes) in enumerate(iteration):
            #set on GPU
        data = data.to(next(model.parameters()).device)
        z = model(data, loss_fn)[1][0]#.squeeze(-1).squeeze(-1)
        var = torch.var(z, dim=-2)
    return var

def _prune_dims(variances, threshold=0.0):
    """Mask for dimensions collapsed to the prior."""
    scale_z = torch.sqrt(variances)
    return scale_z >= threshold

def _generate_training_batch(dataset, model, batch_size, num_points, variances, active_dims, loss_fn):
    votes = torch.zeros(size=(dataset.factor_num, variances.size(-1))).to(next(model.parameters()).device)
    for _ in tqdm(range(num_points), desc='Iteration'):
        factor_index, argmin = _generate_training_sample(dataset, model, batch_size, variances, active_dims, loss_fn)
        votes[factor_index, argmin] += 1
    return votes

def _generate_training_sample(dataset, model, batch_size, variances, active_dims, loss_fn):

    length = len(dataset)
    idx = np.random.choice(length - 1, batch_size, replace=False)
    sampled_factors = dataset.latents_classes[idx]
    np.random.seed(seed=int(time.time()))
    #fixing one latent dimension value
    fixed_idx = np.random.randint(dataset.factor_num)
    sampled_factors[:, fixed_idx] = sampled_factors[0, fixed_idx]

    #find correspond idx of sampeld factors
    idx = find_index_from_factors(sampled_factors, dataset)
    #input of correspond to idx
    observation = torch.Tensor(dataset.data[idx]).to(next(model.parameters()).device)
    #latent vectors
    z = model(observation, loss_fn)[1][0]
    var = torch.var(z, dim=-2)
    dim =torch.argmin(var[active_dims] / variances[active_dims])
    return fixed_idx, dim

def find_index_from_factors(factors, dataset):
    factor_dict = {}
    sampled_idx = []
    for i, classes in enumerate(dataset.latents_classes):
        factor_dict[classes.tobytes()] = i
    for factor in factors:
        sampled_idx.append(factor_dict[factor.tobytes()])
    return sampled_idx