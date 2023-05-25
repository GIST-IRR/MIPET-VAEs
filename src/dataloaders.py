import os
import numpy as np
import PIL
from PIL import Image
import scipy.io as sio
from six.moves import range
from sklearn.utils import extmath
import torch
from torch.utils.data import TensorDataset, DataLoader
import h5py
import torch
import torch.utils.data as data
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data.dataloader import default_collate
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
import pdb
import tarfile

class DisenDataLoader(data.Dataset):
    def __init__(self,
                 path,
                 shuffle_dataset=True,
                 random_seed=42,
                 split_ratio = 0.2):
        self.path = path
        self.shuffle_dataset = shuffle_dataset
        self.random_seed = random_seed
        self.split_ratio = split_ratio

        self.data, self.latents_values, self.latents_classes = None, None, None
        self.train_idxs, self.test_idxs = None, None
        self.factor_num = None

    def __call__(self, *args, **kwargs):
        raise NotImplementedError("Build call function")

    def __getitem__(self, item):
        raise NotImplementedError("Build getitem function")

    def dataset_sample_batch(self, num_samples: int, mode: str, replace: bool):
        raise NotImplementedError("Build dataset_sample_batch function")

    def __len__(self):
        return len(self.data)


class dstripeDataLoader(DisenDataLoader):
    def __init__(self,
                 path,
                 shuffle_dataset=True,
                 random_seed=42,
                 split_ratio = 0.0):
        super(dstripeDataLoader, self).__init__(path,
                                                shuffle_dataset,
                                                random_seed,
                                                split_ratio)

    def __call__(self, *args, **kwargs):
        np_load_old = np.load
        np.load = lambda *a, **k: np_load_old(*a, allow_pickle=True, **k)
        np.load = np_load_old
        dataset_zip = np.load(self.path, allow_pickle=True, encoding="bytes")

        self.data = np.expand_dims(dataset_zip['imgs'], axis=1) # (# of datasets, C, H, W)
        self.latents_values = dataset_zip['latents_values']
        self.latents_classes = dataset_zip['latents_classes'][:, 1:] # (# of datasets, 6 - 1)
        self.factor_num = self.latents_values[:, 1:].shape[-1]

        assert self.factor_num == 5

        dataset_size = len(self.data)
        idxs = list(range(dataset_size))
        split = int(np.floor(self.split_ratio * dataset_size))

        if self.shuffle_dataset:
            np.random.seed(self.random_seed)
            np.random.shuffle(idxs)

        self.train_idxs, self.valid_idxs = idxs[split:], idxs[:split]
        train_dataset = TensorDataset(torch.Tensor(self.data[self.train_idxs]),
                                      torch.Tensor(self.latents_classes[self.train_idxs]))
        valid_dataset = TensorDataset(torch.Tensor(self.data[self.valid_idxs]),
                                      torch.Tensor(self.latents_classes[self.valid_idxs]))

        return (train_dataset, valid_dataset) if self.split_ratio > 0.0 else (train_dataset, None)

    def random_sampling_for_disen_global_variance(self, batch_size, replace=False):
        g = np.random.Generator(np.random.PCG64(seed=np.random.randint(0, 2 ** 32)))
        indices = g.choice(len(self.data), batch_size, replace=replace)
        return torch.Tensor(self.data[indices])

    def sampling_factors_and_img(self, batch_size, num_train):
        dataset_size = len(self.data)
        idxs = list(range(dataset_size))
        factors, imgs = [], []
        np.random.seed(self.random_seed)
        for _ in range(num_train):
            np.random.shuffle(idxs)
            factor_idxs = idxs[:batch_size]
            factors.append(torch.Tensor(self.latents_classes[factor_idxs])) #(B, num factors -1)
            imgs.append(torch.Tensor(self.data[factor_idxs])) # (B, C, H, W)

        return torch.stack(imgs, dim=0), torch.stack(factors, dim=0) # (num_train, B, C, H, W), (num_train, B, -1)

    def __getitem__(self, idx):
        data = self.data[idx]
        #latents = self.latents_values[idx]
        classes = self.latents_classes[idx]
        return data, classes

    def __len__(self):
        return len(self.data)

    def img_from_idx(self, idx):
        return self.data[idx]

    def factor_from_idx(self, idx):
        return self.latents_classes[idx]

    def idx_from_factor(self, factor):
        base = np.concatenate(self.latents_values[1:][::-1].cumprod()[::-1][1:], np.array([1,]))
        return np.dot(factor, base).astype(int)

    def dataset_sample_batch(self, num_samples, mode, replace=False):
        g = np.random.Generator(np.random.PCG64(seed=np.random.randint(0, 2 ** 32)))
        indices = g.choice(len(self), num_samples, replace=replace)
        return self.dataset_batch_from_indices(indices, mode=mode)

    def dataset_batch_from_indices(self, indices, mode):
        return default_collate([self.dataset_get(idx, mode=mode) for idx in indices])

    def dataset_get(self, idx, mode: str):

        try:
            idx = int(idx)
        except:
            raise TypeError(f'Indices must be integer-like ({type(idx)}): {idx}')

class _3DshapeDataLoader(DisenDataLoader):
    def __init__(self,
                 path,
                 shuffle_dataset=True,
                 random_seed=42,
                 split_ratio = 0.0
                 ):
        super(_3DshapeDataLoader, self).__init__(path,
                                                 shuffle_dataset=True,
                                                 random_seed=42,
                                                 split_ratio = 0.0)

    def __call__(self, *args, **kwargs):
        _FACTOR_TO_COLUMN_INDEX = {'floor_hue': 0, 'wall_hue': 1, 'object_hue': 2, 'scale': 3, 'shape': 4,
                                   'orientation': 5}
        _FACTOR_TO_ALLOWED_VALUES = {'floor_hue': list(range(10)), 'wall_hue': list(range(10)),
                                     'object_hue': list(range(10)), 'scale': list(range(8)), 'shape': list(range(4)),
                                     'orientation': list(range(15))}

        _NUM_VALUES_PER_FACTOR = {'floor_hue': 10, 'wall_hue': 10, 'object_hue': 10,
                                  'scale': 8, 'shape': 4, 'orientation': 15}

        file = h5py.File(self.path, 'r')
        #self.latents_values = dataset_zip['latents_values']
        self.data = np.array(file['images'][:]).transpose(0,3,1,2) / 255.0 #[480000, 64, 64, 3] --> [480000, 3, 64, 64]
        #self.latents_classes = np.floor(np.array(file['labels'][:]) * np.array([10, 10, 10, 100, 1, 1] - np.array([0,0,0,75,0,0]))).astype('int32')  # (480000, 6)
        self.latents_classes = np.array(file['labels'][:])
        self.latents_classes = self.latents_classes.astype(float)
        # convert float to int scales
        self.latents_classes[:, 0] = self.latents_classes[:, 0] * 10  # [0, 1, ..., 9]
        self.latents_classes[:, 1] = self.latents_classes[:, 1] * 10  # [0, 1, ..., 9]
        self.latents_classes[:, 2] = self.latents_classes[:, 2] * 10  # [0, 1, ..., 9]
        self.latents_classes[:, 3] = np.round(self.latents_classes[:, 3], 2)  # first round, since values are very precise floats
        remap = {0.75: 0.0, 0.82: 1.0, 0.89: 2.0, 0.96: 3.0, 1.04: 4.0, 1.11: 5.0, 1.18: 6.0, 1.25: 7.0}
        label_3 = np.copy(self.latents_classes[:, 3])
        for k, v in remap.items():
            label_3[self.latents_classes[:, 3] == k] = v
        self.latents_classes[:, 3] = label_3
        # shape is already on int scale   # # [0, 1, ..., 3]
        self.latents_classes[:, 5] = np.round(self.latents_classes[:, 5], 2)  # first round, since values are very precise floats
        remap = {-30.: 0, -25.71: 1, -21.43: 2, -17.14: 3, -12.86: 4, -8.57: 5, -4.29: 6, 0.: 7, 4.29: 8, 8.57: 9,
                 12.86: 10, 17.14: 11, 21.43: 12, 25.71: 13, 30.: 14}
        label_5 = np.copy(self.latents_classes[:, 5])
        for k, v in remap.items():
            label_5[self.latents_classes[:, 5] == k] = v
        self.latents_classes[:, 5] = label_5  # [0, 1, ..., 15]
        # make self.latents_classes an int, because
        # since 3 in self.latents_classes[:, 0] is actually 3.0000000000004, even though not correctly displayed
        self.latents_classes = self.latents_classes.astype(int)

        self.factor_num = self.latents_classes.shape[-1]
        #pdb.set_trace()
        dataset_size = len(file['images'])
        idxs = list(range(dataset_size))
        split = int(np.floor(self.split_ratio * dataset_size))

        if self.shuffle_dataset:
            np.random.seed(self.random_seed)
            np.random.shuffle(idxs)
        self.train_idxs, self.valid_idxs = idxs[split:], idxs[:split]

        train_dataset = TensorDataset(torch.Tensor(self.data[self.train_idxs, :, :, :]),
                                      torch.Tensor(self.latents_classes[self.train_idxs, :]))
        valid_dataset = TensorDataset(torch.Tensor(self.data[self.valid_idxs, :, :, :]),
                                      torch.Tensor(self.latents_classes[self.valid_idxs, :]))

        return (train_dataset, valid_dataset) if self.split_ratio > 0.0 else (train_dataset, None)

    def __getitem__(self, idx):
        data = self.data[idx]
        classes = self.latents_classes[idx]
        return data, classes

    def random_sampling_for_disen_global_variance(self, batch_size, replace=False):
        g = np.random.Generator(np.random.PCG64(seed=np.random.randint(0, 2 ** 32)))
        indices = g.choice(self.data.shape[0], batch_size, replace=replace)
        return torch.Tensor(self.data[indices,:,:,:])#.permute(0,3,1,2)

    def sampling_factors_and_img(self, batch_size, num_train):
        dataset_size = len(self.data)
        idxs = list(range(dataset_size))
        factors, imgs = [], []
        np.random.seed(self.random_seed)
        for _ in range(num_train):
            np.random.shuffle(idxs)
            factor_idxs = idxs[:batch_size]
            factors.append(torch.Tensor(self.latents_classes[factor_idxs, :])) #(B, num factors)
            imgs.append(torch.Tensor(self.data[factor_idxs, :, :, :])) # (B, C, H, W)

        return torch.stack(imgs, dim=0), torch.stack(factors, dim=0) # (num_train, B, C, H, W), (num_train, B, -1)

    def img_from_idx(self, idx):
        return self.data[idx]

    def factor_from_idx(self, idx):
        return self.latents_classes[idx]

class StateSpaceAtomIndex(object):
    """Index mapping from features to positions of state space atoms."""

    def __init__(self, factor_sizes, features):
        """Creates the StateSpaceAtomIndex.
    Args:
      factor_sizes: List of integers with the number of distinct values for each
        of the factors.
      features: Numpy matrix where each row contains a different factor
        configuration. The matrix needs to cover the whole state space.
    """
        self.factor_sizes = factor_sizes
        num_total_atoms = np.prod(self.factor_sizes)
        self.factor_bases = num_total_atoms / np.cumprod(self.factor_sizes)
        feature_state_space_index = self._features_to_state_space_index(features)
        if np.unique(feature_state_space_index).size != num_total_atoms:
            raise ValueError("Features matrix does not cover the whole state space.")
        lookup_table = np.zeros(num_total_atoms, dtype=np.int64)
        lookup_table[feature_state_space_index] = np.arange(num_total_atoms)
        self.state_space_to_save_space_index = lookup_table

    def features_to_index(self, features):
        """Returns the indices in the input space for given factor configurations.
    Args:
      features: Numpy matrix where each row contains a different factor
        configuration for which the indices in the input space should be
        returned.
    """
        state_space_index = self._features_to_state_space_index(features)
        return self.state_space_to_save_space_index[state_space_index]

    def _features_to_state_space_index(self, features):
        """Returns the indices in the atom space for given factor configurations.
    Args:
      features: Numpy matrix where each row contains a different factor
        configuration for which the indices in the atom space should be
        returned.
    """
        if (np.any(features > np.expand_dims(self.factor_sizes, 0)) or
                np.any(features < 0)):
            raise ValueError("Feature indices have to be within [0, factor_size-1]!")
        return np.array(np.dot(features, self.factor_bases), dtype=np.int64)


class _3DcarDataLoader(DisenDataLoader):
    def __init__(self,
                 path,
                 shuffle_dataset=True,
                 random_seed=42,
                 split_ratio=0.0):
        super(_3DcarDataLoader, self).__init__(path,
                                               shuffle_dataset=True,
                                               random_seed=42,
                                               split_ratio=0.0)
    def __call__(self, *args, **kwargs):
        self.factor_sizes = [4, 24, 183]
        features = extmath.cartesian([np.array(list(range(i))) for i in self.factor_sizes])
        self.latent_factor_indices = [0, 1, 2]
        self.factor_num = features.shape[1]
        self.index = StateSpaceAtomIndex(self.factor_sizes, features)
        self.data_shape = [64, 64, 3]
        self.data, self.latents_classes = self._load_data() # numpy array (17568, 64, 64, 3)
        self.data = self.data.transpose(0,3,1,2) # numpy array (17568, 3, 64, 64)

        dataset_size = self.data.shape[0]
        idxs = list(range(dataset_size))
        split = int(np.floor(self.split_ratio * dataset_size))

        if self.shuffle_dataset:
            np.random.seed(self.random_seed)
            np.random.shuffle(idxs)
        self.train_idxs, self.valid_idxs = idxs[split:], idxs[:split]

        train_dataset = TensorDataset(torch.Tensor(self.data[self.train_idxs, :, :, :]),
                                      torch.Tensor(self.latents_classes[self.train_idxs, :]))
        valid_dataset = TensorDataset(torch.Tensor(self.data[self.valid_idxs, :, :, :]),
                                      torch.Tensor(self.latents_classes[self.valid_idxs, :]))

        return (train_dataset, valid_dataset) if self.split_ratio > 0.0 else (train_dataset, None)

    def __getitem__(self, idx):
        data = self.data[idx]
        classes = self.latents_classes[idx]
        return data, classes

    def random_sampling_for_disen_global_variance(self, batch_size, replace=False):
        g = np.random.Generator(np.random.PCG64(seed=np.random.randint(0, 2 ** 32)))
        indices = g.choice(self.data.shape[0], batch_size, replace=replace)
        return torch.Tensor(self.data[indices,:,:,:])#.permute(0,3,1,2)

    def sampling_factors_and_img(self, batch_size, num_train):
        dataset_size = len(self.data)
        idxs = list(range(dataset_size))
        factors, imgs = [], []
        np.random.seed(self.random_seed)
        for _ in range(num_train):
            np.random.shuffle(idxs)
            factor_idxs = idxs[:batch_size]
            factors.append(torch.Tensor(self.latents_classes[factor_idxs, :])) #(B, num factors)
            imgs.append(torch.Tensor(self.data[factor_idxs, :, :, :])) # (B, C, H, W)

        return torch.stack(imgs, dim=0), torch.stack(factors, dim=0) # (num_train, B, C, H, W), (num_train, B, -1)

    def img_from_idx(self, idx):
        return self.data[idx]

    def factor_from_idx(self, idx):
        return self.latents_classes[idx]

    def _load_data(self):
        dataset = np.zeros((24 * 4 * 183, 64, 64, 3))
        factors = np.zeros((24 * 4 * 183, 3))
        all_files = [x for x in os.listdir(self.path) if ".mat" in x]
        for i, filename in enumerate(all_files):
            data_mesh = self._load_mesh(filename)
            factor1 = np.array(list(range(4)))
            factor2 = np.array(list(range(24)))
            all_factors = np.transpose([
                np.tile(factor1, len(factor2)),
                np.repeat(factor2, len(factor1)),
                np.tile(i,
                        len(factor1) * len(factor2))
            ])
            indexes = self.index.features_to_index(all_factors)
            dataset[indexes] = data_mesh
            factors[indexes] = all_factors
        return dataset, factors

    def _load_mesh(self, filename):
        """Parses a single source file and rescales contained images."""
        with open(os.path.join(self.path, filename), "rb") as f:
            mesh = np.einsum("abcde->deabc", sio.loadmat(f)["im"])
        flattened_mesh = mesh.reshape((-1,) + mesh.shape[2:])
        rescaled_mesh = np.zeros((flattened_mesh.shape[0], 64, 64, 3))
        for i in range(flattened_mesh.shape[0]):
            pic = Image.fromarray(flattened_mesh[i, :, :, :])
            pic.thumbnail((64, 64), Image.ANTIALIAS)
            rescaled_mesh[i, :, :, :] = np.array(pic)
        return rescaled_mesh * 1. / 255
