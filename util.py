import torch
import torch.nn.functional as F
import torchvision

import numpy as np
import matplotlib.pyplot as plt
import random
from PIL import Image, ImageEnhance, ImageOps

from torch.utils.data import DataLoader, Dataset, Sampler


###############################################################################
# Adversarial Samples
###############################################################################
def adv_samples_PGD(model, x, y, epsilon, alpha, iterations = 10):

    x = x.clone().detach()
    adv_x = x.clone().detach()

    # random initialization
    adv_x = adv_x + torch.empty_like(adv_x).uniform_(-epsilon, epsilon)
    adv_x = torch.clamp(adv_x, min=0, max=1).detach()

    for i in range(iterations):
       
        adv_x.requires_grad = True
        
        outputs,_ = model(adv_x)

        # Calculate loss
        cost = F.cross_entropy(outputs, y)

        # Update adversarial images
        grad = torch.autograd.grad(
            cost, adv_x, retain_graph=False, create_graph=False
        )[0]

        adv_x = adv_x.detach() + alpha * torch.sign(grad)
        delta = torch.clamp(adv_x - x, min=-epsilon, max=epsilon)
        adv_x = torch.clamp(x + delta, min=0, max=1).detach()
        

    return adv_x


###############################################################################
# Dataloader
###############################################################################

def get_labels(dataset):
    return [label for _, label in dataset]

class BalancedSampler(Sampler):
    def __init__(self, dataset, num_classes, num_samples_per_class):
        self.labels = np.array(get_labels(dataset))
        self.num_classes = num_classes
        self.num_samples_per_class = num_samples_per_class
        self.class_indices = {i: np.where(self.labels == i)[0] for i in range(num_classes)}

    def __iter__(self):
        num_batches = len(self.labels) // (self.num_classes * self.num_samples_per_class)
        indices = []

        for _ in range(num_batches):
            batch = []
            for i in range(self.num_classes):
                batch.extend(np.random.choice(self.class_indices[i], self.num_samples_per_class, replace=False))
            np.random.shuffle(batch)
            indices.extend(batch)
        
        remaining_samples = len(self.labels) % (self.num_classes * self.num_samples_per_class)
        if remaining_samples > 0:
            batch = []
            for i in range(self.num_classes):
                n_samples = min(self.num_samples_per_class, len(self.class_indices[i]))
                batch.extend(np.random.choice(self.class_indices[i], n_samples, replace=False))
            np.random.shuffle(batch)
            indices.extend(batch[:remaining_samples])
        
        return iter(indices)

    def __len__(self):
        return len(self.labels)
    
    

###############################################################################
# Augmentation
###############################################################################
"""https://github.com/davidstutz/pytorch-adversarial-examples-training-articles/blob/master/005-adversarial-training/common/autoaugment.py"""
class SVHNPolicy(object):
    """ Randomly choose one of the best 25 Sub-policies on SVHN.

        Example:
        >>> policy = SVHNPolicy()
        >>> transformed = policy(image)

        Example as a PyTorch Transform:
        >>> transform=transforms.Compose([
        >>>     transforms.Resize(256),
        >>>     SVHNPolicy(),
        >>>     transforms.ToTensor()])
    """
    def __init__(self, fillcolor=(128, 128, 128)):
        self.policies = [
            SubPolicy(0.9, "shearX", 4, 0.2, "invert", 3, fillcolor),
            SubPolicy(0.9, "shearY", 8, 0.7, "invert", 5, fillcolor),
            SubPolicy(0.6, "equalize", 5, 0.6, "solarize", 6, fillcolor),
            SubPolicy(0.9, "invert", 3, 0.6, "equalize", 3, fillcolor),
            SubPolicy(0.6, "equalize", 1, 0.9, "rotate", 3, fillcolor),

            SubPolicy(0.9, "shearX", 4, 0.8, "autocontrast", 3, fillcolor),
            SubPolicy(0.9, "shearY", 8, 0.4, "invert", 5, fillcolor),
            SubPolicy(0.9, "shearY", 5, 0.2, "solarize", 6, fillcolor),
            SubPolicy(0.9, "invert", 6, 0.8, "autocontrast", 1, fillcolor),
            SubPolicy(0.6, "equalize", 3, 0.9, "rotate", 3, fillcolor),

            SubPolicy(0.9, "shearX", 4, 0.3, "solarize", 3, fillcolor),
            SubPolicy(0.8, "shearY", 8, 0.7, "invert", 4, fillcolor),
            SubPolicy(0.9, "equalize", 5, 0.6, "translateY", 6, fillcolor),
            SubPolicy(0.9, "invert", 4, 0.6, "equalize", 7, fillcolor),
            SubPolicy(0.3, "contrast", 3, 0.8, "rotate", 4, fillcolor),

            SubPolicy(0.8, "invert", 5, 0.0, "translateY", 2, fillcolor),
            SubPolicy(0.7, "shearY", 6, 0.4, "solarize", 8, fillcolor),
            SubPolicy(0.6, "invert", 4, 0.8, "rotate", 4, fillcolor),
            SubPolicy(0.3, "shearY", 7, 0.9, "translateX", 3, fillcolor),
            SubPolicy(0.1, "shearX", 6, 0.6, "invert", 5, fillcolor),

            SubPolicy(0.7, "solarize", 2, 0.6, "translateY", 7, fillcolor),
            SubPolicy(0.8, "shearY", 4, 0.8, "invert", 8, fillcolor),
            SubPolicy(0.7, "shearX", 9, 0.8, "translateY", 3, fillcolor),
            SubPolicy(0.8, "shearY", 5, 0.7, "autocontrast", 3, fillcolor),
            SubPolicy(0.7, "shearX", 2, 0.1, "invert", 5, fillcolor)
        ]


    def __call__(self, img):
        policy_idx = random.randint(0, len(self.policies) - 1)
        return self.policies[policy_idx](img)

    def __repr__(self):
        return "AutoAugment SVHN Policy"


class SubPolicy(object):
    def __init__(self, p1, operation1, magnitude_idx1, p2, operation2, magnitude_idx2, fillcolor=(128, 128, 128)):
        ranges = {
            "shearX": np.linspace(0, 0.3, 10),
            "shearY": np.linspace(0, 0.3, 10),
            "translateX": np.linspace(0, 150 / 331, 10),
            "translateY": np.linspace(0, 150 / 331, 10),
            "rotate": np.linspace(0, 30, 10),
            "color": np.linspace(0.0, 0.9, 10),
            "posterize": np.round(np.linspace(8, 4, 10), 0).astype(np.int64),
            "solarize": np.linspace(256, 0, 10),
            "contrast": np.linspace(0.0, 0.9, 10),
            "sharpness": np.linspace(0.0, 0.9, 10),
            "brightness": np.linspace(0.0, 0.9, 10),
            "autocontrast": [0] * 10,
            "equalize": [0] * 10,
            "invert": [0] * 10
        }

        # from https://stackoverflow.com/questions/5252170/specify-image-filling-color-when-rotating-in-python-with-pil-and-setting-expand
        def rotate_with_fill(img, magnitude):
            rot = img.convert("RGBA").rotate(magnitude)
            return Image.composite(rot, Image.new("RGBA", rot.size, (128,) * 4), rot).convert(img.mode)

        func = {
            "shearX": lambda img, magnitude: img.transform(
                img.size, Image.AFFINE, (1, magnitude * random.choice([-1, 1]), 0, 0, 1, 0),
                Image.BICUBIC, fillcolor=fillcolor),
            "shearY": lambda img, magnitude: img.transform(
                img.size, Image.AFFINE, (1, 0, 0, magnitude * random.choice([-1, 1]), 1, 0),
                Image.BICUBIC, fillcolor=fillcolor),
            "translateX": lambda img, magnitude: img.transform(
                img.size, Image.AFFINE, (1, 0, magnitude * img.size[0] * random.choice([-1, 1]), 0, 1, 0),
                fillcolor=fillcolor),
            "translateY": lambda img, magnitude: img.transform(
                img.size, Image.AFFINE, (1, 0, 0, 0, 1, magnitude * img.size[1] * random.choice([-1, 1])),
                fillcolor=fillcolor),
            "rotate": lambda img, magnitude: rotate_with_fill(img, magnitude),
            "color": lambda img, magnitude: ImageEnhance.Color(img).enhance(1 + magnitude * random.choice([-1, 1])),
            "posterize": lambda img, magnitude: ImageOps.posterize(img, magnitude),
            "solarize": lambda img, magnitude: ImageOps.solarize(img, magnitude),
            "contrast": lambda img, magnitude: ImageEnhance.Contrast(img).enhance(
                1 + magnitude * random.choice([-1, 1])),
            "sharpness": lambda img, magnitude: ImageEnhance.Sharpness(img).enhance(
                1 + magnitude * random.choice([-1, 1])),
            "brightness": lambda img, magnitude: ImageEnhance.Brightness(img).enhance(
                1 + magnitude * random.choice([-1, 1])),
            "autocontrast": lambda img, magnitude: ImageOps.autocontrast(img),
            "equalize": lambda img, magnitude: ImageOps.equalize(img),
            "invert": lambda img, magnitude: ImageOps.invert(img)
        }

        self.p1 = p1
        self.operation1 = func[operation1]
        self.magnitude1 = ranges[operation1][magnitude_idx1]
        self.p2 = p2
        self.operation2 = func[operation2]
        self.magnitude2 = ranges[operation2][magnitude_idx2]


    def __call__(self, img):
        if random.random() < self.p1: img = self.operation1(img, self.magnitude1)
        if random.random() < self.p2: img = self.operation2(img, self.magnitude2)
        return img


###############################################################################
# GMM
###############################################################################

def set_gmm_centers(dimension, num_gmm_components, gmm_distance=10):

    gmm_centers = []
    mu = np.zeros(dimension)
    mu[0] = gmm_distance
    for i in range(0, num_gmm_components):
        gmm_centers.append(np.roll(mu, i))
    gmm_std = 1
    gmm_centers = np.array(gmm_centers)
    gmm_centers = torch.tensor(gmm_centers).cuda().float()
    return gmm_centers, gmm_std

def draw_gmm_samples(num_samples, gmm_centers, gmm_std):

    num_gmm_centers, dimension = gmm_centers.shape

    samples = []
    components = []
    for _ in range(num_samples):
        component = np.random.choice(range(num_gmm_centers))

        component_mean = gmm_centers[component, :]
        component_cov = torch.eye(dimension) * gmm_std

        distribution = torch.distributions.multivariate_normal.MultivariateNormal(
            loc=component_mean.cuda(), covariance_matrix=component_cov.cuda()
        )

        sample = distribution.sample((1,))
        samples.append(sample)
        components.append(component)
    samples = torch.vstack(samples)

    return samples, components


###############################################################################
# Utility
###############################################################################


def accuracy(x, y):
    # x is the logits from the model, y is the true labels
    batch_size = y.size(0)
    
    pred = F.softmax(x, dim=1)
    pred = torch.argmax(pred, dim=1)
    correct = (pred == y).sum().item()
    
    accuracy = correct / batch_size
    
    return accuracy


def find_normalization(train_dl):
    all_images = []

    # Iterate through the DataLoader to collect pixel values
    for batch_images, _ in train_dl:
        all_images.append(batch_images.numpy())
    
    # Concatenate all batch images into a single numpy array
    all_images = np.concatenate(all_images, axis=0)
    
    # Calculate mean and std across all pixels
    mean = np.mean(all_images, axis=(0, 2, 3))
    std = np.std(all_images, axis=(0, 2, 3))
    
    print(mean)
    print(std)
    pass

def get_default_device():
    """Pick GPU if available, else CPU"""
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')
    
def to_device(data, device):
    """Move tensor(s) to chosen device"""
    if isinstance(data, (list,tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)


def append_to_dict(list_dict, value_dict):
    for key, value in value_dict.items():
        if key in list_dict:
            list_dict[key].append(value)

    return list_dict

def plot_acc_progress(epoch, accuracies):
    
    x = np.arange(epoch+1)
    
    for key, value in accuracies.items():
        
        acc = np.array(value)
        plt.plot(x, acc, label=key)

    plt.legend()
    plt.show()