#!/usr/bin/env python3
"""
Advanced Augmentation Utilities for Fine-tuning
Implements Mixup, CutMix, and other advanced augmentations
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional


def mixup_data(
    x: torch.Tensor, 
    y: torch.Tensor, 
    alpha: float = 0.2,
    use_cuda: bool = True
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """
    Mixup augmentation - mixes two samples and their labels
    
    Args:
        x: Input batch tensor
        y: Target labels (can be one-hot or indices)
        alpha: Beta distribution parameter
        use_cuda: Whether to use CUDA
        
    Returns:
        mixed_x: Mixed input batch
        y_a: Original labels
        y_b: Shuffled labels
        lam: Mixing coefficient
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    
    return mixed_x, y_a, y_b, lam


def mixup_criterion(
    criterion,
    pred: torch.Tensor,
    y_a: torch.Tensor,
    y_b: torch.Tensor,
    lam: float
) -> torch.Tensor:
    """
    Compute mixup loss
    
    Args:
        criterion: Loss function
        pred: Model predictions
        y_a: First set of labels
        y_b: Second set of labels
        lam: Mixing coefficient
        
    Returns:
        Mixed loss
    """
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def rand_bbox(
    size: Tuple[int, ...],
    lam: float
) -> Tuple[int, int, int, int]:
    """
    Generate random bounding box for CutMix
    
    Args:
        size: Image size (B, C, H, W)
        lam: Lambda value
        
    Returns:
        Bounding box coordinates (x1, y1, x2, y2)
    """
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    # Uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2


def cutmix_data(
    x: torch.Tensor,
    y: torch.Tensor,
    alpha: float = 1.0,
    use_cuda: bool = True
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """
    CutMix augmentation - cuts and pastes patches between samples
    
    Args:
        x: Input batch tensor
        y: Target labels
        alpha: Beta distribution parameter
        use_cuda: Whether to use CUDA
        
    Returns:
        mixed_x: Mixed input batch
        y_a: Original labels
        y_b: Shuffled labels
        lam: Adjusted mixing coefficient
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    bbx1, bby1, bbx2, bby2 = rand_bbox(x.size(), lam)
    mixed_x = x.clone()
    mixed_x[:, :, bbx1:bbx2, bby1:bby2] = x[index, :, bbx1:bbx2, bby1:bby2]
    
    # Adjust lambda to exactly match pixel ratio
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (x.size()[-1] * x.size()[-2]))
    
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


class MixupCutmixAugmentation:
    """
    Combined Mixup/CutMix augmentation with configurable probability
    """
    
    def __init__(
        self,
        mixup_alpha: float = 0.2,
        cutmix_alpha: float = 1.0,
        mixup_prob: float = 0.5,
        cutmix_prob: float = 0.5,
        use_cuda: bool = True
    ):
        """
        Args:
            mixup_alpha: Beta distribution parameter for mixup
            cutmix_alpha: Beta distribution parameter for cutmix
            mixup_prob: Probability of applying mixup
            cutmix_prob: Probability of applying cutmix
            use_cuda: Whether to use CUDA
        """
        self.mixup_alpha = mixup_alpha
        self.cutmix_alpha = cutmix_alpha
        self.mixup_prob = mixup_prob
        self.cutmix_prob = cutmix_prob
        self.use_cuda = use_cuda
        
        # Normalize probabilities
        total_prob = mixup_prob + cutmix_prob
        if total_prob > 0:
            self.mixup_prob = mixup_prob / total_prob
            self.cutmix_prob = cutmix_prob / total_prob
    
    def __call__(
        self,
        x: torch.Tensor,
        y: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float, str]:
        """
        Apply either Mixup or CutMix based on probability
        
        Returns:
            mixed_x: Augmented input
            y_a: First set of labels
            y_b: Second set of labels
            lam: Mixing coefficient
            method: Which method was used ('mixup', 'cutmix', or 'none')
        """
        r = np.random.rand(1)
        
        if r < self.mixup_prob:
            mixed_x, y_a, y_b, lam = mixup_data(
                x, y, self.mixup_alpha, self.use_cuda
            )
            method = 'mixup'
        elif r < self.mixup_prob + self.cutmix_prob:
            mixed_x, y_a, y_b, lam = cutmix_data(
                x, y, self.cutmix_alpha, self.use_cuda
            )
            method = 'cutmix'
        else:
            mixed_x, y_a, y_b, lam = x, y, y, 1.0
            method = 'none'
        
        return mixed_x, y_a, y_b, lam, method


class StainAugmentation:
    """
    Stain augmentation specific to histopathology images
    Simulates variations in H&E staining
    """
    
    def __init__(
        self,
        hue_shift: float = 0.05,
        saturation_shift: float = 0.1,
        value_shift: float = 0.05
    ):
        self.hue_shift = hue_shift
        self.saturation_shift = saturation_shift
        self.value_shift = value_shift
    
    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        """
        Apply stain augmentation
        
        Args:
            image: Input tensor in RGB format [C, H, W] or [B, C, H, W]
            
        Returns:
            Augmented image
        """
        # Convert to HSV for stain manipulation
        # This is a simplified version - in practice, you might use
        # more sophisticated stain separation methods
        
        # Random shifts
        h_shift = (torch.rand(1) - 0.5) * 2 * self.hue_shift
        s_shift = (torch.rand(1) - 0.5) * 2 * self.saturation_shift
        v_shift = (torch.rand(1) - 0.5) * 2 * self.value_shift
        
        # Apply shifts (simplified - actual implementation would convert to HSV)
        augmented = image.clone()
        
        # Simulate hematoxylin (purple/blue) variation
        augmented[0] = torch.clamp(augmented[0] + h_shift, 0, 1)
        
        # Simulate eosin (pink) variation
        augmented[1] = torch.clamp(augmented[1] + s_shift, 0, 1)
        
        # Overall brightness
        augmented = torch.clamp(augmented + v_shift, 0, 1)
        
        return augmented


def nucleus_augmentation(
    image: torch.Tensor,
    nucleus_scale: float = 0.1,
    nucleus_prob: float = 0.3
) -> torch.Tensor:
    """
    Augment nuclear patterns in histopathology images
    
    Args:
        image: Input image tensor
        nucleus_scale: Scale of nuclear perturbation
        nucleus_prob: Probability of applying augmentation
        
    Returns:
        Augmented image
    """
    if torch.rand(1) > nucleus_prob:
        return image
    
    # Create random nucleus-like patterns
    B, C, H, W = image.shape
    
    # Generate random circular kernels (nucleus-like)
    kernel_size = 5
    kernel = torch.zeros(kernel_size, kernel_size)
    center = kernel_size // 2
    
    for i in range(kernel_size):
        for j in range(kernel_size):
            dist = np.sqrt((i - center)**2 + (j - center)**2)
            if dist <= center:
                kernel[i, j] = 1 - dist / center
    
    kernel = kernel.unsqueeze(0).unsqueeze(0)
    kernel = kernel.repeat(C, 1, 1, 1)
    
    # Apply convolution to simulate nuclear patterns
    noise = torch.randn_like(image) * nucleus_scale
    filtered_noise = F.conv2d(
        noise,
        kernel,
        padding=kernel_size//2,
        groups=C
    )
    
    # Add to original image
    augmented = torch.clamp(image + filtered_noise, 0, 1)
    
    return augmented


class TestTimeAugmentation:
    """
    Test-time augmentation for improved predictions
    """
    
    def __init__(
        self,
        n_augmentations: int = 5,
        include_flips: bool = True,
        include_rotations: bool = True,
        include_color: bool = False
    ):
        self.n_augmentations = n_augmentations
        self.include_flips = include_flips
        self.include_rotations = include_rotations
        self.include_color = include_color
    
    def __call__(
        self,
        image: torch.Tensor,
        model: torch.nn.Module
    ) -> torch.Tensor:
        """
        Apply test-time augmentation and average predictions
        
        Args:
            image: Input image [B, C, H, W]
            model: Model to use for predictions
            
        Returns:
            Averaged predictions
        """
        predictions = []
        
        # Original prediction
        with torch.no_grad():
            pred = model(image)
            predictions.append(pred)
        
        # Augmented predictions
        for i in range(self.n_augmentations - 1):
            aug_image = image.clone()
            
            # Random horizontal flip
            if self.include_flips and torch.rand(1) > 0.5:
                aug_image = torch.flip(aug_image, dims=[3])
            
            # Random vertical flip
            if self.include_flips and torch.rand(1) > 0.5:
                aug_image = torch.flip(aug_image, dims=[2])
            
            # Random rotation (90 degree increments)
            if self.include_rotations:
                k = torch.randint(0, 4, (1,)).item()
                if k > 0:
                    aug_image = torch.rot90(aug_image, k, dims=[2, 3])
            
            # Color augmentation (slight)
            if self.include_color:
                color_shift = (torch.rand(3, 1, 1) - 0.5) * 0.1
                aug_image[:, :3] = torch.clamp(
                    aug_image[:, :3] + color_shift.to(aug_image.device),
                    0, 1
                )
            
            # Get prediction
            with torch.no_grad():
                pred = model(aug_image)
                predictions.append(pred)
        
        # Average predictions
        predictions = torch.stack(predictions)
        return predictions.mean(dim=0)


# Example usage
if __name__ == "__main__":
    # Test mixup
    x = torch.randn(4, 3, 224, 224)
    y = torch.tensor([0, 1, 2, 1])
    
    mixed_x, y_a, y_b, lam = mixup_data(x, y, alpha=0.2, use_cuda=False)
    print(f"Mixup - Lambda: {lam:.3f}")
    print(f"Mixed shape: {mixed_x.shape}")
    
    # Test cutmix
    mixed_x, y_a, y_b, lam = cutmix_data(x, y, alpha=1.0, use_cuda=False)
    print(f"\nCutMix - Lambda: {lam:.3f}")
    print(f"Mixed shape: {mixed_x.shape}")
    
    # Test combined augmentation
    augmenter = MixupCutmixAugmentation(use_cuda=False)
    mixed_x, y_a, y_b, lam, method = augmenter(x, y)
    print(f"\nCombined - Method: {method}, Lambda: {lam:.3f}") 