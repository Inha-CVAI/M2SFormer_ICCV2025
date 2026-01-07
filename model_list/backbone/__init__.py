import torch


def dct_low_frequency_mask(height: int, width: int, alpha: int = 200) -> torch.Tensor:
    """
    Creates a low-frequency DCT mask in the form of a right isosceles triangle
    from the top-left corner. Positions where row + col <= alpha are set to 1,
    otherwise 0.

    Args:
        height (int): The height of the mask (e.g., image height).
        width (int): The width of the mask (e.g., image width).
        alpha (int, optional): Cutoff parameter defining the triangular region.
                               Defaults to 200.

    Returns:
        torch.Tensor: A (height x width) mask with dtype float32.
    """
    # Create coordinate grids for row and column indices
    rows = torch.arange(height).unsqueeze(1).expand(-1, width)  # Shape [H, W]
    cols = torch.arange(width).unsqueeze(0).expand(height, -1)  # Shape [H, W]

    # Sum of row index + column index
    position_sum = rows + cols

    # Generate mask: 1 if within the alpha cut-off, 0 otherwise
    mask = (position_sum <= alpha).float()
    return mask


if __name__ == "__main__":
    # Example usage
    H, W = 256, 256
    alpha_val = 56  # default hyperparameter
    low_freq_mask = dct_low_frequency_mask(H, W, alpha_val)

    print("Mask shape:", low_freq_mask.shape)
    print("Mask sum:", low_freq_mask.sum().item())
    import matplotlib.pyplot as plt
    plt.imshow(low_freq_mask, cmap='gray')
    plt.axis('off'); plt.show()