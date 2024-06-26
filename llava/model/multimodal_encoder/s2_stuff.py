import math
import torch
import torch.nn.functional as F
from einops import rearrange


def debug_device(tensor, name):
    print(f"{name} is on {tensor.device} and is of type {tensor.dtype}")

def split_chessboard(x, num_split):
    """
        x: b * c * h * w
        Deividing x into num_split**2 sub-squares, and concatenate all the sub-squares on the batch dimension
    """
    B, C, H, W = x.shape
    assert H % num_split == 0 and W % num_split == 0
    h, w = H // num_split, W // num_split
    x_split = torch.cat([x[:, :, i*h:(i+1)*h, j*w:(j+1)*w] for i in range(num_split) for j in range(num_split)], dim=0)
    return x_split

def merge_chessboard(x, num_split):
    """
        x: b * c * h * w
        Assuming x contains num_split**2 sub-squares concatenated along batch dimension, merge the sub-squares back to the original whole square.
        (inverse of split_chessboard)
    """
    B, C, H, W = x.shape
    assert B % (num_split**2) == 0
    b = B // (num_split**2)
    x_merge = torch.cat([torch.cat([x[(i*num_split + j)*b:(i*num_split + j + 1)*b] for j in range(num_split)], dim=-1)
                         for i in range(num_split)], dim=-2)
    return x_merge

def batched_forward(model, x, batch_size=-1):
    if batch_size == -1:
        return model(x)
    else:
        x_batched = x.split(batch_size)
        outs = [model(x) for x in x_batched]
        return torch.cat(outs, dim=0)


def forward(model, input, scales=None, img_sizes=None, max_split_size=None, resize_output_to_idx=0, num_prefix_token=0,
            output_shape='bnc', split_forward=False):
    assert input.dim() == 4, "Input image must be in the shape of BxCxHxW."
    assert input.shape[2] == input.shape[3], "Currently only square images are supported."
    assert output_shape in ['bnc',
                            'bchw'], "Output shape should be either BxNxC (e.g., ViT) or BxCxHxW (e.g., ConvNet)."
    assert output_shape == 'bnc' or num_prefix_token == 0, "For ConvNet there shouldn't be any prefix token."

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    input = input.to(device).to(torch.float16)

    debug_device(input, "Initial input")

    b, c, input_size, _ = input.shape

    assert scales is not None or img_sizes is not None, "Please assign either scales or img_sizes."
    img_sizes = img_sizes or [int(input_size * scale) for scale in scales]

    max_split_size = max_split_size or input_size
    num_splits = [math.ceil(size / max_split_size) for size in img_sizes]
    input_multiscale = []

    for size, num_split in zip(img_sizes, num_splits):
        x = F.interpolate(input, size=(size, size), mode='bicubic').to(input.dtype)
        debug_device(x, f"Interpolated input for size {size}")

        x = split_chessboard(x, num_split=num_split).to(device)
        debug_device(x, f"Split input for size {size}")

        print(f"Image size for scale {size}: {x.shape}")  # Add this line to print the image size

        input_multiscale.append(x)

    for idx, item in enumerate(input_multiscale):
        debug_device(item, f"input_multiscale[{idx}]")

    outs_multiscale = []
    for x in input_multiscale:
        x = x.to(device)
        if split_forward:
            out = batched_forward(model, x, b)
        else:
            import time

            start_time = time.time()
            out = model(x)
            end_time = time.time()
            execution_time = end_time - start_time
            print("Execution time:", execution_time, "seconds")

        out = out.to(device)  # Ensure output is on the right device
        debug_device(out, "Model output before concatenating scales")

        outs_multiscale.append(out)

    if num_prefix_token > 0:
        outs_prefix_multiscale = [out[:, :num_prefix_token] for out in outs_multiscale]
        outs_multiscale = [out[:, num_prefix_token:] for out in outs_multiscale]
        for idx, out in enumerate(outs_prefix_multiscale):
            debug_device(out, f"Prefix tokens at scale {idx}")

    if output_shape == 'bnc':
        outs_multiscale = [
            rearrange(out, 'b (h w) c -> b c h w', h=int(out.shape[1] ** 0.5), w=int(out.shape[1] ** 0.5))
            for out in outs_multiscale]
        for idx, out in enumerate(outs_multiscale):
            debug_device(out, f"Rearranged output at scale {idx}")

    outs_multiscale = [merge_chessboard(out, num_split=num_split).to(device) for num_split, out in
                       zip(num_splits, outs_multiscale)]
    for idx, out in enumerate(outs_multiscale):
        debug_device(out, f"Merged chessboard at scale {idx}")

    output_size = outs_multiscale[resize_output_to_idx].shape[-2]
    out = torch.cat(
        [F.interpolate(outs_multiscale[i].to(torch.float16), size=output_size, mode='area').to(outs_multiscale[i].dtype)
         for i in range(len(outs_multiscale))], dim=1).to(device)
    debug_device(out, "Final concatenated output")

    if output_shape == 'bnc':
        out = rearrange(out, 'b c h w -> b (h w) c').to(device)
        debug_device(out, "Final rearranged output")

    if num_prefix_token > 0:
        outs_prefix_multiscale = [torch.stack(out.split(b, dim=0), dim=0).mean(dim=0).to(device) for out in
                                  outs_prefix_multiscale]
        out_prefix_multiscale = torch.cat(outs_prefix_multiscale, dim=-1).to(device)
        out = torch.cat([out_prefix_multiscale, out], dim=1).to(device)
        debug_device(out, "Final output with prefix tokens")

    return out
