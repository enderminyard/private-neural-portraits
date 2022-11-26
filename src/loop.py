"""Module containing optimization loop."""
import lpips
import torch
from src.perceptual import create_tensor, maximize_loss

def loop(ref, pred, length):
    """Function containing optimization loop."""
    optimizer = torch.optim.Adam([pred,], lr=1e-3, betas=(0.9, 0.999))
    results = []

    for i in range(length):
        loss_fn = lpips.LPIPS(net='vgg')
        dist = loss_fn.forward(pred, ref)
        optimizer.zero_grad()
        dist.backward()
        optimizer.step()
        pred.data = torch.clamp(pred.data, -1, 1)
        results.append([i,dist, ref,pred.data])
        pred.data = torch.clamp(pred.data, -1, 1)

    return results

def natural_image_loop(ref_path, pred_path, length):
    """Maximize loss for natural image."""
    ref, pred = maximize_loss(ref_path, pred_path)
    result = loop(ref, pred, length)
    return result


def stylized_image_loop(ref_path, pred_path, length):
    """Minimize loss between natural and stylized image."""
    ref, pred = create_tensor(ref_path, pred_path)
    result = loop(ref, pred, length)
    return result
