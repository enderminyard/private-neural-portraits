"""Module computing perceptual loss."""
import lpips
from torch.autograd import Variable


def compute_lpips(ref, pred):
    """Function computing perceptual loss."""
    loss_fn_vgg = lpips.LPIPS(net='vgg')
    result = loss_fn_vgg(ref,pred)
    return result

def create_tensor(ref_path, pred_path):
    """Function converting image paths to tensors of the image at that path."""
    ref = lpips.im2tensor(lpips.load_image(ref_path))
    pred = Variable(lpips.im2tensor(lpips.load_image(pred_path)), requires_grad=True)
    return (ref,pred)

def maximize_loss(ref_path, pred_path):
    """Creating tensors for loss maximization."""
    ref = lpips.im2tensor(lpips.load_image(ref_path))
    pred = Variable(lpips.im2tensor(lpips.load_image(pred_path)), requires_grad=True)
    return (-ref,pred)
