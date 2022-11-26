"""Module testing image input tensor format."""
import lpips
from lpips import im2tensor

def get_tensor(first_img_path, second_img_path):
    """Check whether LPIPS loss accepts the tensor format."""
    first_tensor = im2tensor(first_img_path)
    second_tensor = im2tensor(second_img_path)
    loss_fn_vgg = lpips.LPIPS(net='vgg')
    # Images tested must be of same size.
    assert loss_fn_vgg(first_tensor, second_tensor)

if __name__ == '__main__':
    get_tensor('./tests/test.jpg','./tests/edited.jpg')
    print("Test passed.")
