"""Module testing usage of pretrained RetinaFace on image containing a face."""

from retinaface.pre_trained_models import get_model
from lpips import im2tensor
from lpips import tensor2im

def test_model(img):
    """Function testing usage of pretrained RetinaFace."""
    model = get_model("resnet50_2020-07-20", max_size=2048)
    model.eval()
    annotation = model.predict_jsons(img)
    return annotation

if __name__ == '__main__':
    tensor = im2tensor('./tests/test.jpg')
    result = test_model(tensor2im(tensor))
    print("Test passed.")
