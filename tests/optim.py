"""Module testing execution of optimization loop;"""
from src.loop import natural_image_loop


def test_loop():
    """Function testing execution of optimization loop;"""
    assert natural_image_loop('./tests/test.jpg','./tests/edited.jpg',1)

if __name__ == '__main__':
    test_loop()
    print("Test passed.")
