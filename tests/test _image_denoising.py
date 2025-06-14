import unittest

from src.image_denoising import load_image, denoise_image, show_image, denoise_image_vectorized


class TestImageDenoising(unittest.TestCase):
    @staticmethod
    def load_image():
        return load_image('noisy_img.jpg')

    def test_denoise_image(self):
        Y = self.load_image()
        X = denoise_image(Y, 0.3)
        show_image(X, "cleaned")

    def test_denoise_image_vectorized(self):
        Y = self.load_image()
        X = denoise_image_vectorized(Y, 0.3)
        show_image(X, "cleaned_vectorized")


if __name__ == '__main__':
    unittest.main()