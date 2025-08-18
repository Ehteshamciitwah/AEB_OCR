import torch
from PIL import Image
from strhub.data.module import SceneTextDataModule
from strhub.models.utils import load_from_checkpoint


class TextRecognizer:
    def __init__(self, ckpt_path, device='cpu'):
        """
        Initialize the TextRecognizer with the OCR model.
        :param ckpt_path: Path to the OCR model checkpoint.
        :param device: Device to run the OCR model ('cpu' or 'cuda').
        """
        self.ckpt_path = ckpt_path
        self.device = device
        self.parseq = None
        self.img_transform = None

    def load_model(self):
        """
        Load the OCR model from the checkpoint.
        """
        self.parseq = load_from_checkpoint(self.ckpt_path)
        self.parseq.eval().to(self.device)
        self.img_transform = SceneTextDataModule.get_transform(self.parseq.hparams.img_size)

    def preprocess_image(self, img_path):
        """
        Preprocess the input image for OCR.
        :param img_path: Path to the input image.
        :return: Preprocessed image tensor.
        """
        img = Image.open(img_path).convert('RGB')
        return self.img_transform(img).unsqueeze(0)

    def predict(self, img):
        """
        Perform text recognition on the input image tensor.
        :param img: Preprocessed image tensor.
        :return: Recognized text.
        """
        logits = self.parseq(img)
        pred = logits.softmax(-1)
        label, confidence = self.parseq.tokenizer.decode(pred)
        return label[0]

    def run(self, img_path):
        """
        Run the OCR pipeline on the input image.
        :param img_path: Path to the input image.
        :return: Recognized text.
        """
        self.load_model()
        img = self.preprocess_image(img_path)
        label = self.predict(img)
        return label


if __name__ == "__main__":
    # Test the TextRecognizer independently
    ckpt_path = r'D:\Ehtesham_ku500944\Codes\OCR\parseq\outputs\parseq-tiny\2025-04-28_17-09-24\checkpoints\last.ckpt'
    img_path = r'D:\Ehtesham_ku500944\Codes\OCR\YOLOv8-CRNN-Scene-Text-Recognition\demo\image_TR.jpg'

    # Initialize the TextRecognizer
    text_recognizer = TextRecognizer(ckpt_path)

    # Perform text recognition
    recognized_text = text_recognizer.run(img_path)

    # Print the recognized text
    if recognized_text:
        print(f"Recognized Text: {recognized_text}")
    else:
        print("No text recognized.")