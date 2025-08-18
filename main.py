from STD import OBBPredictor
from STR import TextRecognizer
import cv2

class OCRInference:
    def __init__(self, obb_model_path, ocr_ckpt_path, device='cpu'):
        """
        Initialize the OCRInference pipeline with OBB predictor and OCR recognizer.
        :param obb_model_path: Path to the YOLO model weights for OBB prediction.
        :param ocr_ckpt_path: Path to the OCR model checkpoint.
        :param device: Device to run the OCR model ('cpu' or 'cuda').
        """
        self.obb_predictor = OBBPredictor(obb_model_path)
        self.text_recognizer = TextRecognizer(ocr_ckpt_path, device)

    def run(self, image_path):
        """
        Run the OCR pipeline: OBB prediction followed by text recognition.
        :param image_path: Path to the input image.
        :return: Recognized text or None if no text is detected.
        """
        # Step 1: Predict and crop text region using OBB predictor
        cropped_image = self.obb_predictor.predict(image_path)

        if cropped_image is None:
            print("No text region detected.")
            return None

        # Save the cropped image temporarily for OCR processing
        temp_cropped_path = "temp_cropped_image.jpg"
        cv2.imwrite(temp_cropped_path, cropped_image)

        # Step 2: Recognize text using OCR
        recognized_text = self.text_recognizer.run(temp_cropped_path)

        # Step 3: Write the recognized text on the original image
        original_image = cv2.imread(image_path)
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 5
        font_color = (0, 255, 0)  # Green
        thickness = 3
        position = (150, 150)  # Top-left corner

        if recognized_text:
            cv2.putText(original_image, recognized_text, position, font, font_scale, font_color, thickness, cv2.LINE_AA)
           # Resize the image to 640x640 for display
        resized_image = cv2.resize(original_image, (640, 640))
        # Display the image with the recognized text
        cv2.imshow("OCR Result", resized_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        return recognized_text


if __name__ == "__main__":
    # Paths to the models and input image
    detection_model_path = "Pretrained_model\\STD.pt"
    recognition_ckpt_path = "Pretrained_model\\STR.ckpt"
    image_path = "demo_images\\1.png"

    # Initialize the OCR pipeline
    ocr_pipeline = OCRInference(detection_model_path, recognition_ckpt_path)

    # Run the OCR pipeline
    recognized_text = ocr_pipeline.run(image_path)
    if recognized_text:
        print(f"Recognized Text: {recognized_text}")
    else:
        print("No text recognized.")