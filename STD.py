import cv2
import numpy as np
from ultralytics import YOLO


class OBBPredictor:
    def __init__(self, model_path):
        """
        Initialize the OBBPredictor with the YOLO model.
        :param model_path: Path to the YOLO model weights.
        """
        self.model = YOLO(model_path)

    @staticmethod
    def image_show(results):
        """
        Display the results using the YOLO built-in function.
        :param results: YOLO prediction results.
        """
        results[0].show()

    @staticmethod
    def order_points(pts):
        """
        Order points in clockwise order starting from top-left.
        :param pts: Array of points.
        :return: Ordered points.
        """
        rect = np.zeros((4, 2), dtype=np.float32)
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]  # top-left
        rect[2] = pts[np.argmax(s)]  # bottom-right

        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]  # top-right
        rect[3] = pts[np.argmax(diff)]  # bottom-left
        return rect

    @staticmethod
    def crop_obb_region(image, points):
        """
        Crop region defined by four corner points using perspective transform.
        :param image: Original image.
        :param points: Four corner points of the region.
        :return: Cropped image.
        """
        ordered_pts = OBBPredictor.order_points(points).astype(np.float32)

        width = int(max(
            np.linalg.norm(ordered_pts[0] - ordered_pts[1]),
            np.linalg.norm(ordered_pts[2] - ordered_pts[3])
        ))
        height = int(max(
            np.linalg.norm(ordered_pts[1] - ordered_pts[2]),
            np.linalg.norm(ordered_pts[3] - ordered_pts[0])
        ))

        dst_pts = np.array([
            [0, 0],
            [width - 1, 0],
            [width - 1, height - 1],
            [0, height - 1]
        ], dtype=np.float32)

        M = cv2.getPerspectiveTransform(ordered_pts, dst_pts)
        warped = cv2.warpPerspective(image, M, (width, height))
        return warped

    def extract_crop(self, results):
        """
        Extract the cropped region from the YOLO results.
        :param results: YOLO prediction results.
        :return: Cropped image or None if no detection.
        """
        if results is not None:
            for result in results:
                points = result.obb.xyxyxyxy[0].cpu().numpy()
                cropped = self.crop_obb_region(result.orig_img, points)
                return cropped
        else:
            print("No Detection")
            return None

    def predict(self, image_path):
        """
        Perform prediction on the given image and return the cropped region.
        :param image_path: Path to the input image.
        :return: Cropped image or None if no detection.
        """
        results = self.model(image_path)
        self.image_show(results)
        return self.extract_crop(results)


if __name__ == "__main__":
    # Test the OBBPredictor independently
    model_path = "D:\\Ehtesham_ku500944\\Implementation_Code\\Inference_code\\code\\Detection\\runs\\obb\\obb_640_2\\weights\\best.pt"
    image_path = "D:\\Ehtesham_ku500944\\Sanad_Images\\OCR\\Clear_images\\Text_Detection\\OBB\\data\\images\\train\\image_0035.png"

    # Initialize the OBB predictor
    obb_predictor = OBBPredictor(model_path)

    # Perform prediction
    cropped_image = obb_predictor.predict(image_path)

    # Display the cropped image
    if cropped_image is not None:
        cv2.imshow("Cropped Text", cropped_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("No cropped region detected.")