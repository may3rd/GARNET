import unittest
from unittest.mock import MagicMock, patch
import numpy as np
import os
import sys

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from garnet.azure_inference.custom_vision_detector import CustomVisionSAHIDetector
from garnet.azure_inference.config import CustomVisionConfig
from garnet.azure_inference.preprocessing import preprocess_image
from garnet.azure_inference.postprocessing import convert_outputs_to_detections

class TestCustomVisionDetector(unittest.TestCase):
    
    def setUp(self):
        self.config = CustomVisionConfig(
            model_path="dummy.onnx",
            class_names=["class1", "class2"],
            input_size=(640, 640),
            confidence_threshold=0.5
        )
        
    @patch('onnxruntime.InferenceSession')
    def test_initialization(self, mock_session):
        # Setup mock
        mock_sess_instance = MagicMock()
        mock_session.return_value = mock_sess_instance
        
        input_mock = MagicMock()
        input_mock.name = 'data'
        mock_sess_instance.get_inputs.return_value = [input_mock]
        
        o1 = MagicMock(); o1.name = 'detected_boxes'
        o2 = MagicMock(); o2.name = 'detected_scores'
        o3 = MagicMock(); o3.name = 'detected_classes'
        mock_sess_instance.get_outputs.return_value = [o1, o2, o3]
        
        detector = CustomVisionSAHIDetector(self.config)
        self.assertIsNotNone(detector.session)
        self.assertEqual(detector.input_name, 'data')
        self.assertEqual(len(detector.output_names), 3)

    def test_preprocessing(self):
        # Create a dummy image 100x100 RGB
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        target_size = (50, 50)
        
        processed, r_w, r_h = preprocess_image(image, target_size)
        
        self.assertEqual(processed.shape, (1, 3, 50, 50))
        self.assertEqual(processed.dtype, np.float32)
        # Check ratio: original / target
        self.assertEqual(r_w, 2.0)
        self.assertEqual(r_h, 2.0)
        
    def test_postprocessing(self):
        # Dummy outputs
        # Boxes: 1 detection, normalized [0.1, 0.1, 0.2, 0.2]
        boxes = np.array([[[0.1, 0.1, 0.2, 0.2]]], dtype=np.float32)
        scores = np.array([[0.9]], dtype=np.float32)
        classes = np.array([[0]], dtype=np.int32)
        
        outputs = [boxes, scores, classes]
        output_names = ['detected_boxes', 'detected_scores', 'detected_classes']
        
        detections = convert_outputs_to_detections(
            outputs=outputs,
            output_names=output_names,
            image_width=100,
            image_height=100,
            ratio_w=1.0,
            ratio_h=1.0,
            confidence_threshold=0.5,
            class_names=["class1"]
        )
        
        self.assertEqual(len(detections), 1)
        det = detections[0]
        self.assertEqual(det.category.name, "class1")
        self.assertAlmostEqual(det.score.value, 0.9, places=5)
        # Boxes scaled by 100
        # 0.1 * 100 = 10
        self.assertEqual(det.bbox.minx, 10.0)
        self.assertEqual(det.bbox.maxx, 20.0)
        
    @patch('onnxruntime.InferenceSession')
    def test_inference(self, mock_session):
        # Setup mock
        mock_sess_instance = MagicMock()
        mock_session.return_value = mock_sess_instance
        
        # Inputs/Outputs info
        input_mock = MagicMock()
        input_mock.name = 'data'
        mock_sess_instance.get_inputs.return_value = [input_mock]
        
        o1 = MagicMock(); o1.name = 'detected_boxes'
        o2 = MagicMock(); o2.name = 'detected_scores'
        o3 = MagicMock(); o3.name = 'detected_classes'
        mock_sess_instance.get_outputs.return_value = [o1, o2, o3]
        
        # Run return values
        # 1 detection
        boxes = np.array([[[0.1, 0.1, 0.2, 0.2]]], dtype=np.float32)
        scores = np.array([[0.9]], dtype=np.float32)
        classes = np.array([[0]], dtype=np.int32)
        mock_sess_instance.run.return_value = [boxes, scores, classes]
        
        detector = CustomVisionSAHIDetector(self.config)
        
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        result = detector.perform_inference(image)
        
        self.assertEqual(len(result.object_prediction_list), 1)
        
if __name__ == '__main__':
    unittest.main()
