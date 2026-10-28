import platform
import tempfile
import unittest
from pathlib import Path

import coremltools as ct
import numpy as np

from spike.ane_matmul import (
    Shape,
    inspect_compute_plan,
    quantize_per_output_channel,
    reference_matmul,
    relative_l2,
    convert_model,
)


class QuantizationTests(unittest.TestCase):
    def test_reference_uses_dequantized_weights(self):
        weights = np.array([[0.01, -0.02, 0.03], [0.20, -0.10, 0.05]], dtype=np.float32)
        qweight, scales, dequantized = quantize_per_output_channel(weights)
        activations = np.array([[1.0, 2.0, -1.0]], dtype=np.float16)
        expected = np.matmul(activations.astype(np.float32), dequantized.T)
        actual = reference_matmul(activations, qweight, scales)
        np.testing.assert_allclose(actual, expected, rtol=0.0, atol=0.0)

    def test_zero_row_remains_finite(self):
        qweight, scales, dequantized = quantize_per_output_channel(np.zeros((2, 4), dtype=np.float32))
        self.assertEqual(qweight.dtype, np.int8)
        self.assertTrue(np.all(np.isfinite(scales)))
        np.testing.assert_array_equal(dequantized, np.zeros((2, 4), dtype=np.float32))

    def test_mismatched_k_is_rejected(self):
        with self.assertRaises(ValueError):
            reference_matmul(np.zeros((1, 3), dtype=np.float16), np.zeros((2, 4), dtype=np.int8), np.ones(2, dtype=np.float16))


@unittest.skipUnless(platform.system() == "Darwin" and platform.machine() == "arm64", "requires Apple Silicon CoreML")
class CoreMLTests(unittest.TestCase):
    def test_tiny_prediction_and_public_compute_plan(self):
        rng = np.random.default_rng(19)
        shape = Shape(2, 8, 6)
        weights = rng.standard_normal((shape.n, shape.k), dtype=np.float32) * 0.02
        qweight, scales, _ = quantize_per_output_channel(weights)
        activations = rng.standard_normal((shape.m, shape.k), dtype=np.float32).astype(np.float16)
        expected = reference_matmul(activations, qweight, scales)

        with tempfile.TemporaryDirectory() as temp_dir:
            package_path = Path(temp_dir) / "tiny.mlpackage"
            convert_model(shape, qweight, scales, package_path)
            model = ct.models.MLModel(str(package_path), compute_units=ct.ComputeUnit.CPU_AND_NE)
            output = np.asarray(next(iter(model.predict({"x": activations}).values())))
            self.assertEqual(output.shape, (shape.m, shape.n))
            self.assertLessEqual(relative_l2(output, expected), 0.02)
            operations = inspect_compute_plan(model.get_compiled_model_path(), ct.ComputeUnit.CPU_AND_NE)
            self.assertTrue(any("matmul" in operation["operation"] for operation in operations))


if __name__ == "__main__":
    unittest.main()
