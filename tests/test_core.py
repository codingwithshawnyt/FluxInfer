import unittest
import sys
import os

# Ensure we can import the package
# We need to add the directory containing the 'flux_infer' package to the path.
# Structure is: project_root/flux_infer/flux_infer/__init__.py
# So we add project_root/flux_infer to sys.path
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "flux_infer"))

from flux_infer import FluxPipeline, InferenceConfig, OptimizationLevel, QuantizationMode

class TestFluxInferCore(unittest.TestCase):
    
    def test_config_initialization(self):
        """Verify that InferenceConfig correctly applies defaults and overrides."""
        # Test Default
        config = InferenceConfig(batch_size=1, max_seq_len=1024)
        self.assertEqual(config.batch_size, 1)
        self.assertEqual(config.max_seq_len, 1024)
        self.assertEqual(config.optimization_level, OptimizationLevel.O2)
        self.assertEqual(config.quantization_mode, QuantizationMode.F16) # Default

        # Test Overrides
        config_opt = InferenceConfig(
            batch_size=32, 
            max_seq_len=2048, 
            optimization_level=OptimizationLevel.O3,
            quantization_mode=QuantizationMode.Int4
        )
        self.assertEqual(config_opt.optimization_level, OptimizationLevel.O3)
        self.assertEqual(config_opt.quantization_mode, QuantizationMode.Int4)

    def test_engine_optimization_pipeline(self):
        """Verify that the engine 'compiles' the optimization graph correctly."""
        config = InferenceConfig(
            batch_size=8,
            max_seq_len=1024,
            optimization_level=OptimizationLevel.O3,
            use_flash_attention=True,
            quantization_mode=QuantizationMode.Int8
        )
        pipeline = FluxPipeline("Test-Model-7B", config)
        
        # Capture stdout to verify logs
        # In a real test we might mock stdout, but for now we just check internal state
        pipeline.compile()
        self.assertTrue(pipeline._compiled)

    def test_inference_simulation(self):
        """Test the end-to-end generation simulation."""
        config = InferenceConfig(batch_size=1, max_seq_len=128)
        pipeline = FluxPipeline("Test-Model-7B", config)
        
        prompt = "Hello world"
        result = pipeline.generate(prompt, complexity_score=0.2)
        
        self.assertIn("text", result)
        self.assertIn("metrics", result)
        self.assertGreater(result["metrics"]["latency_ms"], 0)
        self.assertGreater(result["metrics"]["tokens_generated"], 0)

    def test_moe_routing_logic(self):
        """Verify the deterministic behavior of the mocked MoE router."""
        from flux_infer import MoERouter
        
        router = MoERouter(num_experts=8, active_experts=2)
        
        # Test High Complexity (Expects heavy experts)
        experts_high = router.route(0.95)
        self.assertTrue(0 in experts_high and 7 in experts_high)
        
        # Test Low Complexity (Expects fast path)
        experts_low = router.route(0.1)
        self.assertEqual(experts_low, [0])

if __name__ == "__main__":
    unittest.main()
