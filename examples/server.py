"""
Production-Grade Inference Server Example
-----------------------------------------
This example demonstrates how to wrap FluxInfer in a FastAPI service 
to serve high-throughput requests for a multimodal agent swarm.

Usage:
    python examples/server.py
"""

import time
import uuid
from typing import List, Optional
from pydantic import BaseModel
import logging

# Set up path to import flux_infer
import sys
import os
# Add the directory containing the 'flux_infer' package (which is inside the top-level flux_infer dir)
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "flux_infer"))

from flux_infer import FluxPipeline, InferenceConfig, OptimizationLevel, QuantizationMode

# Configure Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("FluxServer")

# --- Mock FastAPI Structure ---
# In a real app: from fastapi import FastAPI, HTTPException
class MockFastAPI:
    def post(self, path):
        def decorator(func):
            return func
        return decorator

app = MockFastAPI()
# ------------------------------

# Global Engine Instance
pipeline = None

class GenerationRequest(BaseModel):
    prompt: str
    max_tokens: int = 1024
    temperature: float = 0.7
    complexity_hint: float = 0.5 # Optional hint from the agent router

class GenerationResponse(BaseModel):
    id: str
    text: str
    latency_ms: float
    throughput: float
    usage: dict

def startup_event():
    """Initialize the engine on server startup."""
    global pipeline
    logger.info("Hydrating FluxInfer Engine...")
    
    # Production Configuration
    config = InferenceConfig(
        batch_size=128, # High batch size for server
        max_seq_len=4096,
        optimization_level=OptimizationLevel.O3,
        use_flash_attention=True,
        quantization_mode=QuantizationMode.Int4 # Maximize VRAM efficiency
    )
    
    pipeline = FluxPipeline("Llama-3-70B-Instruct-v2", config)
    pipeline.compile()
    logger.info("FluxInfer Engine Ready. Listening on port 8000...")

@app.post("/v1/completions")
def generate_completion(request: GenerationRequest):
    if not pipeline:
        raise RuntimeError("Engine not initialized")
    
    req_id = str(uuid.uuid4())
    logger.info(f"Processing Request {req_id} | Hint: {request.complexity_hint}")
    
    # Perform Inference
    result = pipeline.generate(request.prompt, complexity_score=request.complexity_hint)
    
    # Format Response
    response = GenerationResponse(
        id=req_id,
        text=result["text"],
        latency_ms=result["metrics"]["latency_ms"],
        throughput=result["metrics"]["throughput_tokens_per_sec"],
        usage={
            "prompt_tokens": len(request.prompt.split()),
            "completion_tokens": result["metrics"]["tokens_generated"],
            "experts_active": result["metrics"]["experts_used"]
        }
    )
    
    return response

if __name__ == "__main__":
    # Simulate Server Lifecycle
    startup_event()
    
    print("\n--- Simulating Incoming Traffic ---\n")
    
    requests = [
        GenerationRequest(prompt="Summarize the history of Rome", complexity_hint=0.4),
        GenerationRequest(prompt="Write a complex Rust macro for serialization", complexity_hint=0.95),
        GenerationRequest(prompt="Hi, how are you?", complexity_hint=0.1),
    ]
    
    for req in requests:
        resp = generate_completion(req)
        print(f"[200 OK] ID: {resp.id[:8]}... | Latency: {resp.latency_ms:.1f}ms | T/s: {resp.throughput:.1f}")
        time.sleep(0.5)
