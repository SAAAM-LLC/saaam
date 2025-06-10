"""
FastAPI server for the QuantumSacredTokenizer model
"""
import os
from typing import Dict, List, Optional, Union
from pydantic import BaseModel, Field
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

# Import your tokenizer
from tokenizer import QuantumSacredTokenizer, SacredConstants, EnhancedStabilityMetrics

# Initialize the FastAPI app
app = FastAPI(
    title="Quantum Sacred Tokenizer API",
    description="API for tokenizing, encoding, and analyzing text with quantum-sacred patterns",
    version="1.0.0"
)

# Add CORS middleware to allow cross-origin requests from your website
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Modify this in production to only include your website's domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize tokenizer - this will happen once when the app starts
TOKENIZER_PATH = "tokenizer/quantum_sacred_tokenizer.json"

# Check if a saved tokenizer exists, otherwise train a new one
if os.path.exists(TOKENIZER_PATH):
    tokenizer = QuantumSacredTokenizer.load(TOKENIZER_PATH)
    print(f"Loaded tokenizer with {tokenizer.get_vocab_size()} tokens")
else:
    # Create directories if they don't exist
    os.makedirs(os.path.dirname(TOKENIZER_PATH), exist_ok=True)
    
    # Create and train tokenizer with sample data
    tokenizer = QuantumSacredTokenizer()
    sample_texts = [
        "The quantum field harmonizes with the sacred patterns of consciousness",
        "Sacred geometry reveals hidden dimensions through resonant field patterns",
        "class QuantumField(object):\n    def __init__(self, dimensions=11):\n        self.dimensions = dimensions",
    ]
    tokenizer.train(sample_texts, min_freq=1, sacred_geometry_alignment=True)
    
    # Create some example entanglements
    entanglements = [
        ("quantum", "field", 0.95),
        ("sacred", "geometry", 0.9),
        ("resonance", "pattern", 0.85)
    ]
    tokenizer.batch_entangle(entanglements)
    
    # Save tokenizer for future use
    tokenizer.save(TOKENIZER_PATH)
    print(f"Created and saved new tokenizer with {tokenizer.get_vocab_size()} tokens")

# Define API request and response models
class TokenizeRequest(BaseModel):
    text: str = Field(..., description="Text to be tokenized")
    
class TokenizeResponse(BaseModel):
    tokens: List[str] = Field(..., description="List of tokens")

class EncodeRequest(BaseModel):
    text: str = Field(..., description="Text to be encoded")
    apply_sacred_geometry: bool = Field(True, description="Whether to apply sacred geometry transformations")
    
class EncodeResponse(BaseModel):
    token_ids: List[int] = Field(..., description="List of token IDs")
    
class EncodeWithScoresRequest(BaseModel):
    text: str = Field(..., description="Text to be encoded with scores")
    include_entanglement: bool = Field(True, description="Whether to include entanglement effects")
    
class TokenScore(BaseModel):
    token: str = Field(..., description="Token string")
    token_id: int = Field(..., description="Token ID")
    score: float = Field(..., description="Resonance score")
    
class EncodeWithScoresResponse(BaseModel):
    tokens: List[TokenScore] = Field(..., description="List of tokens with scores")

class DecodeRequest(BaseModel):
    token_ids: List[int] = Field(..., description="Token IDs to be decoded")
    
class DecodeResponse(BaseModel):
    text: str = Field(..., description="Decoded text")

class ResonanceAnalysisRequest(BaseModel):
    text: str = Field(..., description="Text to analyze for quantum resonance")
    
class ResonanceAnalysisResponse(BaseModel):
    metrics: Dict[str, float] = Field(..., description="Resonance metrics")

class EntanglementRequest(BaseModel):
    token_a: str = Field(..., description="First token to entangle")
    token_b: str = Field(..., description="Second token to entangle")
    strength: Optional[float] = Field(None, description="Entanglement strength (0-1)")
    
class EntanglementResponse(BaseModel):
    token_a: str = Field(..., description="First token")
    token_b: str = Field(..., description="Second token") 
    strength: float = Field(..., description="Resulting entanglement strength")
    
class LatentVectorRequest(BaseModel):
    token: str = Field(..., description="Token to get latent vector for")
    
class LatentVectorResponse(BaseModel):
    token: str = Field(..., description="Token")
    vector: List[float] = Field(..., description="Latent geometry vector")

# Define API endpoints
@app.get("/")
async def root():
    return {
        "message": "Welcome to the Quantum Sacred Tokenizer API",
        "description": "Use this API to tokenize, encode, and analyze text with quantum-sacred patterns",
        "documentation": "/docs",
        "version": "1.0.0"
    }

@app.post("/tokenize", response_model=TokenizeResponse)
async def tokenize_text(request: TokenizeRequest):
    """Tokenize text into tokens"""
    try:
        tokens = tokenizer.tokenize(request.text)
        return {"tokens": tokens}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error tokenizing text: {str(e)}")

@app.post("/encode", response_model=EncodeResponse)
async def encode_text(request: EncodeRequest):
    """Encode text into token IDs"""
    try:
        token_ids = tokenizer.encode(
            request.text, 
            apply_sacred_geometry=request.apply_sacred_geometry
        )
        return {"token_ids": token_ids}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error encoding text: {str(e)}")

@app.post("/encode_with_scores", response_model=EncodeWithScoresResponse)
async def encode_with_scores(request: EncodeWithScoresRequest):
    """Encode text into token IDs with quantum resonance scores"""
    try:
        scored_tokens = tokenizer.encode_with_scores(
            request.text,
            include_entanglement=request.include_entanglement
        )
        
        # Convert to response format
        tokens_with_scores = []
        for token_id, score in scored_tokens:
            token = tokenizer.inverse_vocab.get(token_id, "<UNK>")
            tokens_with_scores.append({
                "token": token,
                "token_id": token_id,
                "score": score
            })
        
        return {"tokens": tokens_with_scores}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error encoding text with scores: {str(e)}")

@app.post("/decode", response_model=DecodeResponse)
async def decode_tokens(request: DecodeRequest):
    """Decode token IDs back to text"""
    try:
        text = tokenizer.decode(request.token_ids)
        return {"text": text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error decoding tokens: {str(e)}")

@app.post("/analyze_resonance", response_model=ResonanceAnalysisResponse)
async def analyze_resonance(request: ResonanceAnalysisRequest):
    """Analyze quantum resonance patterns in text"""
    try:
        metrics = tokenizer.analyze_resonance(request.text)
        return {"metrics": metrics}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error analyzing resonance: {str(e)}")

@app.post("/entangle", response_model=EntanglementResponse)
async def entangle_tokens(request: EntanglementRequest):
    """Create quantum entanglement between two tokens"""
    try:
        strength = tokenizer.entangle(request.token_a, request.token_b, request.strength)
        return {
            "token_a": request.token_a,
            "token_b": request.token_b,
            "strength": strength
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error entangling tokens: {str(e)}")

@app.post("/latent_vector", response_model=LatentVectorResponse)
async def get_latent_vector(request: LatentVectorRequest):
    """Get latent geometry vector for a token"""
    try:
        vector = tokenizer.get_token_latent_vector(request.token)
        return {
            "token": request.token,
            "vector": vector.tolist()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting latent vector: {str(e)}")

@app.get("/metrics")
async def get_stability_metrics():
    """Get tokenizer stability metrics"""
    try:
        metrics = tokenizer.get_stability_metrics()
        # Convert to dictionary for JSON response
        metrics_dict = {
            "coherence": metrics.coherence,
            "field_stability": metrics.field_stability,
            "resonance_stability": metrics.resonance_stability,
            "phase_stability": metrics.phase_stability,
            "evolution_stability": metrics.evolution_stability,
            "dimension_bridging": metrics.dimension_bridging,
            "entanglement_factor": metrics.entanglement_factor,
            "neural_convergence": metrics.neural_convergence,
            "sacred_geometry_alignment": metrics.sacred_geometry_alignment,
            "torus_flow_integrity": metrics.torus_flow_integrity,
            "tree_of_life_balance": metrics.tree_of_life_balance,
            "frequency_harmony": metrics.frequency_harmony,
            "timestamp": metrics.timestamp
        }
        return metrics_dict
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting stability metrics: {str(e)}")

# Run the app
if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
