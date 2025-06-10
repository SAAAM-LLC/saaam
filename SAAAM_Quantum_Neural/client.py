"""
Client for interacting with the Quantum Sacred Tokenizer API
"""
import requests
import json
from typing import Dict, List, Any, Optional, Union
import time
import numpy as np

class QuantumTokenizerClient:
    """Client for the Quantum Sacred Tokenizer API"""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        """
        Initialize the client
        
        Args:
            base_url: Base URL of the API server (default: http://localhost:8000)
        """
        self.base_url = base_url.rstrip('/')
        
        # Test connection
        try:
            response = requests.get(f"{self.base_url}/")
            response.raise_for_status()
            info = response.json()
            print(f"Connected to {info.get('description', 'Quantum Tokenizer API')}")
            print(f"API version: {info.get('version', 'unknown')}")
        except requests.exceptions.RequestException as e:
            print(f"Warning: Could not connect to API at {self.base_url}")
            print(f"Error: {str(e)}")
            print("API calls will still be attempted but may fail")
    
    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize text into tokens
        
        Args:
            text: Text to tokenize
            
        Returns:
            List of tokens
        """
        response = requests.post(
            f"{self.base_url}/tokenize",
            json={"text": text}
        )
        response.raise_for_status()
        return response.json()["tokens"]
    
    def encode(self, text: str, apply_sacred_geometry: bool = True) -> List[int]:
        """
        Encode text into token IDs
        
        Args:
            text: Text to encode
            apply_sacred_geometry: Whether to apply sacred geometry transformations
            
        Returns:
            List of token IDs
        """
        response = requests.post(
            f"{self.base_url}/encode",
            json={
                "text": text,
                "apply_sacred_geometry": apply_sacred_geometry
            }
        )
        response.raise_for_status()
        return response.json()["token_ids"]
    
    def encode_with_scores(self, text: str, include_entanglement: bool = True) -> List[Dict[str, Any]]:
        """
        Encode text with resonance scores
        
        Args:
            text: Text to encode
            include_entanglement: Whether to include entanglement effects
            
        Returns:
            List of tokens with resonance scores
        """
        response = requests.post(
            f"{self.base_url}/encode_with_scores",
            json={
                "text": text,
                "include_entanglement": include_entanglement
            }
        )
        response.raise_for_status()
        return response.json()["tokens"]
    
    def decode(self, token_ids: List[int]) -> str:
        """
        Decode token IDs back to text
        
        Args:
            token_ids: List of token IDs to decode
            
        Returns:
            Decoded text
        """
        response = requests.post(
            f"{self.base_url}/decode",
            json={"token_ids": token_ids}
        )
        response.raise_for_status()
        return response.json()["text"]
    
    def analyze_resonance(self, text: str) -> Dict[str, float]:
        """
        Analyze quantum resonance patterns in text
        
        Args:
            text: Text to analyze
            
        Returns:
            Resonance metrics
        """
        response = requests.post(
            f"{self.base_url}/analyze_resonance",
            json={"text": text}
        )
        response.raise_for_status()
        return response.json()["metrics"]
    
    def entangle(self, token_a: str, token_b: str, strength: Optional[float] = None) -> Dict[str, Any]:
        """
        Create quantum entanglement between two tokens
        
        Args:
            token_a: First token
            token_b: Second token
            strength: Entanglement strength (optional)
            
        Returns:
            Entanglement details
        """
        response = requests.post(
            f"{self.base_url}/entangle",
            json={
                "token_a": token_a,
                "token_b": token_b,
                "strength": strength
            }
        )
        response.raise_for_status()
        return response.json()
    
    def batch_entangle(self, token_pairs: List[Union[tuple, list]]) -> int:
        """
        Create multiple token entanglements in one go
        
        Args:
            token_pairs: List of token pairs 
                Each pair can be a tuple of (token_a, token_b) or (token_a, token_b, strength)
                
        Returns:
            Number of successful entanglements
        """
        count = 0
        for pair in token_pairs:
            try:
                if len(pair) == 2:
                    token_a, token_b = pair
                    self.entangle(token_a, token_b)
                elif len(pair) == 3:
                    token_a, token_b, strength = pair
                    self.entangle(token_a, token_b, strength)
                else:
                    raise ValueError("Token pairs must be (token_a, token_b) or (token_a, token_b, strength)")
                count += 1
            except Exception as e:
                print(f"Error entangling {pair}: {str(e)}")
        return count
    
    def get_latent_vector(self, token: str) -> Dict[str, Any]:
        """
        Get latent geometry vector for a token
        
        Args:
            token: Token to get vector for
            
        Returns:
            Token and its latent vector
        """
        response = requests.post(
            f"{self.base_url}/latent_vector",
            json={"token": token}
        )
        response.raise_for_status()
        result = response.json()
        
        # Convert vector to numpy array for easier manipulation
        if "vector" in result:
            result["vector_np"] = np.array(result["vector"])
        
        return result
    
    def get_stability_metrics(self) -> Dict[str, float]:
        """
        Get tokenizer stability metrics
        
        Returns:
            Stability metrics
        """
        response = requests.get(f"{self.base_url}/metrics")
        response.raise_for_status()
        return response.json()
    
    def batch_process(self, texts: List[str], process_type: str = "tokenize") -> List[Any]:
        """
        Process multiple texts in a batch
        
        Args:
            texts: List of texts to process
            process_type: Type of processing to perform 
                Options: "tokenize", "encode", "encode_with_scores", "analyze_resonance"
                
        Returns:
            List of results corresponding to each input text
        """
        results = []
        
        for text in texts:
            if process_type == "tokenize":
                result = self.tokenize(text)
            elif process_type == "encode":
                result = self.encode(text)
            elif process_type == "encode_with_scores":
                result = self.encode_with_scores(text)
            elif process_type == "analyze_resonance":
                result = self.analyze_resonance(text)
            else:
                raise ValueError(f"Unknown process_type: {process_type}")
            
            results.append(result)
            
            # Small delay to avoid overwhelming the server
            time.sleep(0.1)
            
        return results
    
    def compare_texts(self, text1: str, text2: str) -> Dict[str, Any]:
        """
        Compare quantum resonance between two texts
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            Comparison metrics
        """
        # Get resonance metrics for both texts
        metrics1 = self.analyze_resonance(text1)
        metrics2 = self.analyze_resonance(text2)
        
        # Calculate differences
        differences = {}
        for key in metrics1:
            if key in metrics2:
                differences[key] = metrics2[key] - metrics1[key]
        
        # Get token vectors for comparison
        tokens1 = self.tokenize(text1)
        tokens2 = self.tokenize(text2)
        
        # Find common tokens
        common_tokens = set(tokens1).intersection(set(tokens2))
        
        return {
            "text1_metrics": metrics1,
            "text2_metrics": metrics2,
            "differences": differences,
            "common_tokens": list(common_tokens),
            "common_token_count": len(common_tokens),
            "text1_token_count": len(tokens1),
            "text2_token_count": len(tokens2),
            "text1_unique_tokens": list(set(tokens1) - set(tokens2)),
            "text2_unique_tokens": list(set(tokens2) - set(tokens1)),
        }
    
    def get_highest_resonance_tokens(self, text: str, top_n: int = 5) -> List[Dict[str, Any]]:
        """
        Get tokens with highest resonance scores in a text
        
        Args:
            text: Input text
            top_n: Number of top tokens to return
            
        Returns:
            List of top tokens with their scores
        """
        scored_tokens = self.encode_with_scores(text)
        
        # Sort by score in descending order
        sorted_tokens = sorted(scored_tokens, key=lambda x: x["score"], reverse=True)
        
        # Return top N
        return sorted_tokens[:top_n]
    
    def find_entangled_patterns(self, text: str, threshold: float = 0.7) -> List[Dict[str, Any]]:
        """
        Find token patterns in text with high entanglement
        
        Args:
            text: Input text
            threshold: Minimum entanglement score threshold
            
        Returns:
            List of token pairs with high entanglement
        """
        # Get tokens with scores
        scored_tokens = self.encode_with_scores(text, include_entanglement=True)
        
        # Find tokens with high scores (potentially entangled)
        high_score_tokens = [token for token in scored_tokens if token["score"] > threshold]
        
        # Get token strings
        tokens = [token["token"] for token in high_score_tokens]
        
        # Create pairs from adjacent high-score tokens
        pairs = []
        for i in range(len(tokens) - 1):
            pairs.append({
                "token_a": tokens[i],
                "token_b": tokens[i+1],
                "score_a": high_score_tokens[i]["score"],
                "score_b": high_score_tokens[i+1]["score"],
                "combined_score": (high_score_tokens[i]["score"] + high_score_tokens[i+1]["score"]) / 2
            })
        
        return pairs

# Example usage
if __name__ == "__main__":
    # Create client
    client = QuantumTokenizerClient()
    
    # Sample text
    sample_text = "The toroidal quantum field synchronizes with the Tree of Life frequency pattern."
    
    print("=== Quantum Tokenization Client Demo ===")
    print(f"Original text: {sample_text}")
    
    # Test tokenization
    print("\n1. Tokenizing text:")
    tokens = client.tokenize(sample_text)
    print(f"Tokens: {tokens}")
    
    # Test encoding
    print("\n2. Encoding text:")
    token_ids = client.encode(sample_text)
    print(f"Token IDs: {token_ids}")
    
    # Test decoding
    print("\n3. Decoding token IDs:")
    decoded_text = client.decode(token_ids)
    print(f"Decoded: {decoded_text}")
    
    # Test resonance analysis
    print("\n4. Analyzing resonance:")
    resonance = client.analyze_resonance(sample_text)
    for metric, value in resonance.items():
        print(f"  {metric}: {value:.4f}")
    
    # Test highest resonance tokens
    print("\n5. Finding highest resonance tokens:")
    top_tokens = client.get_highest_resonance_tokens(sample_text, top_n=3)
    for i, token_data in enumerate(top_tokens, 1):
        print(f"  #{i}: {token_data['token']} (score: {token_data['score']:.4f})")
    
    print("\nClient demo complete!")
