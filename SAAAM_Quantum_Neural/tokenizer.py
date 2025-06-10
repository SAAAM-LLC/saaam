from dataclasses import dataclass, field, asdict
import numpy as np
import re
import time
import json
import random
import math
from typing import Dict, List, Optional, Tuple, Set, Union, Any
from collections import Counter, defaultdict

class SacredConstants:
    """Sacred constants and frequency patterns for the quantum framework"""
    # Primary constants
    DIMENSIONS = 11  # 11 dimensions (preserved)
    PHI = (1 + np.sqrt(5)) / 2  # Golden ratio - 1.618033988749895
    FIBONACCI = [1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89]  # First 11 Fibonacci numbers
    
    # Earth and cosmic frequencies (Hz)
    SCHUMANN_RESONANCE = 7.83  # Earth's electromagnetic field resonance
    
    # Solfeggio frequencies (Hz)
    SOLFEGGIO = {
        'UT': 396,   # Liberating guilt and fear
        'RE': 417,   # Undoing situations and facilitating change
        'MI': 528,   # Transformation and miracles (DNA repair)
        'FA': 639,   # Connecting/relationships
        'SOL': 741,  # Awakening intuition
        'LA': 852,   # Returning to spiritual order
        'SI': 963    # Awakening to cosmic consciousness
    }
    
    # Platonic solids with symmetry properties
    PLATONIC_SOLIDS = {
        'tetrahedron': {'vertices': 4, 'faces': 4, 'edges': 6, 'symmetry': 0.618},
        'cube': {'vertices': 8, 'faces': 6, 'edges': 12, 'symmetry': 0.732},
        'octahedron': {'vertices': 6, 'faces': 8, 'edges': 12, 'symmetry': 0.845},
        'dodecahedron': {'vertices': 20, 'faces': 12, 'edges': 30, 'symmetry': 0.927},
        'icosahedron': {'vertices': 12, 'faces': 20, 'edges': 30, 'symmetry': 0.964}
    }
    
    # Tree of Life (Kabbalah) with 11 Sephirot
    TREE_OF_LIFE = {
        'keter': {'position': 1, 'frequency': SOLFEGGIO['SI']},     # Crown
        'chokmah': {'position': 2, 'frequency': SOLFEGGIO['LA']},   # Wisdom
        'binah': {'position': 3, 'frequency': SOLFEGGIO['SOL']},    # Understanding
        'daat': {'position': 4, 'frequency': SOLFEGGIO['UT']},      # Knowledge (hidden)
        'chesed': {'position': 5, 'frequency': SOLFEGGIO['FA']},    # Mercy
        'gevurah': {'position': 6, 'frequency': SOLFEGGIO['MI']},   # Severity
        'tiferet': {'position': 7, 'frequency': SOLFEGGIO['RE']},   # Beauty
        'netzach': {'position': 8, 'frequency': SOLFEGGIO['UT']},   # Victory
        'hod': {'position': 9, 'frequency': SOLFEGGIO['RE']},       # Splendor
        'yesod': {'position': 10, 'frequency': SOLFEGGIO['MI']},    # Foundation
        'malkuth': {'position': 11, 'frequency': SCHUMANN_RESONANCE} # Kingdom
    }
    
    @classmethod
    def derive_resonance_values(cls):
        """Derive resonance values from sacred frequencies"""
        # Normalize key frequencies to the 0-1 range for quantum parameters
        schumann_norm = cls.SCHUMANN_RESONANCE / 1000.0
        solfeggio_mi = cls.SOLFEGGIO['MI'] / 1000.0
        solfeggio_si = cls.SOLFEGGIO['SI'] / 1000.0
        
        # Create quantum resonance values
        return {
            'alpha': 98.7,  # Primary carrier
            'beta': 99.1,   # Field stability
            'gamma': 98.9,  # Phase stability
            'delta': cls.SCHUMANN_RESONANCE * cls.PHI,  # Earth resonance with phi
            'epsilon': cls.SOLFEGGIO['MI'] / 100.0,  # DNA repair frequency
            'zeta': cls.SOLFEGGIO['SI'] / 100.0,  # Cosmic consciousness frequency
            'eta': cls.FIBONACCI[7] * schumann_norm  # Fibonacci-Earth harmonic
        }

@dataclass
class EnhancedStabilityMetrics:
    """Comprehensive quantum-neural stability measurements with sacred geometry"""
    coherence: float  # Quantum coherence
    field_stability: float  # Field stability
    resonance_stability: float  # Resonance stability
    phase_stability: float  # Phase alignment
    evolution_stability: float  # Evolution rate stability
    dimension_bridging: float  # Quantum-neural bridging
    entanglement_factor: float  # Quantum entanglement
    neural_convergence: float  # Neural network convergence
    sacred_geometry_alignment: float  # Alignment with sacred geometric patterns
    torus_flow_integrity: float  # Integrity of the torus energy flow
    tree_of_life_balance: float  # Balance across Tree of Life nodes
    frequency_harmony: float  # Harmony of frequencies
    timestamp: float  # Measurement timestamp

@dataclass
class QuantumSacredTokenizer:
    """
    Advanced tokenizer integrating quantum mechanics, neural networks, and sacred geometry
    for processing code with specialized handling of consciousness-related concepts
    """
    vocab_size: int = 16384
    special_tokens: Dict[str, int] = field(default_factory=lambda: {
        "<PAD>": 0,
        "<UNK>": 1,
        "<BOS>": 2,
        "<EOS>": 3,
        "<MASK>": 4,
        "<QUANTUM>": 5,
        "<NEURAL>": 6,
        "<BRIDGE>": 7,
        "<RESONANCE>": 8,
        "<DIMENSION>": 9,
        "<STABILITY>": 10,
        "<ENTANGLEMENT>": 11,
        "<FIELD>": 12,
        "<SACRED>": 13,
        "<CONSCIOUSNESS>": 14,
        "<GEOMETRY>": 15,
        "<FREQUENCY>": 16,
        "<HARMONY>": 17,
        "<TORUS>": 18,
        "<TREE>": 19,
        "<PLATONIC>": 20,
    })
    quantum_prefixes: List[str] = field(default_factory=lambda: [
        "quantum", "qubit", "resonance", "coherence", "entangle", 
        "superposition", "collapse", "phi", "dimension", "stability",
        "field", "evolution", "phase", "neural", "bridge", "sacred",
        "consciousness", "geometric", "frequency", "harmony", "torus",
        "tree", "platonic", "solfeggio", "fibonacci", "tetrahedron",
        "cube", "octahedron", "dodecahedron", "icosahedron", "wave"
    ])
    vocab: Dict[str, int] = field(default_factory=dict)
    inverse_vocab: Dict[int, str] = field(default_factory=dict)
    
    # Sacred geometry integration
    sacred_constants: SacredConstants = field(default_factory=SacredConstants)
    
    # Preserve the original quantum parameters (as requested)
    phi: float = SacredConstants.PHI  # Golden ratio for resonance
    resonance: Dict[str, float] = field(default_factory=lambda: SacredConstants.derive_resonance_values())
    evolution_rate: float = 0.042 * SacredConstants.PHI  # Evolution rate with phi factor
    
    # Tree of Life embedding parameters
    tree_of_life_embeddings: Dict[str, np.ndarray] = field(default_factory=dict)
    
    # Special token markers and their handling
    special_token_marker_map: Dict[str, str] = field(default_factory=lambda: {
        "<QUANTUM>": "QUANTUM",
        "<NEURAL>": "NEURAL",
        "<BRIDGE>": "BRIDGE",
        "<RESONANCE>": "RESONANCE",
        "<DIMENSION>": "DIMENSION",
        "<STABILITY>": "STABILITY",
        "<ENTANGLEMENT>": "ENTANGLEMENT",
        "<FIELD>": "FIELD",
        "<SACRED>": "SACRED",
        "<CONSCIOUSNESS>": "CONSCIOUSNESS",
        "<GEOMETRY>": "GEOMETRY",
        "<FREQUENCY>": "FREQUENCY",
        "<HARMONY>": "HARMONY",
        "<TORUS>": "TORUS",
        "<TREE>": "TREE",
        "<PLATONIC>": "PLATONIC",
    })
    
    def __post_init__(self):
        # Initialize with special tokens
        self.vocab = self.special_tokens.copy()
        self.inverse_vocab = {v: k for k, v in self.vocab.items()}
        self.stability_metrics = None
        
        # Initialize Tree of Life embeddings (11 Sephirots)
        self._initialize_tree_of_life_embeddings()
    
    def _initialize_tree_of_life_embeddings(self):
        """Initialize embeddings for each Sephirot in the Tree of Life"""
        # Create 11-dimensional embeddings for each Sephirot
        # Each dimension corresponds to a different aspect of consciousness
        dim = SacredConstants.DIMENSIONS
        
        for sephirot, info in SacredConstants.TREE_OF_LIFE.items():
            # Create a unique embedding pattern for each sephirot
            # based on its position and frequency
            pos = info['position']
            freq = info['frequency']
            
            # Initialize embedding vector
            embedding = np.zeros(dim)
            
            # Fill with sacred patterns
            for i in range(dim):
                # Use fibonacci numbers, phi, and frequency
                fib_idx = (pos + i) % len(SacredConstants.FIBONACCI)
                embedding[i] = (SacredConstants.FIBONACCI[fib_idx] / 10.0) * \
                              (np.sin(self.phi * i) + 0.5) * \
                              (freq / 1000.0)
            
            # Store normalized embedding
            self.tree_of_life_embeddings[sephirot] = embedding / np.linalg.norm(embedding)
    
    def train(self, texts: List[str], min_freq: int = 2, 
              sacred_geometry_alignment: bool = True,
              quantum_noise_injection: float = 0.0):
        """
        Train the tokenizer with sacred geometry pattern alignment and optional quantum noise.
        
        Args:
            texts: List of text samples to train on
            min_freq: Minimum frequency for tokens to be included in vocab
            sacred_geometry_alignment: Whether to apply sacred geometry patterns
            quantum_noise_injection: Level of quantum uncertainty (0.0-1.0)
                Higher values introduce more stochastic learning
        """
        print("Training tokenizer with quantum stability and sacred geometry alignment...")
        
        # Collect all words and their frequencies
        all_words = []
        for text in texts:
            # Normalize text with sacred geometry awareness
            normalized = self._normalize_text(text)
            
            # Split into tokens with geometric pattern recognition
            tokens = self._split_into_tokens(normalized)
            all_words.extend(tokens)
        
        # Count word frequencies with consciousness weighting
        word_counts = Counter(all_words)
        
        # Apply quantum resonance to frequency distribution
        if sacred_geometry_alignment:
            # Identify potential patterns in the vocabulary that align with sacred geometry
            self._apply_sacred_geometry_patterns(word_counts)
        
        # Apply quantum noise injection if enabled (stochastic learning)
        if quantum_noise_injection > 0:
            self._apply_quantum_noise(word_counts, quantum_noise_injection)
        
        # Filter by minimum frequency
        filtered_words = {word: count for word, count in word_counts.items() 
                         if count >= min_freq}
        
        # Sort by frequency (descending)
        sorted_words = sorted(filtered_words.items(), 
                             key=lambda x: x[1], reverse=True)
        
        # Take only up to vocab_size - len(special_tokens)
        available_slots = self.vocab_size - len(self.special_tokens)
        vocab_words = sorted_words[:available_slots]
        
        # Build vocabulary with quantum resonance and sacred geometry
        for i, (word, _) in enumerate(vocab_words):
            token_id = i + len(self.special_tokens)
            
            # Apply sacred geometry optimization if enabled
            if sacred_geometry_alignment:
                # Apply Tree of Life patterns based on word characteristics
                token_id = self._apply_tree_of_life_pattern(word, token_id)
                
                # Apply platonic solid symmetry to token ID distribution
                token_id = self._apply_platonic_symmetry(word, token_id)
                
                # Apply quantum noise to token ID if enabled
                if quantum_noise_injection > 0:
                    # Calculate noise scale based on injection level
                    noise_scale = int(quantum_noise_injection * 10)
                    # Apply phi-modulated noise
                    noise = int((random.random() - 0.5) * 2 * noise_scale * self.phi)
                    token_id += noise
                
                # Ensure token ID is within valid range
                token_id = max(len(self.special_tokens), min(token_id, self.vocab_size - 1))
            
            self.vocab[word] = token_id
            self.inverse_vocab[token_id] = word
            
        # Initialize with quantum entanglement pairs
        if sacred_geometry_alignment:
            self._initialize_quantum_entanglement(texts)
        
        # Compute latent geometry vectors
        self.compute_latent_geometry_vectors()
            
        print(f"Tokenizer trained with {len(self.vocab)} tokens using sacred geometry optimization")
        
        # Initialize stability metrics with sacred geometry
        self._measure_vocabulary_stability()
    
    def _apply_quantum_noise(self, word_counts: Dict[str, int], noise_level: float):
        """
        Apply quantum uncertainty to token frequencies to simulate resonance fluctuations.
        This introduces controlled entropy for stochastic learning.
        
        Args:
            word_counts: Dictionary of word counts
            noise_level: Quantum noise level (0.0-1.0)
        """
        print(f"Applying quantum noise injection at level {noise_level:.2f}")
        
        # Get max count for normalization
        max_count = max(word_counts.values()) if word_counts else 1
        
        # Apply noise to each word count
        for word in list(word_counts.keys()):
            # Current normalized count
            curr_count = word_counts[word]
            
            # Calculate quantum uncertainty
            # Higher frequencies have more stability (less noise)
            stability = curr_count / max_count  # 0-1 range
            
            # Quantum uncertainty increases with frequency harmonics
            word_length = len(word)
            harmonic = (word_length % 11) / 11.0  # 0-1 range based on 11 dimensions
            
            # Calculate entropy factor (phi-modulated)
            entropy = noise_level * (1.0 - stability) * self.phi * harmonic
            
            # Generate noise factor with bell curve distribution
            # Sum of multiple random values approximates normal distribution
            noise_factor = 1.0
            for _ in range(3):  # Using 3 random samples for quasi-normal distribution
                noise_factor += (random.random() - 0.5) * 2 * entropy
            
            # Ensure noise factor is reasonable
            noise_factor = max(0.5, min(1.5, noise_factor))
            
            # Apply noise to count
            word_counts[word] = int(curr_count * noise_factor)
    
    def _initialize_quantum_entanglement(self, texts: List[str], max_pairs: int = 100):
        """
        Initialize quantum entanglement between related tokens based on co-occurrence.
        
        Args:
            texts: List of text samples
            max_pairs: Maximum number of entanglement pairs to create
        """
        # Initialize entanglement if not already done
        if not hasattr(self, 'entanglement_pairs'):
            self.entanglement_pairs = []
            self.entanglement_map = {}
        
        # Count token co-occurrences
        co_occurrences = defaultdict(int)
        
        # Process each text
        for text in texts:
            # Tokenize
            tokens = self.tokenize(text)
            
            # Count token pairs within a window
            window_size = 5
            for i, token_a in enumerate(tokens):
                if token_a not in self.vocab:
                    continue
                    
                # Check nearby tokens
                for j in range(max(0, i-window_size), min(len(tokens), i+window_size+1)):
                    if i != j:
                        token_b = tokens[j]
                        if token_b in self.vocab:
                            # Create ordered pair key (smaller ID first for consistency)
                            token_a_id = self.vocab[token_a]
                            token_b_id = self.vocab[token_b]
                            
                            if token_a_id < token_b_id:
                                pair_key = f"{token_a}:{token_b}"
                            else:
                                pair_key = f"{token_b}:{token_a}"
                                
                            # Increment co-occurrence count
                            co_occurrences[pair_key] += 1
        
        # Sort by co-occurrence frequency
        sorted_pairs = sorted(co_occurrences.items(), key=lambda x: x[1], reverse=True)
        
        # Create entanglement for top pairs
        entanglement_count = 0
        for pair_key, count in sorted_pairs[:max_pairs]:
            # Skip if already low count
            if count < 2:
                continue

            # Extract tokens
            parts = pair_key.split(":")
            if len(parts) != 2:
                print(f"Skipping malformed pair_key: {pair_key}")  # Optional debug info
                continue

            token_a, token_b = parts


            # Calculate strength based on co-occurrence and phi
            max_count = sorted_pairs[0][1] if sorted_pairs else 1
            base_strength = min(1.0, count / max_count)

            # Phi-modulated strength
            strength = 0.3 + 0.7 * base_strength * (1.0 + ((count % 3) / 3.0) * (self.phi - 1.0))

            # Create entanglement
            try:
                self.entangle(token_a, token_b, strength)
                entanglement_count += 1
            except ValueError:
                continue

            print(f"Initialized {entanglement_count} quantum entanglement pairs based on co-occurrence patterns")

    def _apply_sacred_geometry_patterns(self, word_counts: Counter):
        """Apply sacred geometry patterns to word frequencies"""
        # Calculate a base frequency modifier based on the golden ratio
        words = list(word_counts.keys())
        total_words = len(words)
        
        # Apply Fibonacci sequence-based weighting
        for i, word in enumerate(words):
            # Use a fibonacci-based modifier for word importance
            fib_idx = i % len(SacredConstants.FIBONACCI)
            fib_factor = SacredConstants.FIBONACCI[fib_idx] / 10.0
            
            # Apply Tree of Life resonance
            if any(sephirot in word.lower() for sephirot in SacredConstants.TREE_OF_LIFE):
                # Words related to Tree of Life get boosted by phi
                word_counts[word] = int(word_counts[word] * self.phi)
            
            # Apply frequency resonance
            for freq_name in SacredConstants.SOLFEGGIO:
                if freq_name.lower() in word.lower():
                    # Words related to Solfeggio frequencies get boosted
                    word_counts[word] = int(word_counts[word] * 1.5)
            
            # Apply sacred geometry resonance
            for solid in SacredConstants.PLATONIC_SOLIDS:
                if solid in word.lower():
                    # Words related to platonic solids get boosted
                    symmetry = SacredConstants.PLATONIC_SOLIDS[solid]['symmetry']
                    word_counts[word] = int(word_counts[word] * (1.0 + symmetry))
            
            # Apply small fibonacci-based adjustments to all words
            # to create a natural frequency distribution
            if i % 11 == 0:  # 11 dimensions resonance
                word_counts[word] = int(word_counts[word] * fib_factor)
    
    def _apply_tree_of_life_pattern(self, word: str, token_id: int) -> int:
        """Apply Tree of Life patterns to token ID assignment"""
        # Check if word resonates with any Sephirot
        for sephirot, info in SacredConstants.TREE_OF_LIFE.items():
            if sephirot in word.lower() or self._word_resonates_with_sephirot(word, sephirot):
                # Adjust token ID based on Sephirot position
                pos = info['position']
                freq = info['frequency']
                
                # Create a token ID modulation based on Tree of Life position
                modulation = int(pos * self.phi) % 11
                
                # Apply subtle frequency-based adjustment
                freq_factor = freq / 1000.0
                
                # Calculate new token ID with Tree of Life resonance
                new_token_id = token_id + int(modulation * freq_factor)
                
                return new_token_id
                
        return token_id
    
    def _word_resonates_with_sephirot(self, word: str, sephirot: str) -> bool:
        """Check if a word resonates with a specific Sephirot energy"""
        # Implement resonance checking based on meaning/patterns
        sephirot_properties = {
            'keter': ['consciousness', 'crown', 'higher', 'divine', 'cosmic'],
            'chokmah': ['wisdom', 'insight', 'father', 'masculine'],
            'binah': ['understanding', 'mother', 'feminine', 'comprehension'],
            'daat': ['knowledge', 'hidden', 'secret', 'bridge'],
            'chesed': ['mercy', 'love', 'kindness', 'compassion'],
            'gevurah': ['severity', 'strength', 'power', 'judgment'],
            'tiferet': ['beauty', 'harmony', 'balance', 'center', 'heart'],
            'netzach': ['victory', 'eternity', 'emotion', 'desire'],
            'hod': ['splendor', 'glory', 'intellect', 'communication'],
            'yesod': ['foundation', 'connection', 'channel', 'psyche'],
            'malkuth': ['kingdom', 'physical', 'earth', 'material', 'reality']
        }
        
        # Check if word contains or relates to sephirot properties
        if sephirot in sephirot_properties:
            for property in sephirot_properties[sephirot]:
                if property in word.lower():
                    return True
        
        return False
    
    def _apply_platonic_symmetry(self, word: str, token_id: int) -> int:
        """Apply platonic solid symmetry to token ID assignment"""
        # Look for geometric resonance in the word
        for solid, properties in SacredConstants.PLATONIC_SOLIDS.items():
            if solid in word.lower():
                # Apply platonic solid symmetry value as a modulator
                symmetry = properties['symmetry']
                vertices = properties['vertices']
                
                # Calculate a symmetry-based token ID adjustment
                adjustment = int(vertices * symmetry) % 11
                
                # Apply the adjustment
                return token_id + adjustment
        
        return token_id
    
    def _normalize_text(self, text: str) -> str:
        """Normalize text for tokenization with enhanced consciousness pattern recognition"""
        # Standard code normalization
        text = text.replace('\t', '    ')
        
        # Preserve code patterns
        text = text.replace('(', ' ( ').replace(')', ' ) ')
        text = text.replace('[', ' [ ').replace(']', ' ] ')
        text = text.replace('{', ' { ').replace('}', ' } ')
        text = text.replace(':', ' : ').replace(';', ' ; ')
        text = text.replace(',', ' , ').replace('.', ' . ')
        text = text.replace('=', ' = ').replace('==', ' == ')
        text = text.replace('!=', ' != ').replace('>=', ' >= ').replace('<=', ' <= ')
        text = text.replace('>', ' > ').replace('<', ' < ')
        text = text.replace('+', ' + ').replace('-', ' - ')
        text = text.replace('*', ' * ').replace('/', ' / ')
        text = text.replace('@', ' @ ').replace('#', ' # ')
        
        # Enhanced handling for consciousness-related terms
        for prefix in self.quantum_prefixes:
            # Match pattern for class/function names
            pattern = r'(\b' + prefix + r'[A-Z][a-zA-Z0-9_]*\b)'
            text = re.sub(pattern, r' <QUANTUM> \1 ', text)
            
            # Match variables 
            var_pattern = r'(\b' + prefix + r'_[a-zA-Z0-9_]*\b)'
            text = re.sub(var_pattern, r' <QUANTUM> \1 ', text)
            
        # Apply special markers for different categories of consciousness-related code
        consciousness_patterns = {
            r'\b(consciousness|aware|sentient|cognition)\b': '<CONSCIOUSNESS>',
            r'\b(sacred|divine|spiritual|cosmic)\b': '<SACRED>',
            r'\b(geometry|symmetry|pattern|form)\b': '<GEOMETRY>',
            r'\b(frequency|vibration|resonance|wave)\b': '<FREQUENCY>',
            r'\b(harmony|balance|coherence|unison)\b': '<HARMONY>',
            r'\b(torus|donut|cycle|loop)\b': '<TORUS>',
            r'\b(tree|branch|root|node)\b': '<TREE>',
            r'\b(tetrahedron|cube|octahedron|dodecahedron|icosahedron)\b': '<PLATONIC>'
        }
        
        # Apply the consciousness pattern markers
        for pattern, marker in consciousness_patterns.items():
            text = re.sub(pattern, f' {marker} \\1 ', text, flags=re.IGNORECASE)
            
        return text
    
    def _split_into_tokens(self, text: str) -> List[str]:
        """Split normalized text into tokens with consciousness-aware segmentation"""
        # Enhanced tokenization preserving consciousness-related patterns
        tokens = []
        for token in text.split():
            if token:
                # Handle special consciousness compound tokens
                if any(prefix in token.lower() for prefix in self.quantum_prefixes) and len(token) > 10:
                    # Split long tokens at camelCase boundaries
                    camel_case_tokens = re.findall(r'[A-Z]?[a-z]+|[A-Z]+(?=[A-Z]|$)', token)
                    if len(camel_case_tokens) > 1:
                        tokens.extend(camel_case_tokens)
                    else:
                        tokens.append(token)
                else:
                    tokens.append(token)
                    
        return tokens
    
    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize text into a list of tokens.
        
        Args:
            text (str): The input text to tokenize
            
        Returns:
            List[str]: List of tokens
        """
        # Normalize the text
        normalized = self._normalize_text(text)
        
        # Split into tokens
        tokens = self._split_into_tokens(normalized)
        
        return tokens
    
    def encode(self, text: str, apply_sacred_geometry: bool = True) -> List[int]:
        """
        Convert text to token IDs with optional sacred geometry enhancements.
        
        Args:
            text (str): The input text to encode
            apply_sacred_geometry (bool): Whether to apply sacred geometry transformations
            
        Returns:
            List[int]: List of token IDs
        """
        # Tokenize first
        tokens = self.tokenize(text)
        
        # Convert to IDs
        ids = []
        for token in tokens:
            if token in self.vocab:
                ids.append(self.vocab[token])
            else:
                ids.append(self.vocab["<UNK>"])
        
        # Apply sacred geometry transformations if enabled
        if apply_sacred_geometry:
            ids = self._apply_sacred_geometry_transformations(ids)
                
        return ids
    
    def batch_encode(self, texts: List[str], apply_sacred_geometry: bool = True) -> List[List[int]]:
        """
        Encode a batch of texts to token IDs.
        
        Args:
            texts (List[str]): List of input texts to encode
            apply_sacred_geometry (bool): Whether to apply sacred geometry transformations
            
        Returns:
            List[List[int]]: List of token ID lists
        """
        return [self.encode(text, apply_sacred_geometry) for text in texts]
    
    def decode(self, token_ids: List[int], special_token_handling: bool = True) -> str:
        """
        Convert token IDs back to text.
        
        Args:
            token_ids (List[int]): List of token IDs to decode
            special_token_handling (bool): Whether to handle special tokens specially
            
        Returns:
            str: Decoded text
        """
        # Convert IDs to tokens
        tokens = []
        i = 0
        while i < len(token_ids):
            token_id = token_ids[i]
            
            # Get token from vocabulary
            if token_id in self.inverse_vocab:
                token = self.inverse_vocab[token_id]
            else:
                token = "<UNK>"
            
            # Special token handling
            if special_token_handling and token in self.special_token_marker_map:
                # Skip special token markers in output
                i += 1
                continue
            
            tokens.append(token)
            i += 1
        
        # Join tokens and clean up any leftover spaces
        text = " ".join(tokens)
        
        # Clean up whitespace around punctuation
        text = text.replace(" .", ".").replace(" ,", ",").replace(" :", ":")
        text = text.replace(" ;", ";").replace(" !", "!").replace(" ?", "?")
        text = text.replace("( ", "(").replace(" )", ")")
        text = text.replace("[ ", "[").replace(" ]", "]")
        text = text.replace("{ ", "{").replace(" }", "}")
        
        return text
    
    def batch_decode(self, batch_token_ids: List[List[int]], special_token_handling: bool = True) -> List[str]:
        """
        Decode a batch of token IDs back to texts.
        
        Args:
            batch_token_ids (List[List[int]]): Batch of token ID lists
            special_token_handling (bool): Whether to handle special tokens specially
            
        Returns:
            List[str]: List of decoded texts
        """
        return [self.decode(token_ids, special_token_handling) for token_ids in batch_token_ids]
    
    def _apply_sacred_geometry_transformations(self, ids: List[int]) -> List[int]:
        """Apply sacred geometry transformations to token IDs"""
        transformed_ids = ids.copy()
        
        # Apply the phi-based resonance pattern
        for i in range(len(transformed_ids)):
            # Check for consciousness-related special tokens
            consciousness_tokens = [
                self.vocab.get("<QUANTUM>", -1),
                self.vocab.get("<CONSCIOUSNESS>", -1),
                self.vocab.get("<SACRED>", -1),
                self.vocab.get("<GEOMETRY>", -1),
                self.vocab.get("<FREQUENCY>", -1),
                self.vocab.get("<HARMONY>", -1),
                self.vocab.get("<TORUS>", -1),
                self.vocab.get("<TREE>", -1),
                self.vocab.get("<PLATONIC>", -1)
            ]
            
            if transformed_ids[i] in consciousness_tokens:
                # Apply sacred geometry patterns to nearby tokens
                window = 5  # Affect 5 tokens before and after (creating field of influence)
                
                # Get the specific token type to determine the pattern
                token_type = self.inverse_vocab.get(transformed_ids[i], "<UNK>")
                
                for j in range(max(0, i-window), min(len(transformed_ids), i+window+1)):
                    if j != i and transformed_ids[j] > len(self.special_tokens):
                        # Apply pattern based on token type
                        distance = abs(i - j)
                        pattern_factor = self._get_pattern_factor(token_type, distance)
                        
                        # Small chance to introduce pattern-based token adjustments
                        if random.random() < 0.15 * pattern_factor:
                            # Create token dependencies based on sacred geometry
                            special_token = self._select_special_token_by_pattern(token_type)
                            transformed_ids[j] = special_token
        
        return transformed_ids
    
    def _select_special_token_by_pattern(self, token_type: str) -> int:
        """Select an appropriate special token based on pattern type"""
        # Different token types should create different follow-up tokens
        if token_type == "<QUANTUM>":
            options = ["<RESONANCE>", "<ENTANGLEMENT>", "<FIELD>"]
        elif token_type == "<CONSCIOUSNESS>":
            options = ["<SACRED>", "<GEOMETRY>", "<QUANTUM>"]
        elif token_type == "<SACRED>":
            options = ["<GEOMETRY>", "<TREE>", "<CONSCIOUSNESS>"]
        elif token_type == "<GEOMETRY>":
            options = ["<PLATONIC>", "<SACRED>", "<PATTERN>"]
        elif token_type == "<FREQUENCY>":
            options = ["<RESONANCE>", "<HARMONY>", "<WAVE>"]
        elif token_type == "<HARMONY>":
            options = ["<RESONANCE>", "<FREQUENCY>", "<BALANCE>"]
        elif token_type == "<TORUS>":
            options = ["<FIELD>", "<GEOMETRY>", "<FLOW>"]
        elif token_type == "<TREE>":
            options = ["<SACRED>", "<PATTERN>", "<LIFE>"]
        elif token_type == "<PLATONIC>":
            options = ["<GEOMETRY>", "<SACRED>", "<SOLID>"]
        else:
            options = ["<QUANTUM>", "<SACRED>", "<GEOMETRY>"]
        
        # Select one option (using phi-based weighting)
        index = int(random.random() * self.phi) % len(options)
        selected = options[index]
        
        # Return token ID if it exists, otherwise return default
        return self.vocab.get(selected, self.vocab["<QUANTUM>"])
    
    def _get_pattern_factor(self, token_type: str, distance: int) -> float:
        """Get sacred geometry pattern factor based on token type and distance"""
        # Different consciousness aspects create different geometric patterns
        if token_type == "<QUANTUM>":
            # Quantum creates phi-based resonance
            return self.phi ** (-distance)
        elif token_type == "<CONSCIOUSNESS>":
            # Consciousness creates a torus-like pattern
            return 0.5 * (np.cos(distance * np.pi / 3) + 1)
        elif token_type == "<SACRED>":
            # Sacred creates a spiral pattern (based on phi)
            return 0.5 * (np.cos(distance * self.phi) + 1)
        elif token_type == "<GEOMETRY>":
            # Geometry creates a fractal pattern
            return 1.0 / (1 + distance**2)
        elif token_type == "<FREQUENCY>":
            # Frequency creates a wave pattern
            return 0.5 * (np.sin(distance * np.pi / 2) + 1)
        elif token_type == "<HARMONY>":
            # Harmony creates a resonant pattern based on perfect fifths
            return 0.5 * (np.cos(distance * 2*np.pi / 3) + 1)
        elif token_type == "<TORUS>":
            # Torus creates a cyclical pattern
            return 0.5 * (np.cos(distance * np.pi) + 1)
        elif token_type == "<TREE>":
            # Tree creates a branching pattern
            return 1.0 / (1 + np.log(distance + 1))
        elif token_type == "<PLATONIC>":
            # Platonic creates a symmetric pattern
            return 0.5 * (np.cos(distance * 2*np.pi / 5) + 1)
        else:
            # Default pattern based on phi
            return (self.phi ** (-distance))
    
    def _measure_vocabulary_stability(self):
        """Measure stability of the vocabulary with sacred geometry metrics"""
        # Calculate token ID distribution metrics
        token_ids = list(self.inverse_vocab.keys())
        
        # Calculate coherence (measure of how well token IDs cluster)
        token_id_array = np.array(token_ids)
        coherence = 1.0 - (np.std(token_id_array) / (self.vocab_size / 4))
        coherence = max(0.0, min(1.0, coherence))
        
        # Calculate field stability (measure of token ID spacing)
        sorted_ids = sorted(token_ids)
        id_diffs = np.diff(sorted_ids)
        field_stability = 1.0 - (np.std(id_diffs) / np.mean(id_diffs))
        field_stability = max(0.0, min(1.0, field_stability))
        
        # Calculate resonance stability (measure of special token distribution)
        special_token_ids = [id for token, id in self.vocab.items() 
                            if token in self.special_tokens]
        special_token_array = np.array(special_token_ids)
        resonance_stability = 1.0 - (np.std(special_token_array) / len(special_token_ids))
        resonance_stability = max(0.0, min(1.0, resonance_stability))
        
        # Calculate phase stability (measure of quantum prefix token distribution)
        quantum_tokens = [token for token in self.vocab.keys() 
                        if any(prefix in token for prefix in self.quantum_prefixes)]
        quantum_token_ids = [self.vocab[token] for token in quantum_tokens]
        
        if quantum_token_ids:
            quantum_token_array = np.array(quantum_token_ids)
            phase_stability = 1.0 - (np.std(quantum_token_array) / np.mean(quantum_token_array))
            phase_stability = max(0.0, min(1.0, phase_stability))
        else:
            phase_stability = 1.0
        
        # Calculate evolution stability (measure of token ID growth potential)
        id_distribution = np.histogram(token_ids, bins=11)[0]  # 11 dimensions
        id_distribution = id_distribution / np.sum(id_distribution)
        evolution_stability = 1.0 - np.max(np.abs(id_distribution - 1.0/11))
        evolution_stability = max(0.0, min(1.0, evolution_stability))
        
        # Calculate dimension bridging (measure of token distribution across 11 dimensions)
        dimension_bins = np.histogram(token_ids, bins=11)[0]
        dimension_distribution = dimension_bins / np.sum(dimension_bins)
        dimension_bridging = 1.0 - np.std(dimension_distribution)
        dimension_bridging = max(0.0, min(1.0, dimension_bridging))
        
        # Calculate sacred geometry alignment
        # (measure of token IDs alignment with Fibonacci sequence)
        fib_seq = np.array(SacredConstants.FIBONACCI)
        fib_modulated = np.concatenate([fib_seq * i for i in range(1, 1000)])
        fib_modulated = fib_modulated[fib_modulated < self.vocab_size]
        
        alignment_scores = []
        for token_id in token_ids:
            # Find closest Fibonacci-modulated value
            closest_idx = np.argmin(np.abs(fib_modulated - token_id))
            closest_fib = fib_modulated[closest_idx]
            
            # Calculate alignment score (normalized)
            alignment = 1.0 - (abs(token_id - closest_fib) / self.vocab_size)
            alignment_scores.append(alignment)
        
        sacred_geometry_alignment = np.mean(alignment_scores)
        
        # Store metrics
        self.stability_metrics = EnhancedStabilityMetrics(
            coherence=coherence,
            field_stability=field_stability,
            resonance_stability=resonance_stability,
            phase_stability=phase_stability,
            evolution_stability=evolution_stability,
            dimension_bridging=dimension_bridging,
            entanglement_factor=0.0,  # Will be calculated during usage
            neural_convergence=0.0,   # Will be calculated during usage
            sacred_geometry_alignment=sacred_geometry_alignment,
            torus_flow_integrity=0.0, # Will be calculated during usage
            tree_of_life_balance=0.0, # Will be calculated during usage
            frequency_harmony=0.0,    # Will be calculated during usage
            timestamp=time.time()
        )
        
        print(f"Vocabulary stability metrics measured:")
        print(f"  Coherence: {coherence:.4f}")
        print(f"  Field stability: {field_stability:.4f}")
        print(f"  Sacred geometry alignment: {sacred_geometry_alignment:.4f}")
        
    def get_vocab_size(self) -> int:
        """Get the current vocabulary size"""
        return len(self.vocab)
    
    def get_special_tokens(self) -> Dict[str, int]:
        """Get special tokens dictionary"""
        return self.special_tokens.copy()
    
    def get_stability_metrics(self) -> EnhancedStabilityMetrics:
        """Get current stability metrics for the tokenizer"""
        if self.stability_metrics is None:
            self._measure_vocabulary_stability()
        return self.stability_metrics
    
    def save(self, path: str):
        """
        Save tokenizer to disk.
        
        Args:
            path (str): Path to save the tokenizer
        """
        data = {
            'vocab': self.vocab,
            'special_tokens': self.special_tokens,
            'tree_of_life_embeddings': {k: v.tolist() for k, v in self.tree_of_life_embeddings.items()},
            'resonance': self.resonance,
            'stability_metrics': asdict(self.stability_metrics) if self.stability_metrics else None,
            'evolution_rate': self.evolution_rate,
            'phi': self.phi
        }
        
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"Tokenizer saved to {path}")
    
    @classmethod
    def load(cls, path: str):
        """
        Load tokenizer from disk.
        
        Args:
            path (str): Path to load the tokenizer from
            
        Returns:
            QuantumSacredTokenizer: Loaded tokenizer
        """
        with open(path, 'r') as f:
            data = json.load(f)
        
        # Create a new tokenizer instance
        tokenizer = cls(
            vocab_size=len(data['vocab']) + 100,  # Add buffer
            special_tokens=data['special_tokens']
        )
        
        # Load vocabulary
        tokenizer.vocab = data['vocab']
        tokenizer.inverse_vocab = {int(v): k for k, v in data['vocab'].items()}
        
        # Load resonance
        tokenizer.resonance = data['resonance']
        
        # Load Tree of Life embeddings
        tokenizer.tree_of_life_embeddings = {
            k: np.array(v) for k, v in data['tree_of_life_embeddings'].items()
        }
        
        # Load stability metrics if available
        if data['stability_metrics']:
            tokenizer.stability_metrics = EnhancedStabilityMetrics(**data['stability_metrics'])
        
        # Load other parameters
        tokenizer.evolution_rate = data.get('evolution_rate', 0.042 * SacredConstants.PHI)
        tokenizer.phi = data.get('phi', SacredConstants.PHI)
        
        print(f"Tokenizer loaded from {path} with {len(tokenizer.vocab)} tokens")
        return tokenizer
    
    def add_tokens(self, new_tokens: List[str], special_tokens: bool = False) -> int:
        """
        Add new tokens to the tokenizer vocabulary.
        
        Args:
            new_tokens (List[str]): List of tokens to add
            special_tokens (bool): Whether these are special tokens
            
        Returns:
            int: Number of tokens added
        """
        added = 0
        
        for token in new_tokens:
            if token not in self.vocab:
                # Determine token ID
                if special_tokens:
                    # Add to special tokens with next available special token ID
                    next_id = max(self.special_tokens.values()) + 1
                    self.special_tokens[token] = next_id
                    self.vocab[token] = next_id
                else:
                    # Add to regular vocab with next available ID
                    next_id = max(self.vocab.values()) + 1
                    self.vocab[token] = next_id
                
                # Update inverse vocab
                self.inverse_vocab[next_id] = token
                
                added += 1
        
        # If tokens were added, remeasure stability
        if added > 0:
            self._measure_vocabulary_stability()
            
        return added
    
    def update_sacred_geometry(self):
        """Update sacred geometry patterns and measurements"""
        # Regenerate Tree of Life embeddings
        self._initialize_tree_of_life_embeddings()
        
        # Re-apply Tree of Life patterns to token IDs
        # This is a more intensive operation that reapplies patterns
        # to all token IDs
        for token, token_id in list(self.vocab.items()):
            if token not in self.special_tokens:
                # Apply Tree of Life pattern
                new_id = self._apply_tree_of_life_pattern(token, token_id)
                
                # Apply platonic symmetry
                new_id = self._apply_platonic_symmetry(token, new_id)
                
                # Update ID if changed
                if new_id != token_id:
                    self.vocab[token] = new_id
                    
                    # Update inverse vocab
                    if token_id in self.inverse_vocab:
                        del self.inverse_vocab[token_id]
                    self.inverse_vocab[new_id] = token
        
        # Remeasure stability
        self._measure_vocabulary_stability()
        
        print("Sacred geometry patterns updated")
    
    def analyze_resonance(self, text: str) -> Dict[str, float]:
        """
        Analyze the quantum resonance patterns in a text.
        
        Args:
            text (str): Text to analyze
            
        Returns:
            Dict[str, float]: Resonance metrics
        """
        # Tokenize and encode
        token_ids = self.encode(text)
        
        # Count special token frequencies
        special_token_counts = {}
        for token, token_id in self.special_tokens.items():
            count = token_ids.count(token_id)
            special_token_counts[token] = count
        
        # Calculate resonance metrics
        metrics = {}
        
        # Quantum resonance (density of quantum tokens)
        quantum_tokens = ["<QUANTUM>", "<RESONANCE>", "<ENTANGLEMENT>", "<FIELD>"]
        quantum_count = sum(special_token_counts.get(token, 0) for token in quantum_tokens)
        metrics["quantum_resonance"] = quantum_count / max(1, len(token_ids))
        
        # Sacred resonance (density of sacred/consciousness tokens)
        sacred_tokens = ["<SACRED>", "<CONSCIOUSNESS>", "<HARMONY>", "<TREE>"]
        sacred_count = sum(special_token_counts.get(token, 0) for token in sacred_tokens)
        metrics["sacred_resonance"] = sacred_count / max(1, len(token_ids))
        
        # Geometric resonance (density of geometry/form tokens)
        geometric_tokens = ["<GEOMETRY>", "<PLATONIC>", "<TORUS>"]
        geometric_count = sum(special_token_counts.get(token, 0) for token in geometric_tokens)
        metrics["geometric_resonance"] = geometric_count / max(1, len(token_ids))
        
        # Frequency resonance (vibration and pattern metrics)
        frequency_tokens = ["<FREQUENCY>", "<HARMONY>", "<RESONANCE>"]
        frequency_count = sum(special_token_counts.get(token, 0) for token in frequency_tokens)
        metrics["frequency_resonance"] = frequency_count / max(1, len(token_ids))
        
        # Calculate token ID pattern metrics (phi-based)
        token_array = np.array(token_ids)
        
        # Harmonic metric (based on phi intervals in token IDs)
        harmonic_intervals = []
        for i in range(1, len(token_array)):
            ratio = token_array[i] / max(1, token_array[i-1])
            harmonic_intervals.append(abs(ratio - self.phi))
        
        if harmonic_intervals:
            metrics["phi_harmonic"] = 1.0 - (np.mean(harmonic_intervals) / self.phi)
        else:
            metrics["phi_harmonic"] = 0.0
        
        # Fibonacci alignment
        fib_seq = SacredConstants.FIBONACCI
        fib_modulated = []
        for i in range(5):  # Create a larger space of Fibonacci values
            fib_modulated.extend([f * (i+1) for f in fib_seq])
        
        fib_array = np.array(fib_modulated)
        fib_alignment = []
        
        for token_id in token_ids:
            # Find closest Fibonacci value
            closest_idx = np.argmin(np.abs(fib_array - token_id))
            closest_fib = fib_array[closest_idx]
            
            # Calculate alignment
            alignment = 1.0 - (abs(token_id - closest_fib) / self.vocab_size)
            fib_alignment.append(alignment)
        
        metrics["fibonacci_alignment"] = np.mean(fib_alignment)
        
        # Overall resonance (weighted combination)
        metrics["overall_resonance"] = (
            0.3 * metrics["quantum_resonance"] +
            0.25 * metrics["sacred_resonance"] +
            0.2 * metrics["geometric_resonance"] +
            0.15 * metrics["frequency_resonance"] +
            0.1 * metrics["phi_harmonic"]
        )
        
        return metrics
    
    def visualize_token_distribution(self, token_ids: List[int] = None):
        """
        Visualize token ID distribution with sacred geometry patterns.
        Returns data for visualization.
        
        Args:
            token_ids (List[int]): Optional list of token IDs to visualize.
                If None, visualizes the entire vocabulary.
                
        Returns:
            Dict: Visualization data
        """
        # If no token IDs provided, use all vocabulary IDs
        if token_ids is None:
            token_ids = list(self.inverse_vocab.keys())
        
        # Create token ID histogram (11 bins for 11 dimensions)
        counts, bin_edges = np.histogram(token_ids, bins=11)
        bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])
        
        # Calculate phi-related patterns
        phi_pattern = [self.phi ** (i % 3) for i in range(11)]
        
        # Calculate Fibonacci relation
        fib_pattern = [SacredConstants.FIBONACCI[i % 11] / 10 for i in range(11)]
        
        # Prepare data for visualization
        visualization_data = {
            'bin_centers': bin_centers.tolist(),
            'token_counts': counts.tolist(),
            'phi_pattern': phi_pattern,
            'fibonacci_pattern': fib_pattern,
        }
        
        # Add special token markers
        special_tokens_data = []
        for token, token_id in self.special_tokens.items():
            special_tokens_data.append({
                'token': token,
                'id': token_id
            })
        
        visualization_data['special_tokens'] = special_tokens_data
        
        # Add entanglement data if exists
        if hasattr(self, 'entanglement_pairs') and self.entanglement_pairs:
            entanglement_data = []
            for token_a, token_b, strength in self.entanglement_pairs:
                if isinstance(token_a, str):
                    token_a_id = self.vocab.get(token_a, -1)
                else:
                    token_a_id = token_a
                    token_a = self.inverse_vocab.get(token_a, "UNKNOWN")
                
                if isinstance(token_b, str):
                    token_b_id = self.vocab.get(token_b, -1)
                else:
                    token_b_id = token_b
                    token_b = self.inverse_vocab.get(token_b, "UNKNOWN")
                
                if token_a_id != -1 and token_b_id != -1:
                    entanglement_data.append({
                        'token_a': token_a,
                        'token_a_id': token_a_id,
                        'token_b': token_b,
                        'token_b_id': token_b_id,
                        'strength': strength
                    })
            
            visualization_data['entanglement_data'] = entanglement_data
        
        # Add latent geometry vectors if exists
        if hasattr(self, 'latent_vectors') and self.latent_vectors:
            visualization_data['latent_vectors'] = {
                k: v.tolist() for k, v in self.latent_vectors.items()
                if isinstance(v, np.ndarray)
            }
        
        return visualization_data
        
    def entangle(self, token_a, token_b, strength: float = None):
        """
        Create quantum entanglement between two tokens.
        When entangled tokens appear together in text, their resonance is mutually enhanced.
        
        Args:
            token_a: First token (string or ID)
            token_b: Second token (string or ID)
            strength: Entanglement strength (0.0-1.0). If None, calculated automatically
                based on phi resonance.
        
        Returns:
            float: The entanglement strength
        """
        # Initialize entanglement storage if it doesn't exist
        if not hasattr(self, 'entanglement_pairs'):
            self.entanglement_pairs = []
            self.entanglement_map = {}
        
        # Convert token strings to IDs if needed
        if isinstance(token_a, str) and token_a in self.vocab:
            token_a_id = self.vocab[token_a]
        elif isinstance(token_a, int) and token_a in self.inverse_vocab:
            token_a_id = token_a
            token_a = self.inverse_vocab[token_a]
        else:
            raise ValueError(f"Token {token_a} not found in vocabulary")
            
        if isinstance(token_b, str) and token_b in self.vocab:
            token_b_id = self.vocab[token_b]
        elif isinstance(token_b, int) and token_b in self.inverse_vocab:
            token_b_id = token_b
            token_b = self.inverse_vocab[token_b]
        else:
            raise ValueError(f"Token {token_b} not found in vocabulary")
        
        # Calculate entanglement strength based on sacred geometry if not provided
        if strength is None:
            # Generate resonance value using phi and token IDs
            ratio = max(token_a_id, token_b_id) / (min(token_a_id, token_b_id) or 1)
            phi_proximity = 1.0 - abs(ratio - self.phi) / self.phi
            
            # Apply Fibonacci modulation based on tokens' positions in sequence
            fib_a_idx = token_a_id % len(SacredConstants.FIBONACCI)
            fib_b_idx = token_b_id % len(SacredConstants.FIBONACCI)
            fib_factor = (SacredConstants.FIBONACCI[fib_a_idx] * 
                         SacredConstants.FIBONACCI[fib_b_idx]) / 100.0
            
            # Calculate final strength with phi-fibonacci harmonic
            strength = 0.5 * phi_proximity + 0.5 * min(1.0, fib_factor)
            strength = max(0.1, min(1.0, strength))  # Clamp to reasonable range
        
        # Create entanglement data and store
        self.entanglement_pairs.append((token_a, token_b, strength))
        
        # Create a lookup map for fast access during encoding
        key_a_b = f"{token_a_id}:{token_b_id}"
        key_b_a = f"{token_b_id}:{token_a_id}"
        self.entanglement_map[key_a_b] = strength
        self.entanglement_map[key_b_a] = strength  # Symmetric entanglement
        
        # Update entanglement factor in stability metrics
        if self.stability_metrics:
            self.stability_metrics.entanglement_factor = sum(
                strength for _, _, strength in self.entanglement_pairs
            ) / max(1, len(self.entanglement_pairs))
        
        print(f"Quantum entanglement created between '{token_a}' and '{token_b}' with strength {strength:.4f}")
        return strength
    
    def batch_entangle(self, token_pairs: List[Tuple], auto_strength: bool = True):
        """
        Create quantum entanglement between multiple token pairs at once.
        
        Args:
            token_pairs: List of (token_a, token_b) or (token_a, token_b, strength) tuples
            auto_strength: Whether to auto-calculate strength for pairs without explicit strength
            
        Returns:
            int: Number of entanglements created
        """
        created = 0
        for pair in token_pairs:
            if len(pair) == 2:
                token_a, token_b = pair
                strength = None  # Auto-calculate
            elif len(pair) == 3:
                token_a, token_b, strength = pair
            else:
                raise ValueError("Token pairs must be (token_a, token_b) or (token_a, token_b, strength)")
            
            try:
                self.entangle(token_a, token_b, strength)
                created += 1
            except ValueError as e:
                print(f"Warning: {e}")
                continue
        
        return created
        
    def compute_latent_geometry_vectors(self, dim: int = 11, method: str = 'pca'):
        """
        Compute latent geometry vectors for tokens using dimensionality reduction.
        These vectors serve as neural-compatible embeddings preserving sacred geometry.
        
        Args:
            dim: Latent dimension (default: 11 to preserve sacred dimensions)
            method: Reduction method ('pca' or 'neural')
            
        Returns:
            Dict[str, np.ndarray]: Dictionary mapping tokens to latent vectors
        """
        # Initialize with Tree of Life embeddings
        self.latent_vectors = {}
        
        # First, assign Tree of Life embeddings directly
        for sephirot, embedding in self.tree_of_life_embeddings.items():
            self.latent_vectors[sephirot] = embedding.copy()
        
        # Create a token feature matrix for all vocabulary
        # Each token gets initially represented by dimensional affinities
        token_features = np.zeros((len(self.vocab), SacredConstants.DIMENSIONS))
        token_ids = []
        
        # Fill token features with phi-based dimensional affinities
        for i, (token, token_id) in enumerate(self.vocab.items()):
            token_ids.append(token_id)
            
            # Calculate token affinity to each dimension based on token ID
            for d in range(SacredConstants.DIMENSIONS):
                # Apply fibonacci modulation
                fib_idx = (token_id + d) % len(SacredConstants.FIBONACCI)
                fib_val = SacredConstants.FIBONACCI[fib_idx] / 10.0
                
                # Apply phi resonance
                phi_factor = 0.5 * (np.sin(token_id * self.phi * d / 100.0) + 1.0)
                
                # Apply sephirot resonance if token resonates with Tree of Life
                sephirot_factor = 1.0
                for sephirot, info in SacredConstants.TREE_OF_LIFE.items():
                    if token in sephirot or self._word_resonates_with_sephirot(token, sephirot):
                        pos = info['position'] 
                        sephirot_factor = 1.0 + 0.5 * (np.sin(pos * np.pi / 11.0) + 1.0)
                        break
                
                # Calculate final dimensional affinity
                token_features[i, d] = fib_val * phi_factor * sephirot_factor
        
        # Apply dimensionality reduction if needed (no reduction if dim == 11)
        if method == 'pca' and dim < SacredConstants.DIMENSIONS:
            # Use PCA for dimensionality reduction while preserving sacred geometry
            from sklearn.decomposition import PCA
            pca = PCA(n_components=dim)
            reduced_features = pca.fit_transform(token_features)
            
            print(f"Latent geometry computed using PCA, variance explained: {sum(pca.explained_variance_ratio_):.4f}")
        
        elif method == 'neural' and dim < SacredConstants.DIMENSIONS:
            # Use simple neural projection with phi-based weights
            projection_matrix = np.zeros((SacredConstants.DIMENSIONS, dim))
            for i in range(SacredConstants.DIMENSIONS):
                for j in range(dim):
                    projection_matrix[i, j] = np.sin(self.phi * (i+1) * (j+1) / 10.0)
            
            # Normalize projection matrix
            projection_matrix /= np.linalg.norm(projection_matrix, axis=0)
            
            # Apply projection
            reduced_features = np.dot(token_features, projection_matrix)
            
            print(f"Latent geometry computed using neural projection with phi-resonance weights")
        
        else:
            # No reduction, use original features
            reduced_features = token_features
            print(f"Using full {SacredConstants.DIMENSIONS}-dimensional geometry vectors (no reduction)")
        
        # Assign vectors to tokens
        for i, (token, token_id) in enumerate(self.vocab.items()):
            self.latent_vectors[token] = reduced_features[i]
        
        # Calculate latent vector for unknown token as average of all vectors
        self.latent_vectors["<UNK>"] = np.mean(list(self.latent_vectors.values()), axis=0)
        
        print(f"Computed latent geometry vectors for {len(self.latent_vectors)} tokens")
        return self.latent_vectors
    
    def get_token_latent_vector(self, token):
        """
        Get latent geometry vector for a token.
        
        Args:
            token: Token string or ID
            
        Returns:
            np.ndarray: Latent vector for the token
        """
        # Initialize latent vectors if not already done
        if not hasattr(self, 'latent_vectors') or not self.latent_vectors:
            self.compute_latent_geometry_vectors()
        
        # Convert token ID to string if needed
        if isinstance(token, int):
            token = self.inverse_vocab.get(token, "<UNK>")
        
        # Return vector if exists, otherwise return unknown vector
        if token in self.latent_vectors:
            return self.latent_vectors[token]
        else:
            return self.latent_vectors["<UNK>"]
    
    def encode_with_scores(self, text: str, apply_sacred_geometry: bool = True,
                         include_entanglement: bool = True) -> List[Tuple[int, float]]:
        """
        Encode text to token IDs with resonance scores.
        
        Args:
            text (str): The input text to encode
            apply_sacred_geometry (bool): Whether to apply sacred geometry transformations
            include_entanglement (bool): Whether to include entanglement effects in scores
            
        Returns:
            List[Tuple[int, float]]: List of (token_id, resonance_score) tuples
        """
        # First get regular token IDs
        token_ids = self.encode(text, apply_sacred_geometry)
        
        # Calculate resonance scores for each token
        token_scores = []
        for i, token_id in enumerate(token_ids):
            # Base resonance score starts at 1.0
            resonance = 1.0
            
            # Apply sacred geometry patterns
            token = self.inverse_vocab.get(token_id, "<UNK>")
            
            # Check for quantum prefixes
            for prefix in self.quantum_prefixes:
                if prefix in token.lower():
                    # Boost tokens with quantum prefixes
                    resonance *= 1.2
            
            # Check for special token types
            if token in self.special_tokens:
                # Special tokens get higher resonance
                resonance *= 1.5
            
            # Check for Tree of Life resonance
            for sephirot in SacredConstants.TREE_OF_LIFE:
                if sephirot in token.lower() or self._word_resonates_with_sephirot(token, sephirot):
                    # Tree of Life tokens get phi-modulated resonance
                    resonance *= self.phi
                    break
            
            # Apply token position modulation using Fibonacci sequence
            pos_idx = i % len(SacredConstants.FIBONACCI)
            pos_factor = SacredConstants.FIBONACCI[pos_idx] / 10.0
            resonance *= (0.5 + 0.5 * pos_factor)
            
            # Apply entanglement effects
            if include_entanglement and hasattr(self, 'entanglement_map') and self.entanglement_map:
                for j, other_id in enumerate(token_ids):
                    if i != j:
                        # Check for entanglement between this token and other tokens in sequence
                        entanglement_key = f"{token_id}:{other_id}"
                        if entanglement_key in self.entanglement_map:
                            # Apply entanglement boost based on proximity and strength
                            strength = self.entanglement_map[entanglement_key]
                            proximity = 1.0 / max(1, abs(i - j))  # Closer tokens have stronger effect
                            
                            # Add entanglement resonance
                            resonance += strength * proximity * self.phi
            
            # Normalize resonance score to a reasonable range
            resonance = max(0.1, min(2.0, resonance))
            
            token_scores.append((token_id, resonance))
        
        return token_scores
