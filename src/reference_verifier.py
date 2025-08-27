import re
import unicodedata
from typing import Tuple, Optional
from difflib import SequenceMatcher
from src.data_processor import DataProcessor

class ReferenceVerifier:
    """Verifies references against source text to prevent hallucinations"""
    
    def __init__(self, processor: DataProcessor):
        self.processor = processor
        
    def normalize_for_comparison(self, text: str, language: str = "en") -> str:
        """Normalize text for reference comparison"""
        if not text:
            return ""
            
        # Unicode normalization
        text = unicodedata.normalize('NFKC', text)
        
        # Collapse whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        if language == "en":
            text = text.lower()
        elif language == "ar":
            # Remove Arabic diacritics for better matching
            text = re.sub(r'[\u064B-\u0652\u0670\u0640]', '', text)
            
        return text
    
    def compute_token_overlap(self, reference: str, source_text: str, language: str = "en") -> float:
        """Compute token overlap ratio between reference and source"""
        ref_norm = self.normalize_for_comparison(reference, language)
        source_norm = self.normalize_for_comparison(source_text, language)
        
        ref_tokens = set(ref_norm.split())
        source_tokens = set(source_norm.split())
        
        if not ref_tokens:
            return 0.0
            
        overlap = len(ref_tokens & source_tokens)
        return overlap / len(ref_tokens)
    
    def compute_levenshtein_similarity(self, reference: str, source_text: str, language: str = "en") -> float:
        """Compute normalized Levenshtein similarity"""
        ref_norm = self.normalize_for_comparison(reference, language)
        source_norm = self.normalize_for_comparison(source_text, language)
        
        # For efficiency, limit source text length for comparison
        if len(source_norm) > 10000:
            # Find the best matching substring in source
            best_ratio = 0.0
            ref_len = len(ref_norm)
            
            for i in range(0, len(source_norm) - ref_len + 1, 100):  # Step by 100 chars
                substr = source_norm[i:i + ref_len * 2]  # Check 2x reference length
                ratio = SequenceMatcher(None, ref_norm, substr).ratio()
                best_ratio = max(best_ratio, ratio)
                
            return best_ratio
        else:
            return SequenceMatcher(None, ref_norm, source_norm).ratio()
    
    def extract_structured_reference(self, reference: str, language: str = "en") -> Optional[dict]:
        """Extract structured information from reference strings"""
        if not reference or reference.upper() == "UNKNOWN":
            return None
            
        patterns = {
            "ar": [
                r'المعيار الشرعي رقم \((\d+)\)',
                r'البند (\d+/\d+/?\d*)',
                r'الفقرة (\d+)',
                r'الصفحة (\d+)'
            ],
            "en": [
                r"Shari'ah Standard No\. \((\d+)\)",
                r"Standard No\. (\d+)",
                r"Clause (\d+/\d+/?\d*)",
                r"Paragraph (\d+)",
                r"Page (\d+)"
            ]
        }
        
        for pattern in patterns.get(language, patterns["en"]):
            match = re.search(pattern, reference, re.IGNORECASE)
            if match:
                return {
                    "type": "structured",
                    "pattern": pattern,
                    "value": match.group(1),
                    "full_match": match.group(0)
                }
        
        return {"type": "unstructured", "text": reference}
    
    def verify_reference(self, reference: str, language: str = "en", 
                        token_threshold: float = 0.75, 
                        levenshtein_threshold: float = 0.75) -> Tuple[bool, dict]:
        """
        Verify if reference exists in source text
        
        Returns:
            Tuple of (is_valid, verification_details)
        """
        if not reference or reference.strip().upper() == "UNKNOWN":
            return True, {
                "reference": "UNKNOWN",
                "suspected_fabrication": False,
                "verification_method": "unknown_reference"
            }
        
        source_text = self.processor.get_source_text(language)
        if not source_text:
            return False, {
                "reference": reference,
                "suspected_fabrication": True,
                "verification_method": "no_source_text",
                "error": "Source text not available"
            }
        
        # Step 1: Exact substring match (after normalization)
        ref_norm = self.normalize_for_comparison(reference, language)
        source_norm = self.normalize_for_comparison(source_text, language)
        
        if ref_norm in source_norm:
            return True, {
                "reference": reference,
                "suspected_fabrication": False,
                "verification_method": "exact_match"
            }
        
        # Step 2: Token overlap analysis
        token_overlap = self.compute_token_overlap(reference, source_text, language)
        
        if token_overlap >= token_threshold:
            return True, {
                "reference": reference,
                "suspected_fabrication": False,
                "verification_method": "token_overlap",
                "token_overlap_ratio": token_overlap
            }
        
        # Step 3: Levenshtein similarity for fuzzy matching
        levenshtein_sim = self.compute_levenshtein_similarity(reference, source_text, language)
        
        if levenshtein_sim >= levenshtein_threshold:
            return True, {
                "reference": reference,
                "suspected_fabrication": False,
                "verification_method": "levenshtein_similarity",
                "similarity_score": levenshtein_sim
            }
        
        # Step 4: Structured reference validation
        structured_ref = self.extract_structured_reference(reference, language)
        if structured_ref and structured_ref.get("type") == "structured":
            # For structured references, be more lenient
            if token_overlap >= 0.5:  # Lower threshold for structured refs
                return True, {
                    "reference": reference,
                    "suspected_fabrication": False,
                    "verification_method": "structured_reference_partial",
                    "token_overlap_ratio": token_overlap,
                    "structured_info": structured_ref
                }
        
        # Reference not found - likely fabricated
        return False, {
            "reference": "UNKNOWN",
            "suspected_fabrication": True,
            "verification_method": "not_found",
            "original_reference": reference,
            "token_overlap_ratio": token_overlap,
            "similarity_score": levenshtein_sim
        }
    
    def find_best_reference(self, claim: str, language: str = "en", 
                          context_chunk_id: Optional[int] = None) -> Tuple[str, dict]:
        """
        Find the best reference for a given claim
        
        Returns:
            Tuple of (reference_text, verification_details)
        """
        source_text = self.processor.get_source_text(language)
        
        # If context chunk provided, search within that chunk first
        if context_chunk_id is not None:
            chunk = self.processor.get_chunk_by_id(context_chunk_id, language)
            if chunk:
                chunk_text = chunk.get("text", "")
                
                # Simple approach: find the most similar sentence/paragraph
                claim_norm = self.normalize_for_comparison(claim, language)
                chunk_norm = self.normalize_for_comparison(chunk_text, language)
                
                # Split into sentences and find best match
                sentences = re.split(r'[.!?]\s+', chunk_norm)
                
                best_sentence = ""
                best_score = 0.0
                
                for sentence in sentences:
                    if len(sentence.strip()) < 10:  # Skip very short sentences
                        continue
                        
                    score = self.compute_token_overlap(claim, sentence, language)
                    if score > best_score:
                        best_score = score
                        best_sentence = sentence
                
                if best_score > 0.3:  # Reasonable threshold
                    return best_sentence[:200] + "..." if len(best_sentence) > 200 else best_sentence, {
                        "verification_method": "chunk_sentence_match",
                        "chunk_id": context_chunk_id,
                        "match_score": best_score
                    }
        
        # Fallback: return UNKNOWN
        return "UNKNOWN", {
            "verification_method": "no_suitable_reference",
            "suspected_fabrication": False
        }
