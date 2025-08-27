import os
import time
import json
import logging
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import uuid
from google import genai
from google.genai import types

class GeminiClient:
    """Handles Gemini API interactions with key rotation and quota management"""
    
    def __init__(self, api_keys: List[str]):
        self.api_keys = api_keys
        self.current_key_index = 0
        self.blocked_keys = {}  # key_index -> unblock_time
        self.request_counts = {}  # key_index -> (count, window_start)
        self.rate_limit = 12  # requests per minute per key
        self.block_duration = 300  # 5 minutes
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Create directories
        Path("logs").mkdir(exist_ok=True)
        Path("raw").mkdir(exist_ok=True)
        Path("progress").mkdir(exist_ok=True)
        
    def get_current_client(self) -> Optional[genai.Client]:
        """Get current active Gemini client"""
        current_time = time.time()
        
        # Remove expired blocks
        self.blocked_keys = {
            k: v for k, v in self.blocked_keys.items() 
            if v > current_time
        }
        
        # Find next available key
        attempts = 0
        while attempts < len(self.api_keys):
            if self.current_key_index not in self.blocked_keys:
                # Check rate limit
                if self._check_rate_limit(self.current_key_index):
                    api_key = self.api_keys[self.current_key_index]
                    return genai.Client(api_key=api_key)
            
            # Move to next key
            self.current_key_index = (self.current_key_index + 1) % len(self.api_keys)
            attempts += 1
        
        return None
    
    def _check_rate_limit(self, key_index: int) -> bool:
        """Check if key is within rate limit"""
        current_time = time.time()
        window_start = current_time - 60  # 1 minute window
        
        if key_index not in self.request_counts:
            self.request_counts[key_index] = (0, current_time)
            return True
        
        count, last_window = self.request_counts[key_index]
        
        # Reset if window expired
        if last_window < window_start:
            self.request_counts[key_index] = (0, current_time)
            return True
        
        return count < self.rate_limit
    
    def _record_request(self, key_index: int):
        """Record a request for rate limiting"""
        current_time = time.time()
        
        if key_index in self.request_counts:
            count, _ = self.request_counts[key_index]
            self.request_counts[key_index] = (count + 1, current_time)
        else:
            self.request_counts[key_index] = (1, current_time)
    
    def _block_key(self, key_index: int):
        """Block a key for the specified duration"""
        self.blocked_keys[key_index] = time.time() + self.block_duration
        self.logger.warning(f"Blocked key {key_index} for {self.block_duration} seconds")
    
    def _save_raw_response(self, response_data: dict) -> str:
        """Save raw response to file and return path"""
        timestamp = int(time.time())
        filename = f"raw/response_{timestamp}_{uuid.uuid4().hex[:8]}.json"
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(response_data, f, ensure_ascii=False, indent=2)
        
        return filename
    
    def generate_content(self, prompt: str, model: str = "gemini-2.5-flash", 
                        max_retries: int = 3) -> Tuple[Optional[str], dict]:
        """
        Generate content with retry logic and error handling
        
        Returns:
            Tuple of (response_text, metadata)
        """
        for attempt in range(max_retries):
            client = self.get_current_client()
            
            if not client:
                self.logger.error("No available API keys")
                return None, {
                    "error": "no_available_keys",
                    "attempt": attempt + 1,
                    "blocked_keys": list(self.blocked_keys.keys())
                }
            
            try:
                start_time = time.time()
                
                # Record request for rate limiting
                self._record_request(self.current_key_index)
                
                # Make API call
                response = client.models.generate_content(
                    model=model,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        temperature=0.7,
                        max_output_tokens=1024
                    )
                )
                
                end_time = time.time()
                
                # Prepare response metadata
                metadata = {
                    "model": model,
                    "key_index": self.current_key_index,
                    "attempt": attempt + 1,
                    "latency": end_time - start_time,
                    "success": True
                }
                
                # Save raw response
                raw_data = {
                    "prompt": prompt,
                    "response": response.text if response.text else "",
                    "metadata": metadata,
                    "timestamp": int(time.time())
                }
                
                raw_path = self._save_raw_response(raw_data)
                metadata["raw_response_path"] = raw_path
                
                return response.text, metadata
                
            except Exception as e:
                error_msg = str(e)
                self.logger.warning(f"API call failed (attempt {attempt + 1}): {error_msg}")
                
                # Handle specific error types
                if "429" in error_msg or "RESOURCE_EXHAUSTED" in error_msg or "quota" in error_msg.lower():
                    self._block_key(self.current_key_index)
                    # Try next key immediately
                    self.current_key_index = (self.current_key_index + 1) % len(self.api_keys)
                    # Short delay before trying next key
                    time.sleep(1)
                    continue
                elif "5" in error_msg[:3] or "server" in error_msg.lower():  # 5xx errors or server issues
                    # Exponential backoff for server errors
                    wait_time = min(2 ** attempt, 60)  # Shorter max wait
                    self.logger.info(f"Server error, waiting {wait_time}s before retry")
                    time.sleep(wait_time)
                    continue
                else:
                    # Other errors - but still retry with different key
                    self.current_key_index = (self.current_key_index + 1) % len(self.api_keys)
                    time.sleep(2)  # Wait before retry
                    if attempt < max_retries - 1:  # Don't continue on last attempt
                        continue
                    
                    return None, {
                        "error": error_msg,
                        "attempt": attempt + 1,
                        "key_index": self.current_key_index,
                        "model": model
                    }
        
        # All retries exhausted
        return None, {
            "error": "max_retries_exhausted",
            "attempts": max_retries,
            "model": model
        }
    
    def get_key_status(self) -> Dict:
        """Get status of all API keys"""
        current_time = time.time()
        
        status = {
            "total_keys": len(self.api_keys),
            "current_key_index": self.current_key_index,
            "blocked_keys": {},
            "rate_limits": {},
            "available_keys": 0
        }
        
        for i in range(len(self.api_keys)):
            # Check if blocked
            if i in self.blocked_keys:
                unblock_time = self.blocked_keys[i]
                status["blocked_keys"][i] = {
                    "unblock_time": unblock_time,
                    "remaining_seconds": max(0, int(unblock_time - current_time))
                }
            else:
                status["available_keys"] += 1
            
            # Check rate limit status
            if i in self.request_counts:
                count, window_start = self.request_counts[i]
                status["rate_limits"][i] = {
                    "requests_in_window": count,
                    "limit": self.rate_limit,
                    "window_start": window_start
                }
        
        return status
