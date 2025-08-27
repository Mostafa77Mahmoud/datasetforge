import json
import uuid
import time
import random
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import pandas as pd
from src.gemini_client import GeminiClient
from src.reference_verifier import ReferenceVerifier
from src.data_processor import DataProcessor

class DatasetGenerator:
    """Generates judgmental verification datasets with reference validation"""
    
    def __init__(self, api_keys: List[str]):
        self.gemini_client = GeminiClient(api_keys)
        self.processor = DataProcessor()
        self.processor.load_data()
        self.verifier = ReferenceVerifier(self.processor)
        
        # Create output directories
        self._create_directories()
        
        # Generation state
        self.progress = {
            "ar": {"completed": 0, "target": 0, "true_count": 0, "false_count": 0},
            "en": {"completed": 0, "target": 0, "true_count": 0, "false_count": 0}
        }
        
    def _create_directories(self):
        """Create necessary output directories"""
        directories = [
            "data/generation_stage_B/ar",
            "data/generation_stage_B/en", 
            "logs",
            "raw",
            "progress"
        ]
        
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
    
    def _get_arabic_prompt(self, claim: str, context: str, chunk_id: int) -> str:
        """Get Arabic generation prompt"""
        return f"""أنت خبير في التحقق من معايير هيئة المحاسبة والمراجعة للمؤسسات المالية الإسلامية (أيوفي). لا تخترع المراجع.

أجب بكائن JSON واحد فقط (بدون نص إضافي) وفقاً للمخطط التالي:

المدخلات:
- الادعاء: {claim}
- معرف القطعة {chunk_id}: {context}

إذا كان بإمكانك التحقق من الادعاء من السياق، اجعل verdict: "True" و explanation عبارة عن اقتباس قصير من السياق مع مرجع دقيق.
إذا لم تتمكن من التحقق، اجعل verdict: "False" مع reference: "UNKNOWN" و suspected_fabrication: true.
إذا كان عليك التخمين، فضل "False" و reference: "UNKNOWN".

{{
  "id": "<uuid>",
  "language": "ar",
  "claim": "{claim}",
  "context_chunk_id": {chunk_id},
  "context_excerpt": "{context[:512]}",
  "verdict": "True" أو "False",
  "explanation": "<دليل مختصر مبني على الاقتباس>",
  "reference": "<مقطع متطابق بالضبط أو 'UNKNOWN'>",
  "suspected_fabrication": false أو true,
  "generator_model": "<النموذج المستخدم>",
  "raw_response_path": "raw/<filename>.json",
  "meta": {{"confidence": <0-1>, "seed_id": "<seed>"}}
}}"""

    def _get_english_prompt(self, claim: str, context: str, chunk_id: int) -> str:
        """Get English generation prompt"""
        return f"""You are an expert AAOIFI verifier (English). DO NOT INVENT REFERENCES.

Respond with EXACTLY one JSON object (no prose) following the schema below.

Inputs:
- Seed claim: {claim}
- Context chunk id {chunk_id}: {context}

If you can verify the claim from the context, set verdict: "True" and explanation to a short quote from the context and an exact reference phrase.
If you cannot verify, set verdict: "False" (if claim false) or verdict:"False" with reference:"UNKNOWN" and suspected_fabrication:true.

{{
  "id": "<uuid>",
  "language": "en", 
  "claim": "{claim}",
  "context_chunk_id": {chunk_id},
  "context_excerpt": "{context[:512]}",
  "verdict": "True" or "False",
  "explanation": "<concise quote-based evidence>",
  "reference": "<exact matched snippet OR 'UNKNOWN'>",
  "suspected_fabrication": false or true,
  "generator_model": "<model-used>",
  "raw_response_path": "raw/<filename>.json",
  "meta": {{"confidence": <0-1>, "seed_id": "<seed>"}}
}}"""
    
    def _generate_perturbations(self, original_claim: str, language: str) -> List[str]:
        """Generate perturbed versions of claim for FALSE examples"""
        perturbations = []
        
        if language == "ar":
            # Arabic perturbations
            if "المعيار" in original_claim and "رقم" in original_claim:
                # Wrong standard number
                import re
                match = re.search(r'رقم \((\d+)\)', original_claim)
                if match:
                    orig_num = int(match.group(1))
                    new_num = orig_num + 1 if orig_num < 50 else orig_num - 1
                    perturbed = original_claim.replace(f'رقم ({orig_num})', f'رقم ({new_num})')
                    perturbations.append(perturbed)
            
            # Polarity flip
            if "يجوز" in original_claim:
                perturbations.append(original_claim.replace("يجوز", "لا يجوز"))
            elif "لا يجوز" in original_claim:
                perturbations.append(original_claim.replace("لا يجوز", "يجوز"))
                
        else:
            # English perturbations
            if "Standard No." in original_claim:
                # Wrong standard number
                import re
                match = re.search(r'Standard No\. \((\d+)\)', original_claim)
                if match:
                    orig_num = int(match.group(1))
                    new_num = orig_num + 1 if orig_num < 50 else orig_num - 1
                    perturbed = original_claim.replace(f'Standard No. ({orig_num})', f'Standard No. ({new_num})')
                    perturbations.append(perturbed)
            
            # Polarity flip
            if "permissible" in original_claim:
                perturbations.append(original_claim.replace("permissible", "prohibited"))
            elif "prohibited" in original_claim:
                perturbations.append(original_claim.replace("prohibited", "permissible"))
        
        return perturbations[:2]  # Limit to 2 perturbations
    
    def _parse_json_response(self, response_text: str) -> Optional[Dict]:
        """Parse JSON response from model, handling various formats"""
        if not response_text:
            return None
            
        try:
            # Try direct JSON parse
            return json.loads(response_text.strip())
        except json.JSONDecodeError:
            # Try to extract JSON from text
            import re
            
            # Look for JSON object pattern
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                try:
                    return json.loads(json_match.group(0))
                except json.JSONDecodeError:
                    pass
            
            return None
    
    def _validate_example_schema(self, example: Dict) -> Tuple[bool, List[str]]:
        """Validate example against required schema"""
        required_fields = [
            "id", "language", "claim", "context_chunk_id", "context_excerpt",
            "verdict", "explanation", "reference", "suspected_fabrication",
            "generator_model", "meta"
        ]
        
        errors = []
        
        for field in required_fields:
            if field not in example:
                errors.append(f"Missing field: {field}")
        
        # Validate specific field types
        if "verdict" in example and example["verdict"] not in ["True", "False"]:
            errors.append("Invalid verdict value")
            
        if "language" in example and example["language"] not in ["ar", "en"]:
            errors.append("Invalid language value")
            
        if "suspected_fabrication" in example and not isinstance(example["suspected_fabrication"], bool):
            errors.append("suspected_fabrication must be boolean")
        
        return len(errors) == 0, errors
    
    def _generate_single_example(self, seed_qa: Dict, language: str, 
                                is_true_example: bool = True) -> Optional[Dict]:
        """Generate a single verification example"""
        chunk_id = seed_qa.get("chunk_id", 0)
        context = self.processor.extract_context_excerpt(chunk_id, language, 512)
        
        if is_true_example:
            claim = seed_qa.get("answer", "")
        else:
            # Generate perturbed claim
            original_claim = seed_qa.get("answer", "")
            perturbations = self._generate_perturbations(original_claim, language)
            claim = perturbations[0] if perturbations else original_claim + " (modified)"
        
        # Get appropriate prompt
        if language == "ar":
            prompt = self._get_arabic_prompt(claim, context, chunk_id)
            model = "gemini-2.5-flash"
        else:
            prompt = self._get_english_prompt(claim, context, chunk_id)
            model = "gemini-2.5-pro"
        
        # Generate content
        response_text, metadata = self.gemini_client.generate_content(prompt, model)
        
        if not response_text:
            return None
        
        # Parse response
        example = self._parse_json_response(response_text)
        if not example:
            return None
        
        # Validate schema
        is_valid, errors = self._validate_example_schema(example)
        if not is_valid:
            print(f"Schema validation errors: {errors}")
            return None
        
        # Verify reference
        reference = example.get("reference", "")
        is_ref_valid, ref_details = self.verifier.verify_reference(reference, language)
        
        # Update example with verification results
        example.update({
            "id": str(uuid.uuid4()),
            "language": language,
            "claim": claim,
            "context_chunk_id": chunk_id,
            "context_excerpt": context,
            "generator_model": model,
            "raw_response_path": metadata.get("raw_response_path", ""),
            "meta": {
                "confidence": metadata.get("confidence", 0.8),
                "seed_id": seed_qa.get("id", ""),
                **ref_details
            }
        })
        
        # Override reference if fabricated
        if ref_details.get("suspected_fabrication"):
            example["reference"] = "UNKNOWN"
            example["suspected_fabrication"] = True
        
        return example
    
    def run_smoke_test(self, language: str, target_count: int = 20) -> Dict:
        """Run smoke test with small sample"""
        try:
            print(f"Running smoke test for {language} with {target_count} examples...")
            
            qa_pairs = self.processor.arabic_qa_pairs if language == "ar" else self.processor.english_qa_pairs
            if not qa_pairs:
                return {"success": False, "error": "No QA pairs available"}
            
            examples = []
            true_count = 0
            false_count = 0
            failed_attempts = 0
            max_failures = target_count  # Allow up to target_count failures
            
            # Generate examples
            for i in range(target_count):
                if failed_attempts >= max_failures:
                    print(f"Too many failures ({failed_attempts}), stopping early")
                    break
                    
                seed_qa = random.choice(qa_pairs)
                
                # Alternate between True and False examples
                is_true = (true_count < target_count // 2) or (false_count >= target_count // 2)
                
                try:
                    example = self._generate_single_example(seed_qa, language, is_true)
                    if example:
                        examples.append(example)
                        
                        if example.get("verdict") == "True":
                            true_count += 1
                        else:
                            false_count += 1
                        
                        print(f"Generated example {len(examples)}/{target_count}")
                    else:
                        failed_attempts += 1
                        print(f"Failed to generate example {i+1} (attempt {failed_attempts})")
                        
                except Exception as e:
                    failed_attempts += 1
                    print(f"Exception generating example {i+1}: {str(e)}")
                
                # Rate limiting with exponential backoff if many failures
                sleep_time = 2 if failed_attempts > 5 else 1
                time.sleep(sleep_time)
            
            if not examples:
                return {
                    "success": False, 
                    "error": f"No examples generated. Failed attempts: {failed_attempts}",
                    "stats": {"total": 0, "true": 0, "false": 0, "failed_attempts": failed_attempts}
                }
            
            # Save results
            output_file = f"data/generation_stage_B/{language}/preview_{language}_{len(examples)}.jsonl"
            try:
                with open(output_file, 'w', encoding='utf-8') as f:
                    for example in examples:
                        f.write(json.dumps(example, ensure_ascii=False) + '\n')
            except Exception as e:
                print(f"Failed to save results: {str(e)}")
                output_file = "failed_to_save"
            
            # Calculate stats
            total = len(examples)
            fabrication_count = sum(1 for ex in examples if ex.get("suspected_fabrication"))
            fabrication_rate = fabrication_count / total if total > 0 else 0
            
            stats = {
                "total": total,
                "true": true_count,
                "false": false_count,
                "fabrication_count": fabrication_count,
                "fabrication_rate": fabrication_rate,
                "balance": abs(true_count - false_count) / max(1, total),
                "failed_attempts": failed_attempts,
                "success_rate": total / (total + failed_attempts) if (total + failed_attempts) > 0 else 0
            }
            
            # Check success criteria (more lenient for smoke test)
            success = (
                total >= min(10, target_count * 0.5) and  # At least 10 or 50% completion
                fabrication_rate <= 0.1 and  # Max 10% fabrication for smoke test
                stats["success_rate"] >= 0.3  # At least 30% success rate
            )
            
            return {
                "success": success,
                "stats": stats,
                "samples": examples[:3],  # First 3 examples for review
                "output_file": output_file,
                "message": f"Generated {total} examples with {failed_attempts} failures"
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": f"Smoke test failed with exception: {str(e)}",
                "stats": {"total": 0, "true": 0, "false": 0}
            }
    
    def generate_full_dataset(self, language: str, target_count: int, 
                             progress_callback=None) -> Dict:
        """Generate full dataset for specified language"""
        print(f"Generating full dataset for {language} with {target_count} examples...")
        
        qa_pairs = self.processor.arabic_qa_pairs if language == "ar" else self.processor.english_qa_pairs
        if not qa_pairs:
            return {"success": False, "error": "No QA pairs available"}
        
        examples = []
        true_count = 0
        false_count = 0
        batch_size = 50
        
        self.progress[language]["target"] = target_count
        
        try:
            while len(examples) < target_count:
                # Determine if we need more True or False examples
                need_true = true_count < target_count * 0.5
                need_false = false_count < target_count * 0.5
                
                if not need_true and not need_false:
                    # Balanced, choose randomly
                    is_true = random.choice([True, False])
                elif need_true:
                    is_true = True
                else:
                    is_true = False
                
                # Generate example
                seed_qa = random.choice(qa_pairs)
                example = self._generate_single_example(seed_qa, language, is_true)
                
                if example:
                    examples.append(example)
                    
                    if example.get("verdict") == "True":
                        true_count += 1
                    else:
                        false_count += 1
                    
                    # Update progress
                    self.progress[language]["completed"] = len(examples)
                    self.progress[language]["true_count"] = true_count
                    self.progress[language]["false_count"] = false_count
                    
                    # Save checkpoint every batch_size examples
                    if len(examples) % batch_size == 0:
                        checkpoint_file = f"data/generation_stage_B/{language}/checkpoint_{len(examples)}.jsonl"
                        with open(checkpoint_file, 'w', encoding='utf-8') as f:
                            for ex in examples:
                                f.write(json.dumps(ex, ensure_ascii=False) + '\n')
                        
                        # Save progress
                        self._save_progress()
                        
                        print(f"Checkpoint: {len(examples)}/{target_count} examples generated")
                        
                        if progress_callback:
                            progress_callback(len(examples) / target_count)
                
                time.sleep(0.5)  # Rate limiting
        
        except Exception as e:
            print(f"Error during generation: {str(e)}")
            return {"success": False, "error": str(e)}
        
        # Save final dataset
        final_file = f"data/generation_stage_B/{language}/judgmental_{language}_final.jsonl"
        with open(final_file, 'w', encoding='utf-8') as f:
            for example in examples:
                f.write(json.dumps(example, ensure_ascii=False) + '\n')
        
        # Generate splits
        self._generate_splits(examples, language)
        
        # Calculate final stats
        total = len(examples)
        fabrication_count = sum(1 for ex in examples if ex.get("suspected_fabrication"))
        fabrication_rate = fabrication_count / total if total > 0 else 0
        
        stats = {
            "total": total,
            "true": true_count,
            "false": false_count,
            "fabrication_count": fabrication_count,
            "fabrication_rate": fabrication_rate,
            "balance": abs(true_count - false_count) / max(1, total)
        }
        
        return {"success": True, "stats": stats, "output_file": final_file}
    
    def _generate_splits(self, examples: List[Dict], language: str):
        """Generate train/validation/test splits"""
        random.shuffle(examples)
        
        total = len(examples)
        train_size = int(total * 0.8)
        val_size = int(total * 0.1)
        
        train_examples = examples[:train_size]
        val_examples = examples[train_size:train_size + val_size]
        test_examples = examples[train_size + val_size:]
        
        # Save splits
        base_path = f"data/generation_stage_B/{language}"
        
        with open(f"{base_path}/train.jsonl", 'w', encoding='utf-8') as f:
            for ex in train_examples:
                f.write(json.dumps(ex, ensure_ascii=False) + '\n')
        
        with open(f"{base_path}/val.jsonl", 'w', encoding='utf-8') as f:
            for ex in val_examples:
                f.write(json.dumps(ex, ensure_ascii=False) + '\n')
        
        with open(f"{base_path}/test.jsonl", 'w', encoding='utf-8') as f:
            for ex in test_examples:
                f.write(json.dumps(ex, ensure_ascii=False) + '\n')
    
    def _save_progress(self):
        """Save current progress to file"""
        progress_file = "progress/state.json"
        progress_data = {
            "timestamp": int(time.time()),
            "progress": self.progress,
            "key_status": self.gemini_client.get_key_status()
        }
        
        with open(progress_file, 'w', encoding='utf-8') as f:
            json.dump(progress_data, f, indent=2)
