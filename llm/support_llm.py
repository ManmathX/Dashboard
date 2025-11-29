"""
Support LLM system for multi-model comparison (optional).
"""

from typing import List, Dict, Any
from llm.target_llm import TargetLLM


class SupportLLMSystem:
    """
    Manages multiple support LLMs for comparison and consensus.
    Optional component for reducing hallucination bias.
    """
    
    def __init__(self, model_configs: List[Dict[str, str]]):
        """
        Initialize support LLM system.
        
        Args:
            model_configs: List of dicts with 'provider' and 'model_name'
                Example: [
                    {"provider": "openai", "model_name": "gpt-4"},
                    {"provider": "anthropic", "model_name": "claude-3-opus-20240229"}
                ]
        """
        self.llms = []
        for config in model_configs:
            llm = TargetLLM(
                provider=config["provider"],
                model_name=config["model_name"]
            )
            self.llms.append({
                "llm": llm,
                "name": f"{config['provider']}:{config['model_name']}"
            })
    
    async def generate_all(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 2000
    ) -> List[Dict[str, Any]]:
        """
        Generate outputs from all support LLMs.
        
        Args:
            prompt: User prompt
            temperature: Sampling temperature
            max_tokens: Maximum tokens
        
        Returns:
            List of outputs from each model
        """
        import asyncio
        
        tasks = []
        for llm_config in self.llms:
            task = llm_config["llm"].generate(prompt, temperature, max_tokens)
            tasks.append(task)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        outputs = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                # Handle errors gracefully
                outputs.append({
                    "model_name": self.llms[i]["name"],
                    "output": f"[Error: {str(result)}]",
                    "error": True
                })
            else:
                outputs.append({
                    "model_name": self.llms[i]["name"],
                    "output": result["output"],
                    "tokens": result["tokens"],
                    "latency": result["latency"],
                    "error": False
                })
        
        return outputs
    
    def compute_consensus(self, outputs: List[str]) -> Dict[str, Any]:
        """
        Compute consensus among multiple outputs.
        
        Args:
            outputs: List of output strings from different models
        
        Returns:
            Dict with consensus information
        """
        # Simple consensus: check for common themes
        # In production, you might use more sophisticated NLP techniques
        
        # For now, just return basic statistics
        return {
            "total_models": len(outputs),
            "avg_length": sum(len(o) for o in outputs) / len(outputs) if outputs else 0,
            "outputs": outputs
        }
