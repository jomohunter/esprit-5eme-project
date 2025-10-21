from abc import ABC, abstractmethod
from typing import Any, Dict, List
from langchain.llms.base import LLM
from langchain_community.llms import Ollama
import os

class BaseAgent(ABC):
    """
    Base class for all agents in the multi-agent system
    """
    
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
        self.llm = self._initialize_llm()
    
    def _initialize_llm(self) -> LLM:
    """Initialize the language model for the agent using Ollama with Llama 3.1"""
        ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        model_name = os.getenv("OLLAMA_MODEL", "llama3.1")
        
        return Ollama(
            model=model_name,
            base_url=ollama_base_url,
            temperature=0.1
        )
    
    @abstractmethod
    async def execute(self, input_data: Any) -> Dict[str, Any]:
        """
        Execute the agent's main functionality
        
        Args:
            input_data: Input data for the agent
            
        Returns:
            Dict containing the agent's output
        """
        pass
    
    def log(self, message: str) -> None:
        """Log agent activity"""
        print(f"[{self.name}] {message}")
    
    def format_output(self, data: Any, status: str = "success") -> Dict[str, Any]:
        """Format agent output in a consistent structure"""
        return {
            "agent": self.name,
            "status": status,
            "data": data,
            "timestamp": self._get_timestamp()
        }
    
    def _get_timestamp(self) -> str:
        """Get current timestamp"""
        from datetime import datetime
        return datetime.now().isoformat()