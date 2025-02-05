from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch
from typing import Literal

class Llama2Loader:
    def __init__(self, model_name: str = "meta-llama/Llama-2-7b-chat-hf"):
        """
        Initialize loader for Llama 2 model
        Args:
            model_name: Name/path of the model to load
        """
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
    def load_8bit(self):
        """
        Load model in 8-bit quantization
        Memory usage: ~14GB
        """
        print("Loading model in 8-bit quantization...")
        
        model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            device_map="auto",
            load_in_8bit=True,
            torch_dtype=torch.float16,
            trust_remote_code=True
        )
        
        print("Model loaded successfully in 8-bit!")
        return model
    
    def load_4bit(self, use_nested_quant: bool = False):
        """
        Load model in 4-bit quantization
        Args:
            use_nested_quant: Whether to use nested quantization for further memory savings
        Memory usage: ~7GB (or ~6GB with nested quantization)
        """
        print("Loading model in 4-bit quantization...")
        
        # Configure 4-bit quantization
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=use_nested_quant,
            bnb_4bit_quant_type="nf4"  # Normal Float 4 for better accuracy
        )
        
        model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            device_map="auto",
            quantization_config=quantization_config,
            trust_remote_code=True
        )
        
        print("Model loaded successfully in 4-bit!")
        return model
    
    def load_model(self, quantization: Literal["4bit", "4bit_nested", "8bit"]):
        """
        Load model with specified quantization
        Args:
            quantization: Type of quantization to use
        Returns:
            Loaded model
        """
        if quantization == "8bit":
            return self.load_8bit()
        elif quantization == "4bit":
            return self.load_4bit(use_nested_quant=False)
        elif quantization == "4bit_nested":
            return self.load_4bit(use_nested_quant=True)
        else:
            raise ValueError("Invalid quantization type")

def main():
    # Initialize loader
    loader = Llama2Loader()
    
    # Example of loading with different quantization methods
    
    # 1. 8-bit quantization (~14GB)
    try:
        model_8bit = loader.load_model("8bit")
        print("8-bit model loaded successfully")
        del model_8bit  # Free memory
        torch.cuda.empty_cache()
    except Exception as e:
        print(f"Error loading 8-bit model: {e}")
    
    # 2. 4-bit quantization (~7GB)
    try:
        model_4bit = loader.load_model("4bit")
        print("4-bit model loaded successfully")
        del model_4bit  # Free memory
        torch.cuda.empty_cache()
    except Exception as e:
        print(f"Error loading 4-bit model: {e}")
    
    # 3. 4-bit nested quantization (~6GB)
    try:
        model_4bit_nested = loader.load_model("4bit_nested")
        print("4-bit nested quantization model loaded successfully")
    except Exception as e:
        print(f"Error loading 4-bit nested model: {e}")

if __name__ == "__main__":
    main()