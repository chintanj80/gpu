from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from typing import Optional

class Llama2Generator:
    def __init__(self, model_name: str = "meta-llama/Llama-2-7b-chat-hf"):
        """
        Initialize Llama 2 with 8-bit quantization
        Args:
            model_name: Name/path of the model to load
        """
        print("Loading model and tokenizer...")
        
        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Load model in 8-bit with half precision
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",  # Automatically choose best device
            load_in_8bit=True,  # Enable 8-bit quantization
            torch_dtype=torch.float16,  # Use half precision
            trust_remote_code=True
        )
        
        print("Model loaded successfully!")
    
    def generate(
        self,
        prompt: str,
        max_length: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.95,
        top_k: int = 50,
        num_return_sequences: int = 1,
        stop_sequence: Optional[str] = None
    ) -> str:
        """
        Generate text based on the prompt
        Args:
            prompt: Input text to generate from
            max_length: Maximum length of generated text
            temperature: Randomness in generation (higher = more random)
            top_p: Nucleus sampling parameter
            top_k: Top-k sampling parameter
            num_return_sequences: Number of sequences to generate
            stop_sequence: Optional sequence to stop generation
        Returns:
            Generated text
        """
        # Encode the prompt
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        
        # Generate
        outputs = self.model.generate(
            **inputs,
            max_length=max_length,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            num_return_sequences=num_return_sequences,
            pad_token_id=self.tokenizer.eos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            do_sample=True
        )
        
        # Decode the generated text
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Handle stop sequence if provided
        if stop_sequence and stop_sequence in generated_text:
            generated_text = generated_text[:generated_text.index(stop_sequence)]
            
        return generated_text

def main():
    # Initialize generator
    generator = Llama2Generator()
    
    # Example prompts
    prompts = [
        "Write a short story about a robot learning to paint:",
        "Explain quantum computing to a 10-year old:",
        "Create a recipe for a healthy breakfast smoothie:"
    ]
    
    # Generate text for each prompt
    for prompt in prompts:
        print("\nPrompt:", prompt)
        print("\nGenerated Response:")
        response = generator.generate(
            prompt,
            max_length=256,
            temperature=0.7
        )
        print(response)
        print("\n" + "="*50)

if __name__ == "__main__":
    main()