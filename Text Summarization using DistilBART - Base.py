from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

class LightweightSummarizer:
    def __init__(self, model_name="sshleifer/distilbart-cnn-6-6"):
        """
        Initialize a lightweight summarizer using DistilBART
        Args:
            model_name: The model to use for summarization
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")
        
        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(self.device)
        
        # Print model size info
        model_size = sum(p.numel() for p in self.model.parameters()) * 4 / (1024 * 1024)  # Size in MB
        print(f"Model size: {model_size:.2f} MB")
    
    def summarize(self, text, max_length=130, min_length=30):
        """
        Generate a summary of the input text
        Args:
            text: Text to summarize
            max_length: Maximum length of the summary
            min_length: Minimum length of the summary
        Returns:
            str: Generated summary
        """
        # Tokenize
        inputs = self.tokenizer(text, return_tensors="pt", max_length=1024, truncation=True)
        inputs = inputs.to(self.device)
        
        # Generate summary
        summary_ids = self.model.generate(
            inputs["input_ids"],
            num_beams=4,
            max_length=max_length,
            min_length=min_length,
            length_penalty=2.0,
            early_stopping=True
        )
        
        # Decode summary
        return self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)

def main():
    # Initialize summarizer
    summarizer = LightweightSummarizer()
    
    # Example text
    text = """
    Artificial intelligence (AI) is intelligence demonstrated by machines, as opposed to natural 
    intelligence displayed by animals including humans. Leading AI textbooks define the field as the 
    study of "intelligent agents": any system that perceives its environment and takes actions that 
    maximize its chance of achieving its goals. Some popular accounts use the term "artificial intelligence" 
    to describe machines that mimic "cognitive" functions that humans associate with the human mind, 
    such as "learning" and "problem solving", however this definition is rejected by major AI researchers.
    """
    
    # Generate summary
    summary = summarizer.summarize(text)
    print("\nOriginal text length:", len(text))
    print("\nSummary:", summary)
    print("Summary length:", len(summary))

if __name__ == "__main__":
    main()