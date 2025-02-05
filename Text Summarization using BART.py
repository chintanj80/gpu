from transformers import BartForConditionalGeneration, BartTokenizer
import torch

class TextSummarizer:
    def __init__(self, model_name="facebook/bart-large-cnn"):
        """
        Initialize the BART summarizer with a pre-trained model
        
        Args:
            model_name (str): Name of the pre-trained model to use
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = BartTokenizer.from_pretrained(model_name)
        self.model = BartForConditionalGeneration.from_pretrained(model_name).to(self.device)
    
    def summarize(self, text, max_length=130, min_length=30, length_penalty=2.0, num_beams=4):
        """
        Generate a summary for the input text
        
        Args:
            text (str): The text to summarize
            max_length (int): Maximum length of the summary
            min_length (int): Minimum length of the summary
            length_penalty (float): Length penalty for beam search
            num_beams (int): Number of beams for beam search
            
        Returns:
            str: The generated summary
        """
        # Tokenize the input text
        inputs = self.tokenizer(text, max_length=1024, truncation=True, return_tensors="pt")
        inputs = inputs.to(self.device)
        
        # Generate summary
        summary_ids = self.model.generate(
            inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_length=max_length,
            min_length=min_length,
            length_penalty=length_penalty,
            num_beams=num_beams,
            early_stopping=True
        )
        
        # Decode the generated summary
        summary = self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        return summary

def main():
    # Example usage
    summarizer = TextSummarizer()
    
    # Example text to summarize
    text = """
    Artificial intelligence (AI) is intelligence demonstrated by machines, as opposed to natural intelligence displayed by animals including humans. Leading AI textbooks define the field as the study of "intelligent agents": any system that perceives its environment and takes actions that maximize its chance of achieving its goals. Some popular accounts use the term "artificial intelligence" to describe machines that mimic "cognitive" functions that humans associate with the human mind, such as "learning" and "problem solving", however this definition is rejected by major AI researchers.
    AI applications include advanced web search engines, recommendation systems, understanding human speech, self-driving cars, automated decision-making and competing at the highest level in strategic game systems. As machines become increasingly capable, tasks considered to require "intelligence" are often removed from the definition of AI, a phenomenon known as the AI effect. For instance, optical character recognition is frequently excluded from things considered to be AI, having become a routine technology.
    """
    
    # Generate summary
    summary = summarizer.summarize(text)
    print("Original Text:\n", text)
    print("\nSummary:\n", summary)

if __name__ == "__main__":
    main()