import torch
import sys
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import time
import gc
import random

def summarize(self, gpu, model_name):
    self.device = "cuda"
    self.device = torch.device(f"cuda:{gpu}")
    self.tokenizer = AutoTokenizer.from_pretrained(model_name)
    self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(self.device)
    text = """
    Artificial intelligence (AI) is intelligence demonstrated by machines, as opposed to natural 
    intelligence displayed by animals including humans. Leading AI textbooks define the field as the 
    study of "intelligent agents": any system that perceives its environment and takes actions that 
    maximize its chance of achieving its goals. Some popular accounts use the term "artificial intelligence" 
    to describe machines that mimic "cognitive" functions that humans associate with the human mind, 
    such as "learning" and "problem solving", however this definition is rejected by major AI researchers.
    """
    
    iter_1 = random.randomint(1,7)
    iter_2 = random.randomint(2,11)
    iter_3 = random.randomint(3,15)
    
    for i in [iter_1, iter_2, iter_3]:
        for j in range(i):
            inputs = self.tokenizer(text, return_tensors="pt", max_length=1024, truncation=True)
            inputs = inputs.to(self.device)
            summary_ids = self.model.generate(inputs["input_ids"], num_beams=4, length_penalty=2.0, early_stopping=False)
            summary = self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        time.sleep(round((random.random()*10),1))
    
    del self.model
    torch.cuda.empty_cache()
    gc.collect()


if torch.cuda.is_available():
    pass
else:
    sys.exit(0)

GPU_COUNT = torch.cuda.device_count()
print(f"{ GPU_COUNT = })

gpu_free_memory = []

for i in range(GPU_COUNT):
    free = torch.cuda.mem_get_info(i)[0]
    gpu_free_memory.insert(i, free) 
    output = summarize(i, "facebook/bart-large-cnn")
    print(f"i: {gpu_free_memory = }"
    print(output)
