import os
import warnings
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor, logging

# Suppress warnings
logging.set_verbosity_error()
os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore")

PROMPT = "Please provide a detailed caption for the chart."

class Phi():
    def __init__(self, model_id='microsoft/Phi-3.5-vision-instruct', num_crops=4):
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map="cuda",
            trust_remote_code=True,
            torch_dtype="auto",
            _attn_implementation='flash_attention_2'
        )
        
        self.processor = AutoProcessor.from_pretrained(
            model_id,
            trust_remote_code=True,
            num_crops=num_crops
        )

        self.generation_args = {
        "max_new_tokens": 1000,
        "temperature": 0.0,
        "do_sample": False,
        }

    def generate_caption(self, image, prompt=PROMPT):
        messages = [{"role": "user", "content": "<|image_1|>\n" + prompt}]
        
        processed_input = self.processor.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.processor(processed_input, [image], return_tensors="pt").to("cuda:0")
        
        generate_ids = self.model.generate(
            **inputs, 
            eos_token_id=self.processor.tokenizer.eos_token_id, 
            **self.generation_args
        )
        
        # Remove input tokens and decode
        generate_ids = generate_ids[:, inputs['input_ids'].shape[1]:]
        response = self.processor.batch_decode(
            generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        
        return response

def main():
    # Model configurations
    models_config = [
        {
            "id": "microsoft/Phi-3.5-vision-instruct",
            "name": "Phi-3.5-Vision-4B",
        },
        {
            "id": "junyoung-00/Phi-3.5-vision-instruct-ChartCap",
            "name": "Phi-3.5-Vision-4B_ChartCap",
        }
    ]
    
    phi_models = []
    
    for config in models_config:
        print(f"Loading {config['name']}...")
        phi_model = Phi(model_id=config["id"])
        phi_models.append(phi_model)
    
    image = Image.open("./Phi-3.5-vision-instruct-ChartCap/example.png")
    
    print("=" * 80)
    print("ORIGINAL CAPTION:")
    print("Accuracy of GPT-4o and Claude 3.5 Sonnet on coarse-grained tasks and fine-grained tasks.")
    print("\n" + "=" * 80)
    
    # Generate captions from both models
    for i, config in enumerate(models_config):
        print(f"{config['name'].upper()} CAPTION:")
        response = phi_models[i].generate_caption(image)
        print(response)
        if i < len(models_config) - 1:
            print("\n" + "-" * 80)
    
    print("=" * 80)

if __name__ == '__main__':
    main()