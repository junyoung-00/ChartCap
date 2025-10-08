import argparse
import json
import os
import re
import traceback
import multiprocessing
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
from anthropic import Anthropic
from transformers import AutoModel, AutoProcessor
from paddleocr import PaddleOCR

# Configuration
MAX_DEBUG_ATTEMPTS = 3
CODE_TIMEOUT = 20
SIGLIP_MODEL = "google/siglip2-so400m-patch16-512"
CLAUDE_MODEL = "claude-3-5-sonnet-20241022"
TEMP_DIR = "./temp_charts"

# Prompts
CODE_GEN_SYSTEM = (
    "You are an expert in Python and the Matplotlib library. "
    "Your task is to generate a complete Python script that precisely "
    "reflects every detail in the given chart description, without making any guesses."
)

CODE_GEN_USER = (
    'Generate accurate Python code using Matplotlib library strictly based on the given description about a chart.\n'
    'If the description lacks details about required chart components or data points, omit them from the code instead of making assumptions, '
    'but ensure that every detail in the description is included.\n'
    'Instead of using numpy\'s sin, cos, or exp function, manually define data points to represent the chart if needed.\n'
    'Labels are elements that display and specify data points in the chart. They are different from axis labels (titles).\n\n'
    '[Description]\n"{caption}"\n\n'
    'Respond only the generated code.\nCode:'
)

DEBUG_PROMPT = (
    '[Erroneous Code]\n{code}\n\n'
    '[Error Message]\n{error_msg}\n\n'
    'Analyze the provided error message and fix the code accordingly. '
    'Make only the necessary changes to resolve the error while keeping all correctly functioning attributes unchanged. '
    'Return only the corrected code without any explanations or additional output.\n'
    'Corrected Code:'
)

# Global variables for models (initialized once)
siglip_model = None
siglip_processor = None
ocr_model = None
anthropic_client = None

def initialize_models():
    """Initialize all models once at the start"""
    global siglip_model, siglip_processor, ocr_model, anthropic_client
    
    print("Initializing models...")
    
    # Initialize SIGLIP2 model
    siglip_model = AutoModel.from_pretrained(
        SIGLIP_MODEL,
        attn_implementation="flash_attention_2",
        torch_dtype=torch.float16,
        device_map="cuda",
    ).eval()
    siglip_processor = AutoProcessor.from_pretrained(SIGLIP_MODEL)
    
    # Initialize OCR model
    ocr_model = PaddleOCR(use_angle_cls=True, lang="en", show_log=False, use_gpu=True)
    
    # Initialize Anthropic client
    anthropic_client = Anthropic()
    
    print("Models initialized successfully!")

def generate_code_from_caption(caption: str) -> str:
    """Generate matplotlib code from a caption using Claude"""
    response = anthropic_client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=2048,
        system=[{
            "type": "text",
            "text": CODE_GEN_SYSTEM,
            "cache_control": {"type": "ephemeral"}
        }],
        messages=[{
            "role": "user",
            "content": [{
                "type": "text",
                "text": CODE_GEN_USER.format(caption=caption),
                "cache_control": {"type": "ephemeral"}
            }]
        }],
        temperature=0.0,
    )
    return response.content[0].text

def debug_code(code: str, error_msg: str) -> str:
    """Fix erroneous code based on error message using Claude"""
    response = anthropic_client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=2048,
        system=[{
            "type": "text",
            "text": "You are an expert in Python and the Matplotlib library. Your task is to fix the code based on the provided error message.",
            "cache_control": {"type": "ephemeral"}
        }],
        messages=[{
            "role": "user",
            "content": [{
                "type": "text",
                "text": DEBUG_PROMPT.format(code=code, error_msg=error_msg),
                "cache_control": {"type": "ephemeral"}
            }]
        }],
        temperature=0.0,
    )
    return response.content[0].text

def _worker_code(image_filename, code_string, temp_directory, error_queue):
    """Worker process for executing matplotlib code"""
    try:
        temp_directory_ocr = f'{temp_directory}_ocr'
        os.makedirs(temp_directory, exist_ok=True)
        os.makedirs(temp_directory_ocr, exist_ok=True)

        output_path = os.path.join(temp_directory, image_filename)
        output_path_ocr = os.path.join(temp_directory_ocr, image_filename)

        # Clean code
        if code_string.startswith("```python") and code_string.endswith("```"):
            code_string = code_string[9:-3].strip()
        elif code_string.startswith("```") and code_string.endswith("```"):
            code_string = code_string[3:-3].strip()

        plt.switch_backend('Agg')

        # Prepare regular code
        if "plt.show()" in code_string:
            code_string = code_string.replace("plt.show()", f"plt.savefig('{output_path}')")
        else:
            code_string += f"\nplt.savefig('{output_path}')"
            
        if "plt.grid(True)" in code_string:
            code_string = code_string.replace("plt.grid(True)", "")
        if "plt.tight_layout()" not in code_string:
            code_string = re.sub(r'plt\.savefig\(', r'plt.tight_layout()\nplt.savefig(', code_string)

        code_string = re.sub(r'padding\s*=\s*-?\d*\.?\d+', '', code_string)

        # Create OCR version
        code_string_ocr = code_string.replace(f"plt.savefig('{output_path}')", f"plt.savefig('{output_path_ocr}')")
        code_string_ocr = re.sub(r'rotation\s*=\s*-?\d+(?:\.\d+)?', 'rotation=0', code_string_ocr)

        # Execute regular code
        exec_globals = {"plt": plt}
        exec("import numpy as np", exec_globals)
        exec("import pandas as pd", exec_globals)
        exec("from matplotlib import pyplot as plt", exec_globals)
        exec(code_string, exec_globals)
        plt.close('all')

        # Execute OCR code
        exec_globals = {"plt": plt}
        exec("import numpy as np", exec_globals)
        exec("import pandas as pd", exec_globals)
        exec("from matplotlib import pyplot as plt", exec_globals)
        exec(code_string_ocr, exec_globals)
        plt.close('all')

    except Exception as e:
        err_trace = traceback.format_exc()
        last_line = err_trace.strip().split('\n')[-1]
        error_queue.put(last_line)
        raise

def save_chart_from_code(image_filename, code_string, temp_directory):
    """Execute code to generate chart images"""
    error_queue = multiprocessing.Queue()
    p = multiprocessing.Process(
        target=_worker_code,
        args=(image_filename, code_string, temp_directory, error_queue)
    )
    p.start()
    p.join(CODE_TIMEOUT)

    if p.is_alive():
        p.terminate()
        p.join()
        return (1, "Code execution timeout")

    if p.exitcode == 0:
        return (0, None)
    else:
        error_message = None
        try:
            error_message = error_queue.get_nowait()
        except:
            error_message = "Unknown error during code execution"
        return (1, error_message)

def encode_image(image_path: str) -> torch.Tensor:
    """Encode image into feature vector using SIGLIP2"""
    if not os.path.exists(image_path):
        # Create a blank white image with default matplotlib figure size (6.4, 4.8 inches)
        # At 100 DPI (matplotlib default), this is 640x480 pixels
        width, height = int(6.4 * 100), int(4.8 * 100)
        image = Image.new('RGB', (width, height), color='white')
    else:
        image = Image.open(image_path).convert('RGB')
    
    inputs = siglip_processor(images=image, return_tensors="pt").to("cuda")
    
    with torch.no_grad():
        embeddings = siglip_model.get_image_features(**inputs)
    
    return embeddings

def calculate_visual_similarity(image1_path: str, image2_path: str) -> float:
    """Calculate cosine similarity between two images"""
    emb1 = encode_image(image1_path)
    emb2 = encode_image(image2_path)
    
    # Normalize embeddings
    emb1 = F.normalize(emb1, p=2, dim=-1)
    emb2 = F.normalize(emb2, p=2, dim=-1)
    
    similarity = F.cosine_similarity(emb1, emb2, dim=-1)
    return similarity.item()

def extract_text_from_image(image_path: str) -> List[str]:
    """Extract text from image using OCR"""
    if not os.path.exists(image_path):
        return []
    
    try:
        results = ocr_model.ocr(image_path, cls=True)
        text_list = []
        for res in results:
            if res is not None:
                for line in res:
                    text_list.append(line[1][0])
        return text_list
    except Exception as e:
        print(f"OCR error for {image_path}: {e}")
        return []

def calculate_ocr_f1_score(gt_texts: List[str], pred_texts: List[str]) -> Tuple[float, float, float]:
    """Calculate precision, recall, and F1 score for text matching"""
    if not gt_texts or not pred_texts:
        return 0.0, 0.0, 0.0
    
    gt_texts = [text.lower().strip() for text in gt_texts if text.strip()]
    pred_texts = [text.lower().strip() for text in pred_texts if text.strip()]

    if not gt_texts or not pred_texts:
        return 0.0, 0.0, 0.0

    # Count true positives
    true_positives = sum(1 for text in pred_texts if text in gt_texts)
    
    # Calculate metrics
    precision = true_positives / len(pred_texts) if pred_texts else 0
    recall = true_positives / len(gt_texts) if gt_texts else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return precision, recall, f1

def process_single_item(item: Dict, image_dir: str) -> Dict:
    """Process a single chart-caption pair"""
    results = {
        'image_filename': item['image_filename'],
        'caption': item.get('generated_caption', item.get('caption', '')),
        'generated_code': None,
        'code_success': False,
        'code_error': None,
        'vcs_score': 0.0,
        'ocr_precision': 0.0,
        'ocr_recall': 0.0,
        'ocr_f1': 0.0
    }
    
    # Original image path
    original_path = os.path.join(image_dir, item['image_filename'])
    if not os.path.exists(original_path):
        results['code_error'] = f"Original image not found: {original_path}"
        return results
    
    # Step 1: Generate code
    try:
        code = generate_code_from_caption(results['caption'])
        results['generated_code'] = code
    except Exception as e:
        results['code_error'] = f"Code generation error: {str(e)}"
        return results
    
    # Step 2: Reconstruct chart (with debugging)
    output_filename = f"recon_{item['image_filename']}"
    
    for attempt in range(MAX_DEBUG_ATTEMPTS):
        status, error_msg = save_chart_from_code(output_filename, code, TEMP_DIR)
        
        if status == 0:
            results['code_success'] = True
            break
        elif attempt < MAX_DEBUG_ATTEMPTS - 1:
            # Try to debug the code
            try:
                print(f"Code error... Attempting to debug... Attempt {attempt + 1} of {MAX_DEBUG_ATTEMPTS}")
                code = debug_code(code, error_msg)
                results['generated_code'] = code
            except Exception as e:
                results['code_error'] = f"Debug error: {str(e)}"
                break
        else:
            results['code_error'] = error_msg
    
    if not results['code_success']:
        return results
    
    # Paths for reconstructed images
    recon_path = os.path.join(TEMP_DIR, output_filename)
    recon_path_ocr = os.path.join(f"{TEMP_DIR}_ocr", output_filename)
    
    # Step 3: Compute visual similarity
    try:
        results['vcs_score'] = calculate_visual_similarity(original_path, recon_path)
    except Exception as e:
        results['code_error'] = f"Visual similarity error: {str(e)}"
    
    # Step 4: Compute OCR score (use OCR version for better text extraction)
    try:
        gt_texts = extract_text_from_image(original_path)
        pred_texts = extract_text_from_image(recon_path_ocr)
        precision, recall, f1 = calculate_ocr_f1_score(gt_texts, pred_texts)
        
        results['ocr_precision'] = precision
        results['ocr_recall'] = recall
        results['ocr_f1'] = f1
    except Exception as e:
        results['code_error'] = f"OCR error: {str(e)}"
    
    return results

def compute_metrics(input_json: str, image_dir: str, output_json: str):
    """Compute VCS and OCR metrics for all items"""
    # Initialize models
    initialize_models()
    
    # Create temp directories
    os.makedirs(TEMP_DIR, exist_ok=True)
    os.makedirs(f"{TEMP_DIR}_ocr", exist_ok=True)
    
    # Load data
    with open(input_json, 'r') as f:
        data = json.load(f)
    
    # Process each item
    results = []
    vcs_scores = []
    ocr_scores = []
    success_count = 0
    
    for item in tqdm(data, desc="Computing VCS and OCR metrics"):
        result = process_single_item(item, image_dir)
        results.append(result)
        
        if result['code_success']:
            success_count += 1
            vcs_scores.append(result['vcs_score'])
            ocr_scores.append(result['ocr_f1'])
        
        print(f"Image: {result['image_filename']}, Success: {result['code_success']}, "
              f"VCS: {result['vcs_score']:.4f}, OCR F1: {result['ocr_f1']:.4f}")
    
    # Compute aggregate metrics
    aggregate_metrics = {
        'total_items': len(data),
        'successful_reconstructions': success_count,
        'reconstruction_rate': success_count / len(data) if data else 0,
        'avg_vcs_score': np.mean(vcs_scores) if vcs_scores else 0,
        'std_vcs_score': np.std(vcs_scores) if vcs_scores else 0,
        'avg_ocr_f1': np.mean(ocr_scores) if ocr_scores else 0,
        'std_ocr_f1': np.std(ocr_scores) if ocr_scores else 0
    }
    
    # Save results
    output_data = {
        'aggregate_metrics': aggregate_metrics,
        'individual_results': results
    }
    
    with open(output_json, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    # Print summary
    print("\n=== Metrics Summary ===")
    print(f"Total items: {aggregate_metrics['total_items']}")
    print(f"Successful reconstructions: {aggregate_metrics['successful_reconstructions']} "
          f"({aggregate_metrics['reconstruction_rate']:.2%})")
    print(f"Average VCS Score: {aggregate_metrics['avg_vcs_score']:.4f} "
          f"(±{aggregate_metrics['std_vcs_score']:.4f})")
    print(f"Average OCR F1 Score: {aggregate_metrics['avg_ocr_f1']:.4f} "
          f"(±{aggregate_metrics['std_ocr_f1']:.4f})")
    
    return output_data

def main():
    parser = argparse.ArgumentParser(
        description="Compute Visual Consistency Score and OCR Score for chart image captions"
    )
    parser.add_argument(
        "--input_json", 
        type=str, 
        required=True,
        help="Path to input JSON file containing image filenames and captions"
    )
    parser.add_argument(
        "--image_dir", 
        type=str, 
        required=True,
        help="Directory containing original chart images"
    )
    parser.add_argument(
        "--output_json", 
        type=str, 
        default="vcs_ocr_results.json",
        help="Path to output JSON file for results"
    )
    parser.add_argument(
        "--temp_dir", 
        type=str, 
        default="./temp_charts",
        help="Directory for temporary chart images"
    )
    
    args = parser.parse_args()
    
    # Update global temp directory
    global TEMP_DIR
    TEMP_DIR = args.temp_dir
    
    # Compute metrics
    compute_metrics(args.input_json, args.image_dir, args.output_json)

if __name__ == "__main__":
    main()