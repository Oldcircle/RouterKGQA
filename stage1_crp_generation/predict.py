import os
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm
import argparse
from peft import PeftModel


def load_model_and_tokenizer(model_path, adapter_path=None):
    """Load model and tokenizer."""
    print(f"Loading model from {model_path}")

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
        padding_side="left"
    )

    # Ensure pad_token is set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )

    # Load LoRA adapter weights if provided
    if adapter_path:
        print(f"Loading adapter from {adapter_path}")
        model = PeftModel.from_pretrained(model, adapter_path)
        model = model.merge_and_unload()

    model.eval()
    return model, tokenizer


def prepare_prompt(example, template="llama3"):
    """Prepare prompt according to the specified chat template."""
    if template == "qwen":
        # Qwen2.5 ChatML format
        messages = []

        system_content = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."
        if "system" in example and example["system"]:
            system_content = example["system"]

        messages.append(f"<|im_start|>system\n{system_content}<|im_end|>\n")

        if "instruction" in example:
            user_content = example["instruction"]
            if "input" in example and example["input"]:
                user_content += f"\n{example['input']}"
            messages.append(f"<|im_start|>user\n{user_content}<|im_end|>\n")

        messages.append("<|im_start|>assistant\n")
        return "".join(messages)

    elif template == "llama3":
        # Llama 3 format
        messages = []
        if "system" in example and example["system"]:
            messages.append(f"<|start_header_id|>system<|end_header_id|>\n\n{example['system']}<|eot_id|>")

        if "instruction" in example:
            user_content = example["instruction"]
            if "input" in example and example["input"]:
                user_content += f"\n{example['input']}"
            messages.append(f"<|start_header_id|>user<|end_header_id|>\n\n{user_content}<|eot_id|>")

        messages.append("<|start_header_id|>assistant<|end_header_id|>\n\n")
        return "".join(messages)

    elif template == "llama2":
        # Llama 2 format
        system_content = ""
        if "system" in example and example["system"]:
            system_content = f"<<SYS>>\n{example['system']}\n<</SYS>>\n\n"

        user_content = example.get("instruction", "")
        if "input" in example and example["input"]:
            user_content += f"\n{example['input']}"

        return f"[INST] {system_content}{user_content} [/INST]"

    else:
        # Simple format (no chat template)
        prompt = example.get("instruction", "")
        if "input" in example and example["input"]:
            prompt += f"\n{example['input']}"
        return prompt


def beam_search_inference(
    model,
    tokenizer,
    dataset,
    output_file,
    num_beams=15,
    num_return_sequences=15,
    max_new_tokens=512,
    batch_size=1,
    template="llama3"
):
    """Run beam search inference and return all candidate sequences."""

    results = []

    for i in tqdm(range(0, len(dataset), batch_size), desc="Generating"):
        # Build batch
        if batch_size == 1:
            batch = [dataset[i]]
        else:
            batch_indices = list(range(i, min(i + batch_size, len(dataset))))
            batch = [dataset[idx] for idx in batch_indices]

        # Prepare prompts
        prompts = [prepare_prompt(ex, template) for ex in batch]

        # Tokenize
        inputs = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048
        ).to(model.device)

        # Generate with beam search
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                num_beams=num_beams,
                num_return_sequences=num_return_sequences,
                do_sample=False,
                temperature=1.0,
                top_p=1.0,
                return_dict_in_generate=True,
                output_scores=True,
                early_stopping=True
            )

        # Decode all sequences
        generated_sequences = outputs.sequences

        # Process multiple outputs for each input sample
        for j in range(len(prompts)):
            start_idx = j * num_return_sequences
            end_idx = start_idx + num_return_sequences

            sample_sequences = generated_sequences[start_idx:end_idx]

            # Decode all beams
            decoded_outputs = []
            for seq in sample_sequences:
                # Strip the input prompt tokens
                prompt_length = inputs.input_ids[j].shape[0]
                generated_ids = seq[prompt_length:]

                decoded = tokenizer.decode(
                    generated_ids,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False
                )

                decoded_outputs.append(decoded.strip())

            # Save result
            result = {
                "index": i + j,
                "instruction": batch[j].get("instruction", ""),
                "input": batch[j].get("input", ""),
                "prompt": prompts[j],
                "predictions": decoded_outputs,
                "ground_truth": batch[j].get("output", "")
            }
            results.append(result)

        # Periodic checkpoint save
        if (i // batch_size + 1) % 10 == 0:
            save_results(results, output_file)

    # Final save
    save_results(results, output_file)
    return results


def save_results(results, output_file):
    """Save results to a JSON file."""
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"Results saved to {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Beam Search Inference with All Results")
    parser.add_argument("--model_path", type=str, required=True, help="Base model path")
    parser.add_argument("--adapter_path", type=str, default=None, help="LoRA adapter path")
    parser.add_argument("--dataset_path", type=str, required=True, help="Dataset path or name")
    parser.add_argument("--dataset_dir", type=str, default="./data", help="Dataset directory")
    parser.add_argument("--output_file", type=str, default="beam_search_results.json", help="Output file")
    parser.add_argument("--num_beams", type=int, default=15, help="Number of beams")
    parser.add_argument("--num_return_sequences", type=int, default=15, help="Number of sequences to return")
    parser.add_argument("--max_new_tokens", type=int, default=512, help="Max new tokens")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size")
    parser.add_argument("--max_samples", type=int, default=None, help="Max samples to process")
    parser.add_argument("--template", type=str, default="llama3", choices=["llama3", "simple", "qwen", "llama2"])

    args = parser.parse_args()

    # Load model
    model, tokenizer = load_model_and_tokenizer(args.model_path, args.adapter_path)

    # Load dataset
    print(f"Loading dataset from {args.dataset_path}")

    # Try multiple path combinations
    possible_paths = [
        args.dataset_path,
        os.path.join(args.dataset_dir, args.dataset_path),
        os.path.join(args.dataset_dir, args.dataset_path, "test.json"),
        os.path.join(args.dataset_dir, f"{args.dataset_path}.json"),
    ]

    dataset = None
    for path in possible_paths:
        if os.path.exists(path):
            print(f"Found dataset at: {path}")
            try:
                dataset = load_dataset('json', data_files=path, split='train')
                break
            except Exception as e:
                print(f"Failed to load from {path}: {e}")
                continue

    # Fall back to HuggingFace Hub if not found locally
    if dataset is None:
        try:
            print(f"Trying to load from HuggingFace Hub: {args.dataset_path}")
            dataset = load_dataset(args.dataset_path, split='test')
        except Exception as e:
            raise FileNotFoundError(
                f"Could not find dataset. Tried paths:\n" +
                "\n".join(f"  - {p}" for p in possible_paths) +
                f"\n\nError: {e}"
            )

    if args.max_samples:
        dataset = dataset.select(range(min(args.max_samples, len(dataset))))

    print(f"Dataset size: {len(dataset)}")

    # Run inference
    results = beam_search_inference(
        model=model,
        tokenizer=tokenizer,
        dataset=dataset,
        output_file=args.output_file,
        num_beams=args.num_beams,
        num_return_sequences=args.num_return_sequences,
        max_new_tokens=args.max_new_tokens,
        batch_size=args.batch_size,
        template=args.template
    )

    print(f"\nInference completed! Processed {len(results)} samples.")
    print(f"Each sample has {args.num_return_sequences} predictions.")


if __name__ == "__main__":
    main()
