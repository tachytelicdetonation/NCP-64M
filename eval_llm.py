import time
import argparse
import random
import warnings
import torch
from transformers import AutoTokenizer
from model.model_ncp import NCPConfig, NCPForCausalLM
from trainer.trainer_utils import setup_seed, get_model_params
warnings.filterwarnings('ignore')

def init_model(args):
    tokenizer = AutoTokenizer.from_pretrained('./model')
    lm_config = NCPConfig(
        hidden_size=args.hidden_size, n_enc_layers=args.n_enc_layers,
        n_concept_layers=args.n_concept_layers, n_dec_layers=args.n_dec_layers,
        arch=args.arch, concept_chunk=args.concept_chunk,
        pq_segments=args.pq_segments, pq_codewords=args.pq_codewords,
        use_irc=bool(args.use_irc), use_crc=bool(args.use_crc) if args.arch == 'ncp' else False,
    )
    model = NCPForCausalLM(lm_config)
    ckp = f'./{args.save_dir}/{args.weight}_{args.hidden_size}.pth'
    model.load_state_dict(torch.load(ckp, map_location=args.device), strict=True)
    get_model_params(model, model.config)
    return model.eval().to(args.device), tokenizer

def main():
    parser = argparse.ArgumentParser(description="NCP model inference")
    parser.add_argument('--save_dir', default='out', type=str)
    parser.add_argument('--weight', default='pretrain', type=str)
    parser.add_argument('--arch', default='ncp', choices=['ncp', 'vanilla'])
    parser.add_argument('--hidden_size', default=768, type=int)
    parser.add_argument('--n_enc_layers', default=3, type=int)
    parser.add_argument('--n_concept_layers', default=2, type=int)
    parser.add_argument('--n_dec_layers', default=3, type=int)
    parser.add_argument('--concept_chunk', default=4, type=int)
    parser.add_argument('--pq_segments', default=6, type=int)
    parser.add_argument('--pq_codewords', default=128, type=int)
    parser.add_argument('--use_irc', default=1, type=int, choices=[0, 1])
    parser.add_argument('--use_crc', default=1, type=int, choices=[0, 1])
    parser.add_argument('--max_new_tokens', default=256, type=int)
    parser.add_argument('--temperature', default=0.85, type=float)
    parser.add_argument('--top_p', default=0.95, type=float)
    parser.add_argument('--show_speed', default=1, type=int)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu'), type=str)
    args = parser.parse_args()

    prompts = [
        'Once upon a time',
        'The little girl',
        'One day, a cat',
        'In a small village',
    ]

    model, tokenizer = init_model(args)
    input_mode = int(input('[0] auto test\n[1] manual input\n'))
    prompt_iter = prompts if input_mode == 0 else iter(lambda: input('💬: '), '')

    for prompt in prompt_iter:
        setup_seed(random.randint(0, 31415926))
        if input_mode == 0: print(f'💬: {prompt}')
        text = (tokenizer.bos_token or '') + prompt
        inputs = tokenizer(text, return_tensors='pt', truncation=True).to(args.device)

        print('🧠: ', end='')
        st = time.time()
        generated_ids = model.generate(
            input_ids=inputs["input_ids"], max_new_tokens=args.max_new_tokens,
            do_sample=True, eos_token_id=tokenizer.eos_token_id,
            top_p=args.top_p, temperature=args.temperature,
        )
        response = tokenizer.decode(generated_ids[0][len(inputs["input_ids"][0]):], skip_special_tokens=True)
        print(response)
        gen_tokens = len(generated_ids[0]) - len(inputs["input_ids"][0])
        print(f'\n[Speed]: {gen_tokens / (time.time() - st):.2f} tokens/s\n\n') if args.show_speed else print('\n\n')

if __name__ == "__main__":
    main()
