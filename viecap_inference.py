# MeaCap InvLM single-image inference (flat ViECap layout, NOT `from viecap.xxx`).
# If you see ModuleNotFoundError: No module named 'viecap', replace this file from the repo.

import clip
import torch
import argparse
from PIL import Image
from ClipCap import ClipCaptionModel
from transformers import AutoTokenizer
from utils import compose_discrete_prompts
from search import greedy_search, beam_search, opt_search
from meacap_utils.invlm_prompt import MeaCapInvLMResources, compute_continuous_embeddings


def _hf_load_kwargs(local_files_only: bool) -> dict:
    return {'local_files_only': True} if local_files_only else {}


@torch.no_grad()
def main(args) -> None:
    device = torch.device(args.device)
    clip_hidden_size = 640 if 'RN' in args.clip_model else 512
    hf_kw = _hf_load_kwargs(args.local_files_only)

    tokenizer = AutoTokenizer.from_pretrained(args.language_model, **hf_kw)
    model = ClipCaptionModel(
        args.continuous_prompt_length,
        args.clip_project_length,
        clip_hidden_size,
        gpt_type=args.language_model,
        fusion_type=getattr(args, 'fusion_type', 'linear'),
    )
    model.load_state_dict(torch.load(args.weight_path, map_location=device), strict=False)
    model.to(device)
    encoder, preprocess = clip.load(args.clip_model, device=device)

    invlm_resources = None
    if args.use_ilr or args.using_hard_prompt:
        invlm_resources = MeaCapInvLMResources(args, device)

    image = preprocess(Image.open(args.image_path)).unsqueeze(dim=0).to(device)
    image_features = encoder.encode_image(image).float()
    image_features /= image_features.norm(2, dim=-1, keepdim=True)
    continuous_embeddings = compute_continuous_embeddings(
        args, model, image_features, invlm_resources, args.image_path
    )

    if args.using_hard_prompt:
        from meacap_utils.invlm_prompt import retrieve_memory_concepts

        detected_objects = retrieve_memory_concepts(invlm_resources, args.image_path)

        print("memory concepts:", detected_objects)
        discrete_tokens = compose_discrete_prompts(tokenizer, detected_objects).unsqueeze(dim=0).to(device)
        discrete_embeddings = model.word_embed(discrete_tokens)
        if args.only_hard_prompt:
            embeddings = discrete_embeddings
        elif args.soft_prompt_first:
            embeddings = torch.cat((continuous_embeddings, discrete_embeddings), dim=1)
        else:
            embeddings = torch.cat((discrete_embeddings, continuous_embeddings), dim=1)
    else:
        embeddings = continuous_embeddings

    if 'gpt' in args.language_model:
        if not args.using_greedy_search:
            sentence = beam_search(
                embeddings=embeddings,
                tokenizer=tokenizer,
                beam_width=args.beam_width,
                model=model.gpt,
            )
            sentence = sentence[0]
        else:
            sentence = greedy_search(embeddings=embeddings, tokenizer=tokenizer, model=model.gpt)
    else:
        sentence = opt_search(
            prompts=args.text_prompt,
            embeddings=embeddings,
            tokenizer=tokenizer,
            beam_width=args.beam_width,
            model=model.gpt,
        )
        sentence = sentence[0]

    print(f'the generated caption: {sentence}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--clip_model', default='ViT-B/32')
    parser.add_argument(
        '--language_model',
        default='./checkpoints/gpt2',
        help='local GPT-2 dir or HuggingFace model id',
    )
    parser.add_argument(
        '--vl_model',
        type=str,
        default='./checkpoints/clip-vit-base-patch32',
        help='local HF CLIP dir or model id for memory retrieval',
    )
    parser.add_argument(
        '--parser_checkpoint',
        type=str,
        default='./checkpoints/flan-t5-base-VG-factual-sg',
    )
    parser.add_argument(
        '--wte_model_path',
        type=str,
        default='./checkpoints/all-MiniLM-L6-v2',
    )
    parser.add_argument(
        '--local_files_only',
        action='store_true',
        default=True,
        help='offline mode: load HF models only from local paths (no huggingface.co)',
    )
    parser.add_argument('--continuous_prompt_length', type=int, default=10)
    parser.add_argument('--clip_project_length', type=int, default=10)
    parser.add_argument('--weight_path', default='checkpoints/train_coco/coco_prefix-0014.pt')
    parser.add_argument('--image_path', default='images/instance1.jpg')
    parser.add_argument('--using_hard_prompt', action='store_true', default=True)
    parser.add_argument('--soft_prompt_first', action='store_true', default=False)
    parser.add_argument('--only_hard_prompt', action='store_true', default=False)
    parser.add_argument('--using_greedy_search', action='store_true', default=False)
    parser.add_argument('--beam_width', type=int, default=5)
    parser.add_argument('--text_prompt', type=str, default=None)
    parser.add_argument('--memory_id', type=str, default='coco', help='memory bank name')
    parser.add_argument('--memory_caption_num', type=int, default=5)
    parser.add_argument('--use_ilr', action='store_true', default=False, help='fuse image feat with memory cosine top-K before Projector')
    parser.add_argument('--fusion_w1', type=float, default=0.8)
    parser.add_argument('--fusion_w2', type=float, default=0.2)
    parser.add_argument('--fusion_type', default='linear', choices=['linear', 'gated'])
    args = parser.parse_args()
    print('args: {}\n'.format(vars(args)))
    main(args)
