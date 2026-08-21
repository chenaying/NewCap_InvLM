"""Compute C3 modality means (μ_text, μ_image) for --remove_mean preprocessing."""

import argparse
import os
import pickle

import torch


def _normalize_and_accumulate(embed: torch.Tensor, running_sum: torch.Tensor) -> torch.Tensor:
    embed = embed.float().view(-1)
    running_sum += embed / embed.norm()
    return running_sum


def compute_text_mean(path_of_text_features: str, clip_dim: int) -> torch.Tensor:
    with open(path_of_text_features, 'rb') as infile:
        captions_with_entities = pickle.load(infile)

    text_mean = torch.zeros(clip_dim)
    count = 0
    for item in captions_with_entities:
        if len(item) >= 3:
            embed = item[2]
        else:
            raise ValueError(
                f'Expected [[entities, caption, clip_embed], ...] in {path_of_text_features}'
            )
        text_mean = _normalize_and_accumulate(embed, text_mean)
        count += 1

    if count == 0:
        raise ValueError(f'No captions found in {path_of_text_features}')
    return (text_mean / count).view(1, -1)


def _extract_image_embed(item) -> torch.Tensor:
    if len(item) == 3:
        return item[1]
    if len(item) == 4:
        return item[2]
    raise ValueError(f'Unsupported image feature record with length {len(item)}')


def compute_image_mean(path_of_image_features: str, clip_dim: int) -> torch.Tensor:
    with open(path_of_image_features, 'rb') as infile:
        annotations = pickle.load(infile)

    img_mean = torch.zeros(clip_dim)
    count = 0
    for item in annotations:
        embed = _extract_image_embed(item)
        img_mean = _normalize_and_accumulate(embed, img_mean)
        count += 1

    if count == 0:
        raise ValueError(f'No images found in {path_of_image_features}')
    return (img_mean / count).view(1, -1)


def main() -> None:
    parser = argparse.ArgumentParser(description='Compute C3 text/image CLIP embedding means.')
    parser.add_argument(
        '--path_of_text_features',
        default='./annotations/coco/coco_texts_features_ViT-B32.pickle',
        help='Training text CLIP features pickle from texts_features_extraction.py',
    )
    parser.add_argument(
        '--path_of_image_features',
        default='./annotations/coco/test_captions_ViT-B32.pickle',
        help='Image CLIP features pickle from images_features_extraction.py',
    )
    parser.add_argument(
        '--text_embed_mean_path',
        default='./annotations/coco/normalized_text_embed_mean.pt',
    )
    parser.add_argument(
        '--image_embed_mean_path',
        default='./annotations/coco/normalized_image_embed_mean.pt',
    )
    parser.add_argument('--clip_dim', type=int, default=512)
    parser.add_argument('--skip_text', action='store_true', default=False)
    parser.add_argument('--skip_image', action='store_true', default=False)
    args = parser.parse_args()

    if not args.skip_text:
        text_mean = compute_text_mean(args.path_of_text_features, args.clip_dim)
        os.makedirs(os.path.dirname(args.text_embed_mean_path) or '.', exist_ok=True)
        torch.save(text_mean, args.text_embed_mean_path)
        print(f'Saved text mean ({text_mean.shape}) to {args.text_embed_mean_path}')

    if not args.skip_image:
        if not os.path.isfile(args.path_of_image_features):
            raise FileNotFoundError(
                f'Image features not found: {args.path_of_image_features}\n'
                'Run images_features_extraction.py first, or pass --skip_image.'
            )
        image_mean = compute_image_mean(args.path_of_image_features, args.clip_dim)
        os.makedirs(os.path.dirname(args.image_embed_mean_path) or '.', exist_ok=True)
        torch.save(image_mean, args.image_embed_mean_path)
        print(f'Saved image mean ({image_mean.shape}) to {args.image_embed_mean_path}')


if __name__ == '__main__':
    main()
