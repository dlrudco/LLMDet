import json
from collections import defaultdict
import os
import re
from tqdm import tqdm
# 경로는 상황에 맞게 바꿔줘
base_path = '/mnt/sdd/grounding_data/flickr30k_entities'
CAPTION_JSON_PATH = os.path.join(base_path, "captions.txt")   # orig_captions가 들어있는 json 파일
INPUT_JSONL_PATH  = os.path.join(base_path, "flickr_train_vg7.jsonl")  # 원본 jsonl
OUTPUT_JSONL_PATH = os.path.join(base_path, "flickr_train_base.jsonl")  # 결과 jsonl

def build_imageid_to_caption_map(caption_json_path):
    with open(caption_json_path, "r", encoding="utf-8") as f:
        orig_captions = f.readlines()

    imageid_to_captions = defaultdict(list)

    # orig_captions['images'] 원소 형태:
    # {'image_id': 203564, 'id': 37, 'caption': 'A bicycle replica ...'}
    for item in orig_captions[1:]:  # 첫 줄은 헤더이므로 건너뜀
        image_id, caption = item.split('.jpg,')
        caption = caption.strip()
        imageid_to_captions[image_id].append(caption)

    # 하나의 image_id에 있는 여러 캡션들을 하나의 문자열로 합치기
    # 합치는 방식은 필요에 따라 바꿔도 됨 (예: " ".join([...]) 대신 " ".join([...]))
    imageid_to_caption_str = {}
    for image_id, caps in imageid_to_captions.items():
        # 간단히 공백으로 이어 붙임 (문장 여러 개가 이어진 하나의 텍스트)
        joined = " ".join(caps)
        imageid_to_caption_str[image_id] = joined

    return imageid_to_caption_str

def filename_to_image_id(filename: str) -> int:
    """
    '000000272026.jpg' 또는 'COCO_train2014_000000272026.jpg' 같은 형식 모두를 커버할 수 있도록
    파일명에서 숫자만 추출해서 int(image_id)로 변환.
    """
    base = os.path.splitext(os.path.basename(filename))[0]
    digits = re.sub(r"[^0-9]", "", base)
    if not digits:
        raise ValueError(f"Filename '{filename}'에서 image_id를 추출할 수 없습니다.")
    return int(digits)

def replace_gpt_captions_in_jsonl(
    input_jsonl_path: str,
    output_jsonl_path: str,
    imageid_to_caption: dict
):
    """
    jsonl 파일을 한 줄씩 읽어서 conversations 안의 GPT 응답을
    image_id에 해당하는 일반 캡션 문자열로 교체하고,
    결과를 새로운 jsonl 파일로 저장.
    """
    with open(input_jsonl_path, "r", encoding="utf-8") as fin, \
         open(output_jsonl_path, "w", encoding="utf-8") as fout:

        for line in tqdm(fin):
            line = line.strip()
            if not line:
                continue

            data = json.loads(line)

            filename = data.get("filename")
            if filename is None:
                # filename이 없는 줄이면 그대로 통과
                fout.write(json.dumps(data, ensure_ascii=False) + "\n")
                continue

            try:
                image_id = filename_to_image_id(filename)
            except ValueError:
                # image_id를 못 뽑으면 원본 유지
                fout.write(json.dumps(data, ensure_ascii=False) + "\n")
                continue

            # 해당 image_id의 캡션 문자열 가져오기
            caption_text = imageid_to_caption.get(str(image_id), None)
            if caption_text is None:
                breakpoint()
                # 캡션이 없으면 원본 유지 (원하면 빈 문자열로 교체하도록 바꿔도 됨)
                fout.write(json.dumps(data, ensure_ascii=False) + "\n")
                continue

            # conversations 안에서 from == "gpt" 인 value를 교체
            convs = data.get("conversations", [])
            
            for conv in convs:
                if conv.get("from") == "gpt":
                    conv["value"] = caption_text
                elif conv.get("from") == "human":
                    conv["value"] = "<image>\\nGenerate several captions to explain the image in detail."

            data["conversations"] = convs

            # 한 줄씩 jsonl로 기록
            fout.write(json.dumps(data, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    # 1) image_id -> caption 문자열 매핑 만들기
    imageid_to_caption = build_imageid_to_caption_map(CAPTION_JSON_PATH)

    # 2) jsonl을 돌면서 gpt 캡션을 교체하고 새 jsonl로 저장
    replace_gpt_captions_in_jsonl(
        INPUT_JSONL_PATH,
        OUTPUT_JSONL_PATH,
        imageid_to_caption
    )
