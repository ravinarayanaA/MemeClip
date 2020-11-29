from image_caption.caption import get_image_caption
from meme_template.templateselection import get_meme_template
from find_audio.find_audio import get_audio
from PIL import Image
import ffmpeg
from caption_generator.caption_generator import CaptioningLSTM, CaptioningLSTMWithLabels,seq_to_text,split_caption,memeify_image,Vocab
import torch
import re
import pickle
import dill


SPECIAL_TOKENS = {
    'PAD': '<pad>',
    'UNK': '<unk>',
    'BOS': '<bos>',
    'EOS': '<eos>',
    'SEP': '<sep>',
    'EMPTY': '<emp>',
}

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DATA_DIR = 'memes900k'
print('DEVICE:', DEVICE)

def tokenize( text):
    return re.compile(r"[<\w'>]+|[^\w\s]+").findall(text)

def get_meme_caption(caption,meme_template):
    FILE_IDS_NAME = r'caption_generator\file_ids.txt'
    FILE_IDS = {}
    FILE_TO_CLASS = {
        "CaptioningLSTM": CaptioningLSTM,

        "CaptioningLSTMWithLabels": CaptioningLSTMWithLabels,

    }
    with open(FILE_IDS_NAME, 'r') as f:
        for line in f:
            name, gid = line.strip().split('\t')
            FILE_IDS[name] = gid

    ckpt_path = r'caption_generator\LSTMDecoderWords.best.pth'

    # model_without_label = load_and_build_model(gdrive_id, ckpt_path, model_class)
    # ckpt_path = 'LSTMDecoderWithLabelsWords.best.pth'
    model_class = FILE_TO_CLASS['CaptioningLSTM']
    model_without_label = model_class.from_pretrained(ckpt_path).to(DEVICE)
    vocab_words = Vocab.load(r'caption_generator\vocab\vocab_words.txt')
    vocabulary = vocab_words
    vocab_stoi = pickle.load(open("caption_generator/pickle_files/token_words_stoi.pickle", "rb"))
    templates = dill.load(open("caption_generator/pickle_files/datasets_words_templates_dill.pkl", "rb"))
    images = dill.load(open("caption_generator/pickle_files/datasets_words_images_dill.pkl", "rb"))

    label = meme_template
    img_torch = images[label]
    img_pil = Image.open(templates[label])

    img_torch = img_torch.unsqueeze(0)
    model = "withoutlabels"
    T = 1.3
    beam_size = 10
    top_k = 100
    if model == "withoutlabels":
        model = model_without_label
    delimiter = ' '
    max_len = 32
    model.eval()
    print(caption)
    if caption is not None:
        # tokenize
        tokens = re.compile(r"[<\w'>]+|[^\w\s]+").findall(caption)
        # replace with `UNK`
        tokens = [tok if tok in vocab_stoi else SPECIAL_TOKENS['UNK'] for tok in tokens]
        # add `EOS`
        tokens += [SPECIAL_TOKENS['EOS']]
        # convert to ids
        tokens = [vocab_stoi[tok] for tok in tokens]
        caption_tensor = torch.tensor(tokens[:-1]).unsqueeze(0).to(DEVICE)
    else:
        caption_tensor = None

    with torch.no_grad():
        output_seq = model.generate(
            image=img_torch, caption=caption_tensor,
            max_len=max_len, beam_size=beam_size, temperature=T, top_k=top_k
        )

    pred_seq = output_seq
    text = seq_to_text(pred_seq, vocab=vocabulary, delimiter=delimiter)
    top, bottom = split_caption(text, num_blocks=2)
    return memeify_image(img_pil, top, bottom, font_path="C:\Class Documents\DL\project\caption_generator\fonts\impact.ttf"), text

def get_meme_clip(input_image, input_audio, video_name):
    dir_path = r"C:\Class Documents\DL\project\audio_data_downloader\voicy\dataset"
    input_still = ffmpeg.input(r"C:\Class Documents\DL\project\\"+input_image)
    input_audio = ffmpeg.input(dir_path+"\\"+input_audio)

    outPath = r"C:/Class Documents/DL/project/memeclips/"+video_name+".mp4"
    (
        ffmpeg
            .concat(input_still, input_audio, v=1, a=1)
            .output(outPath)
            .run(overwrite_output=True)
    )
    return outPath

if __name__ == "__main__":
    image = "C:\Class Documents\DL\project\image_caption\Test_Images\cat.jpg"
    model_image_caption = "./weights/BEST_checkpoint_coco_5_cap_per_img_5_min_word_freq.pth.tar"
    word_map = "./weights/WORDMAP_coco_5_cap_per_img_5_min_word_freq.json"
    model_audio_mapper = r"C:\Class Documents\DL\project\find_audio\roberta-base-nli-mean-tokens"
    embedding = r"C:\Class Documents\DL\project\find_audio\files"
    template_model = r"C:\Class Documents\DL\project\meme_template\bc50.pt"
    map_file = r"C:\Class Documents\DL\project\meme_template\50templates_map.sav"

    checkpoint = torch.load(model_image_caption, map_location=str(DEVICE))
    caption = get_image_caption(checkpoint, word_map, image)
    meme_template = get_meme_template(caption, 50, template_model, map_file)
    print(meme_template)
    # meme_template = "Awkward Seal"
    meme_image, meme_caption = get_meme_caption(caption,meme_template)
    meme_file_name = meme_caption[:10].replace(" ","").replace("<sep>","")
    meme_image.save("memes_output\\"+meme_file_name+".jpeg", "JPEG")
    # python find_audio.py -model model/roberta-base-nli-mean-tokens -model_type sentence_bert -embeddings files/ -query "laughing man."
    label, audio = get_audio(model_audio_mapper, "sentence_bert", embedding, meme_caption)
    get_meme_clip("memes_output\\"+meme_file_name+".jpeg", audio, meme_file_name)