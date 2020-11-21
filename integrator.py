from image_caption.caption import get_image_caption
from meme_template.templateselection import get_meme_template
from find_audio.find_audio import main
import ffmpeg
import pickle
# from caption_generator import CaptioningLSTM, CaptioningLSTMWithLabels,seq_to_text,split_caption,memeify_image,Vocab
import torch
from PIL import Image
import re
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
print('DEVICE:', DEVICE)
DATA_DIR = 'memes900k'

def tokenize( text):
    return re.compile(r"[<\w'>]+|[^\w\s]+").findall(text)

# def get_meme_caption(img_pil, caption,meme_template):
#     FILE_IDS_NAME = 'file_ids.txt'
#     FILE_IDS = {}
#     FILE_TO_CLASS = {
#         "CaptioningLSTM": CaptioningLSTM,
#
#         "CaptioningLSTMWithLabels": CaptioningLSTMWithLabels,
#
#     }
#     with open(FILE_IDS_NAME, 'r') as f:
#         for line in f:
#             name, gid = line.strip().split('\t')
#             FILE_IDS[name] = gid
#
#     ckpt_path = 'LSTMDecoderWords.best.pth'
#
#     # model_without_label = load_and_build_model(gdrive_id, ckpt_path, model_class)
#     ckpt_path = 'LSTMDecoderWithLabelsWords.best.pth'
#     model_class = FILE_TO_CLASS["CaptioningLSTMWithLabels"]
#     model_with_label = model_class.from_pretrained(ckpt_path).to(DEVICE)
#     vocab_words = Vocab.load('vocab/vocab_words.txt')
#     vocabulary = vocab_words
#     vocab_stoi = pickle.load(open("integration_module/token_words_stoi.pickle", "rb"))
#     templates = dill.load(open("integration_module/datasets_words_templates_dill.pkl", "rb"))
#     images = dill.load(open("datasets_words_images_dill.pkl", "rb"))
#
#     label = meme_template
#     tokens = re.compile(r"[<\w'>]+|[^\w\s]+").findall(label)
#     tokens = [tok if tok in vocab_stoi else SPECIAL_TOKENS['UNK'] for tok in tokens]
#     tokens += [SPECIAL_TOKENS['EOS']]
#     tokens = [vocab_stoi[tok] for tok in tokens]
#
#     labels = torch.tensor(tokens).unsqueeze(0)
#     img_torch = images[label]
#     img_pil = Image.open(templates[label])
#
#     img_torch = img_torch.unsqueeze(0)
#
#
#     model_without_label = model_class.from_pretrained(ckpt_path).to(DEVICE)
#     model = "withoutlabels"
#     T = 1.3
#     beam_size = 10
#     top_k = 100
#     labels = None,
#     if model == "withoutlabels":
#         model = model_without_label
#     else:
#         model = model_with_label
#     delimiter = ' '
#     max_len = 32
#     model.eval()
#     print(caption)
#     if caption is not None:
#         # tokenize
#         tokens = re.compile(r"[<\w'>]+|[^\w\s]+").findall(caption)
#         # replace with `UNK`
#         tokens = [tok if tok in vocab_stoi else SPECIAL_TOKENS['UNK'] for tok in tokens]
#         # add `EOS`
#         tokens += [SPECIAL_TOKENS['EOS']]
#         # convert to ids
#         tokens = [vocab_stoi[tok] for tok in tokens]
#         caption_tensor = torch.tensor(tokens[:-1]).unsqueeze(0).to(DEVICE)
#     else:
#         caption_tensor = None
#
#     if labels is None:
#         with torch.no_grad():
#             output_seq = model.generate(
#                 image=img_torch, caption=caption_tensor,
#                 max_len=max_len, beam_size=beam_size, temperature=T, top_k=top_k
#             )
#     else:
#         with torch.no_grad():
#             output_seq = model.generate(
#                 image=img_torch, label=labels, caption=caption_tensor,
#                 max_len=max_len, beam_size=beam_size, temperature=T, top_k=top_k
#             )
#
#     pred_seq = output_seq
#     text = seq_to_text(pred_seq, vocab=vocabulary, delimiter=delimiter)
#     top, bottom = split_caption(text, num_blocks=2)
#     return memeify_image(img_pil, top, bottom, font_path="/content/impact.tff"), text

def get_audio(model_audio_mapper, embeddings, meme_caption):
    audio = main(model_audio_mapper, "sentence_bert", embeddings, meme_caption)
    return audio

def get_meme_clip():
    input_still = ffmpeg.input("Problem.jpg")
    input_audio = ffmpeg.input("Problem.mpeg")

    (
        ffmpeg
            .concat(input_still, input_audio, v=1, a=1)
            .output("output.mp4")
            .run(overwrite_output=True)
    )

if __name__ == "__main__":
    image = "C:\Class Documents\DL\project\image_caption\Test_Images\cat.jpg"
    model_image_caption = "C:\Class Documents\DL\BEST_checkpoint_coco_5_cap_per_img_5_min_word_freq.pth.tar"
    word_map = "./weights/WORDMAP_coco_5_cap_per_img_5_min_word_freq.json --beam_size=5"
    # model_audio_mapper =
    # embedding =
    model_meme_template = '/content/drive/My Drive/Resume/meme_template_selector.pt'
    # captionsFilePath =
    caption = get_image_caption(image, model_image_caption, word_map)
    meme_template = get_meme_template(caption, model_meme_template)
    # meme_image, meme_caption = get_meme_caption(caption,meme_template)
    # get_audio(model_audio_mapper, embeddings, meme_caption)
    # get_meme_clip()
