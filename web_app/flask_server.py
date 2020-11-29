from flask import Flask, send_from_directory, request, Response, url_for
from end_to_end import *
import warnings
import torch
warnings.filterwarnings("ignore")
app = Flask(__name__, static_url_path='/static')

# File Paths
image = "C:\Class Documents\DL\project\image_caption\Test_Images\cat.jpg"
model_image_caption = "C:\Class Documents\DL\project\weights\BEST_checkpoint_coco_5_cap_per_img_5_min_word_freq.pth.tar"
word_map = "C:\Class Documents\DL\project\weights\WORDMAP_coco_5_cap_per_img_5_min_word_freq.json"
model_audio_mapper = r"C:\Class Documents\DL\project\find_audio\roberta-base-nli-mean-tokens"
embedding = r"C:\Class Documents\DL\project\find_audio\files"
template_model = r"C:\Class Documents\DL\project\meme_template\bc50.pt"
map_file = r"C:\Class Documents\DL\project\meme_template\50templates_map.sav"
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DATA_DIR = 'memes900k'

checkpoint = torch.load(model_image_caption, map_location=str(DEVICE))

@app.route('/', methods=['GET', 'POST'])
def hello_world():
    return send_from_directory('templates/', 'index.html')

@app.route("/i2c", methods=['POST'])
def image2cap():
    form = dict(request.form)
    print(form)
    dir_path = "C:\Class Documents\DL\project\image_caption\Test_Images\\"
    image  = dir_path + form['image_name']
    caption = get_image_caption(checkpoint, word_map, image)
    return caption

@app.route("/selTemp", methods=['POST'])
def selectTemplate():
    form = dict(request.form)
    print(form)
    caption = form['caption']
    meme_template = get_meme_template(caption, 50, template_model, map_file)
    return meme_template

def getImage(form):
    return ''

@app.route("/memeClip", methods=['POST'])
def memeClip():
    form = dict(request.form)
    print(form)
    meme_template = form['meme']
    caption = form['caption']
    meme_image, meme_caption = get_meme_caption(caption, meme_template)
    meme_file_name = meme_caption[:10].replace(" ", "").replace("<sep>", "")
    meme_image.save("C:\Class Documents\DL\project\\"+"memes_output\\"+meme_file_name+".jpeg", "JPEG")
    label, audio = get_audio(model_audio_mapper, "sentence_bert", embedding, meme_caption)
    memePath = get_meme_clip("memes_output\\" + meme_file_name + ".jpeg", audio, meme_file_name)
    # return Response(open(memePath, "rb"), mimetype="video/mp4")
    return memePath

if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True)
