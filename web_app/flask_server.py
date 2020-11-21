from flask import Flask, send_from_directory, request, Response
from end_to_end import *
import warnings
warnings.filterwarnings("ignore")
app = Flask(__name__)

# File Paths
image = "C:\Class Documents\DL\project\image_caption\Test_Images\cat.jpg"
model_image_caption = "./weights/BEST_checkpoint_coco_5_cap_per_img_5_min_word_freq.pth.tar"
word_map = "./weights/WORDMAP_coco_5_cap_per_img_5_min_word_freq.json"
model_audio_mapper = r"C:\Class Documents\DL\project\find_audio\roberta-base-nli-mean-tokens"
embedding = r"C:\Class Documents\DL\project\find_audio\files"
template_model = r"C:\Class Documents\DL\project\meme_template\bc50.pt"
map_file = r"C:\Class Documents\DL\project\meme_template\50templates_map.sav"
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DATA_DIR = 'memes900k'

checkpoint = torch.load(model_image_caption, map_location=str(DEVICE))

@app.route('/')
def hello_world():
    return send_from_directory('templates/', 'index.html')

@app.route("/i2c", methods=['GET'])
def image2cap():
    form = dict(request.args)
    image  = getImage(form)
    caption = get_image_caption(checkpoint, word_map, image)
    return caption

@app.route("/selTemp", methods=['GET'])
def selectTemplate():
    form = dict(request.args)
    caption = form['caption']
    meme_template = get_meme_template(caption, 50, template_model, map_file)
    return meme_template

def getImage(form):
    return ''

@app.route("/selTemp", methods=['GET'])
def selectTemplate():
    form = dict(request.args)
    meme_template = form['meme']
    meme_image, meme_caption = get_meme_caption(caption, meme_template)
    meme_file_name = meme_caption[:10].replace(" ", "").replace("<sep>", "")
    meme_image.save("memes_output\\"+meme_file_name+".jpeg", "JPEG")
    label, audio = get_audio(model_audio_mapper, "sentence_bert", embedding, meme_caption)
    memePath = get_meme_clip("memes_output\\"+meme_file_name+".jpeg", audio, meme_file_name)
    return Response(open(memePath, "rb"), mimetype="video/mp4")

if __name__ == "__main__":
    app.run(host="192.168.1.116", debug=True)