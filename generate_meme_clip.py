
import argparse
from image_caption_integration import final_image_caption
#from import getMemeTemplate
from find_audio import main
parser = argparse.ArgumentParser(description='Meme Clip Generator')

parser.add_argument('--image',  help='path to images')
parser.add_argument('--model_image_caption', help='path to model for image caption generation')
parser.add_argument('--word_map', help='path to word map JSON')
parser.add_argument('--beam_size', default=5, type=int, help='beam size for beam search')
parser.add_argument('--model_meme_template',help='path to model for meme_template')
parser.add_argument('--captionsFilePath',  help='path to captions file')
# parser.add_argument('--dont_smooth', dest='smooth', action='store_false', help='do not smooth alpha overlay')
parser.add_argument('--model_audio_mapper',help='the directory of the model of audio mapper')
parser.add_argument('--embeddings',help='the directory of the sentence embeddings')

args = parser.parse_args()


caption = final_image_caption(args.image,args.model_image_caption,args.word_map,args.beam_size)
#meme_template = getMemeTemplate(caption,args.model_meme_template,args.captionsFilePath)
print(caption)
#meme_caption
audio = main(args.model_audio_mapper,"sentence_bert",args.embeddings, caption)
print(audio)