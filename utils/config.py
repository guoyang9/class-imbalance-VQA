# ----------------------running settings-------------------------- #
cp_data     = True      # using vqa-cp or not
version     = 'v2'      # 'v1' or 'v2'
train_set   = 'train'   # 'train' or 'train+val'
loss_type   = 'bce'     # 'bce' or 'ce'
in_memory   = False     # load all the image feature in memory

# ----------------------flags of using answer mask and rescale--- #
use_mask    = True
use_miu     = True
ft_lr       = 2e-3      # fine-tune learning rate

# ----------------------before-process data paths---------------- #
main_path       = '/disk0/yang/test/vqa-data/'
qa_path         = main_path + 'vqa-cp/' if cp_data else main_path
qa_path        += version # questions and answers
bottom_up_path  = main_path + 'bottom_up_feature/'  # raw image features
glove_path      = main_path + 'word_embed/glove/glove.6B.300d.txt'

# ----------------------image id related paths------------------- #
ids_path    = 'data/'
image_path  = main_path + 'mscoco/' # image paths

# ----------------------processed data paths--------------------- #
rcnn_path           = main_path + 'rcnn-data/'
cache_root          = qa_path + '/cache/'
dict_path           = qa_path + '/dictionary.json'
glove_embed_path    = qa_path + '/glove6b_init.npy'

# ----------------------running settings------------------------- #
max_question_len    = 14
image_dataset       = 'mscoco'
task                = 'OpenEnded' if not cp_data else 'vqacp'
test_split          = 'test2015'    # 'test-dev2015' or 'test2015'
min_occurence       = 9             # answer frequency less than min will be omitted

# ----------------------preprocess image config------------------ #
num_fixed_boxes         = 36        # max number of object proposals per image
output_features         = 2048      # number of features in each object proposal
trainval_num_images     = 123287    # number of images for train and val
test_num_images         = 82783     # number of images for testing
