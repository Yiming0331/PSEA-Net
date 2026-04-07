#获取指定数据集的配置信息
DATASET_CONFIG = {
    'Celeb-DF-v2': {
        'num_classes': 2,
        'train_list_name': 'train.txt',
        'val_list_name': 'val.txt',
        'test_list_name': 'test.txt',
        'filename_seperator': " ",
        'image_tmpl': '{:03d}.jpg',
        'filter_video': 3   
    },
    'Celeb-DF-v3': {
        'num_classes': 2,
        'train_list_name': 'train.txt',
        'val_list_name': 'val.txt',
        'test_list_name': 'test.txt',
        'filename_seperator': " ",
        'image_tmpl': '{:03d}.jpg',
        'filter_video': 3   
    },
    'FF++': {
        'num_classes': 2,
        'train_list_name': 'train.txt',
        'val_list_name': 'val.txt',
        'test_list_name': 'test.txt',
        'filename_seperator': " ",
        'image_tmpl': '{:03d}.jpg',
        'filter_video': 3   
    }
    ,
    'CFF++': {
        'num_classes': 2,
        'train_list_name': 'train.txt',
        'val_list_name': 'val.txt',
        'test_list_name': 'test.txt',
        'filename_seperator': " ",
        'image_tmpl': '{:03d}.jpg',
        'filter_video': 3   
    }
    ,
    'DFDC': {
        'num_classes': 2,
        'train_list_name': 'train.txt',
        'val_list_name': 'val.txt',
        'test_list_name': 'test.txt',
        'filename_seperator': " ",
        'image_tmpl': '{:03d}.jpg',
        'filter_video': 3   
    }
    ,
    'NeuralTextures': {
        'num_classes': 2,
        'train_list_name': 'train_onlyNeuralTextures.txt',
        'val_list_name': 'val_onlyNeuralTextures.txt',
        'test_list_name': 'test_onlyNeuralTextures.txt',
        'filename_seperator': " ",
        'image_tmpl': '{:03d}.jpg',
        'filter_video': 3   
    }
    ,
    'Face2Face': {
        'num_classes': 2,
        'train_list_name': 'train_onlyFace2Face.txt',
        'val_list_name': 'val_onlyFace2Face.txt',
        'test_list_name': 'test_onlyFace2Face.txt',
        'filename_seperator': " ",
        'image_tmpl': '{:03d}.jpg',
        'filter_video': 3   
    }
    ,
    'Deepfakes': {
        'num_classes': 2,
        'train_list_name': 'train_onlyDeepfakes.txt',
        'val_list_name': 'val_onlyDeepfakes.txt',
        'test_list_name': 'test_onlyDeepfakes.txt',
        'filename_seperator': " ",
        'image_tmpl': '{:03d}.jpg',
        'filter_video': 3   
    }
    ,
    'FaceSwap': {
        'num_classes': 2,
        'train_list_name': 'train_onlyFaceSwap.txt',
        'val_list_name': 'val_onlyFaceSwap.txt',
        'test_list_name': 'test_onlyFaceSwap.txt',
        'filename_seperator': " ",
        'image_tmpl': '{:03d}.jpg',
        'filter_video': 3   
    }
    ,
    'UnNeuralTextures': {
        'num_classes': 2,
        'train_list_name': 'train_unNeuralTextures.txt',
        'val_list_name': 'val_unNeuralTextures.txt',
        'test_list_name': 'test_unNeuralTextures.txt',
        'filename_seperator': " ",
        'image_tmpl': '{:03d}.jpg',
        'filter_video': 3   
    }
    ,
    'UnFace2Face': {
        'num_classes': 2,
        'train_list_name': 'train_unFace2Face.txt',
        'val_list_name': 'val_unFace2Face.txt',
        'test_list_name': 'test_unFace2Face.txt',
        'filename_seperator': " ",
        'image_tmpl': '{:03d}.jpg',
        'filter_video': 3   
    }
    ,
    'UnDeepfakes': {
        'num_classes': 2,
        'train_list_name': 'train_unDeepfakes.txt',
        'val_list_name': 'val_unDeepfakes.txt',
        'test_list_name': 'test_unDeepfakes.txt',
        'filename_seperator': " ",
        'image_tmpl': '{:03d}.jpg',
        'filter_video': 3   
    }
    ,
    'UnFaceSwap': {
        'num_classes': 2,
        'train_list_name': 'train_unFaceSwap.txt',
        'val_list_name': 'val_unFaceSwap.txt',
        'test_list_name': 'test_unFaceSwap.txt',
        'filename_seperator': " ",
        'image_tmpl': '{:03d}.jpg',
        'filter_video': 3   
    }
    ,
    'Wilddeepfake': {
        'num_classes': 2,
        'train_list_name': 'train.txt',
        'val_list_name': 'test.txt',
        'test_list_name': 'test.txt',
        'filename_seperator': " ",
        'image_tmpl': '{:03d}.jpg',
        'filter_video': 3   
    }
    ,









}



def get_dataset_config(dataset, use_lmdb=False):
    ret = DATASET_CONFIG[dataset]
    num_classes = ret['num_classes']
    train_list_name = ret['train_list_name'].replace("txt", "lmdb") if use_lmdb \
        else ret['train_list_name']
    val_list_name = ret['val_list_name'].replace("txt", "lmdb") if use_lmdb \
        else ret['val_list_name']
    test_list_name = ret['test_list_name'].replace("txt", "lmdb") if use_lmdb \
        else ret['test_list_name']
    filename_seperator = ret['filename_seperator']
    image_tmpl = ret['image_tmpl']
    filter_video = ret.get('filter_video', 0)
    label_file = ret.get('label_file', None)

    return num_classes, train_list_name, val_list_name, test_list_name, filename_seperator, \
           image_tmpl, filter_video, label_file
