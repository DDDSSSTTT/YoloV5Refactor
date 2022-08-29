from train import Trainer
from dataset.read_data import DataReader, transforms
from dataset.load_data import DataLoader


def init_params():
    # params['train_annotations_dir'] = '../data/voc/voc_train.txt'
    # params['valid_annotations_dir'] = '../data/voc/valid.txt'
    # params['class_name_dir'] = '../data/voc/voc.names'
    params = {'train_annotations_dir': '../data/chess_pieces/train.txt',
              'valid_annotations_dir': '../data/chess_pieces/valid.txt',
              'class_name_dir': '../data/chess_pieces/train/_classes.txt',
              'yaml_dir': 'configs/yolo-s-mish.yaml',
              'log_dir': '../logs',
              'checkpoint_dir': '../weights',
              'saved_model_dir': '../weights/yolov5',
              'n_epochs': 100, 'batch_size': 4,
              'multi_gpus': False, 'init_learning_rate': 2e-4,
              'warmup_learning_rate': 1e-6, 'warmup_epochs': 1,
              'img_size': 416, 'mosaic_data': False,
              'augment_data': True, 'anchor_assign_method': 'wh',
              'anchor_positive_augment': True,
              'label_smoothing': 0.02}

    return params


def datasets_from_params(params: dict, trainer: Trainer):
    train_data_reader = DataReader(params['train_annotations_dir'], img_size=params['img_size'], transforms=transforms,
                                   mosaic=params['mosaic_data'], augment=params['augment_data'], filter_idx=None)
    train_data_loader = DataLoader(train_data_reader,
                                   trainer.anchors,
                                   trainer.stride,
                                   params['img_size'],
                                   params['anchor_assign_method'],
                                   params['anchor_positive_augment'])
    train_dataset = train_data_loader(batch_size=params['batch_size'], anchor_label=True)
    train_dataset.len = len(train_data_reader)

    valid_data_reader = DataReader(params['valid_annotations_dir'], img_size=params['img_size'], transforms=transforms,
                                   mosaic=params['mosaic_data'], augment=params['augment_data'], filter_idx=None)

    valid_data_loader = DataLoader(valid_data_reader,
                                   trainer.anchors,
                                   trainer.stride,
                                   params['img_size'],
                                   params['anchor_assign_method'],
                                   params['anchor_positive_augment'])
    valid_dataset = valid_data_loader(batch_size=params['batch_size'], anchor_label=True)
    valid_dataset.len = len(valid_data_reader)
    return train_dataset, valid_dataset

def create_detect_opt():
    opt = {}
    opt['img_dir'] = '../data/chess_pieces/test/cfc306bf86176b92ffc1afbb98d7896f_jpg.rf.effd71a5dcd98ec0f24072af5f7c0a31.jpg'
    opt['class_name_dir'] = '../data/chess_pieces/train/_classes.txt'
    opt['model_dir'] = '../weights/yolov5'
    opt['img_size'] = 416
    opt['conf_threshold'] = 0.8
    opt['iou_threshold'] = 0.3
    return opt