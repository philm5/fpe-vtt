import tensorflow as tf
from pycocotools.coco import COCO
from pycocoevalcap.eval import COCOEvalCap

def init_coco(annotation_file):
    # create coco object and cocoRes object
    coco = COCO(annotation_file)
    return coco

def calculate_scores(fname, coco, global_step):
    cocoRes = coco.loadRes(fname)

    # create cocoEval object by taking coco and cocoRes
    cocoEval = COCOEvalCap(coco, cocoRes)

    # evaluate on a subset of images by setting
    # cocoEval.params['image_id'] = cocoRes.getImgIds()
    # please remove this line when evaluating the full validation set
    cocoEval.params['image_id'] = cocoRes.getImgIds()

    # evaluate results
    # SPICE will take a few minutes the first time, but speeds up due to caching
    cocoEval.evaluate()

    # print output evaluation scores
    for metric, score in cocoEval.eval.items():
        #result_str += "{}:\t\t{:.4f}\n".format(metric, score)
        tf.summary.scalar("coco_scores_All/{}".format(metric), score, step=global_step)

    return cocoEval
