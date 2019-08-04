import numpy as np


def weighted_nms_dict(input_dict, thresh=0.5):
    dets = np.hstack((input_dict['detection_boxes'],
                      np.expand_dims(input_dict['detection_scores'], 1)))

    max_ids, weighted_boxes = weighted_nms(dets, thresh)

    output_dict = {
                   'num_detections': len(max_ids),
                   'detection_boxes': weighted_boxes,
                   'detection_scores': input_dict['detection_scores'][max_ids],
                   'detection_classes': input_dict['detection_classes'][max_ids]
                   }
    return output_dict


def weighted_nms(dets, thresh=0.5):
    scores = dets[:, 4]
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]

    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]

    max_ids = []
    weighted_boxes = []
    while order.size > 0:
        i = order[0]
        max_ids.append(i)
        xx1 = np.maximum(x1[i], x1[order[:]])
        yy1 = np.maximum(y1[i], y1[order[:]])
        xx2 = np.minimum(x2[i], x2[order[:]])
        yy2 = np.minimum(y2[i], y2[order[:]])

        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        iou = inter / (areas[i] + areas[order[:]] - inter)

        in_inds = np.where(iou >= thresh)[0]
        in_dets = dets[in_inds, :]

        weights = in_dets[:, 4] * iou[in_inds]
        wbox = np.sum((in_dets[:, :4] * weights[..., np.newaxis]), axis=0) \
            / np.sum(weights)
        weighted_boxes.append(wbox)

        out_inds = np.where(iou < thresh)[0]
        order = order[out_inds]
        dets = dets[out_inds]

    return max_ids, np.array(weighted_boxes)
