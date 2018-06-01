from django.http import HttpResponse
from django.shortcuts import render
import os
import tensorflow as tf
import numpy as np
# Create your views here.
from od.utils import label_map_util, vis_util
from PIL import Image
# import cv2
 
class TOD(object):
    def __init__(self):
        self.PATH_TO_CKPT = r'od/model/frozen_inference_graph.pb'
        self.PATH_TO_LABELS = r'od/model/mscoco_label_map.pbtxt'
        self.NUM_CLASSES = 90
        self.detection_graph = self._load_model()
        self.category_index = self._load_label_map()

    def _load_model(self):
        detection_graph = tf.Graph()
        with detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(self.PATH_TO_CKPT, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')
        return detection_graph

    def _load_label_map(self):
        label_map = label_map_util.load_labelmap(self.PATH_TO_LABELS)
        categories = label_map_util.convert_label_map_to_categories(label_map,
                                                                    max_num_classes=self.NUM_CLASSES,
                                                                    use_display_name=True)
        category_index = label_map_util.create_category_index(categories)
        return category_index

    def detect(self, image):
        with self.detection_graph.as_default():
            with tf.Session(graph=self.detection_graph) as sess:
                # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
                image_np_expanded = np.expand_dims(image, axis=0)
                image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')
                boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')
                scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
                classes = self.detection_graph.get_tensor_by_name('detection_classes:0')
                num_detections = self.detection_graph.get_tensor_by_name('num_detections:0')
                # Actual detection.
                (boxes, scores, classes, num_detections) = sess.run(
                    [boxes, scores, classes, num_detections],
                    feed_dict={image_tensor: image_np_expanded})
                # return (boxes, scores, classes, num_detections)
                # Visualization of the results of a detection.
                vis_util.visualize_boxes_and_labels_on_image_array(
                    image,
                    np.squeeze(boxes),
                    np.squeeze(classes).astype(np.int32),
                    np.squeeze(scores),
                    self.category_index,
                    use_normalized_coordinates=True,
                    line_thickness=4)
        return image
detecotr = TOD()

def save(file,path,filename):
    destination = open(os.path.join(path, filename), 'wb+')
    for chunk in file.chunks():
        destination.write(chunk)
    destination.close()
def index(request):
    if request.method=="GET":
        return render(request, 'index.html')
    if request.method == "POST":
        file = request.FILES.get('file')
        # import io
        # buf=io.BytesIO()
        # im.save(buf,'png')
        filename=file.name
        if not file:
            return HttpResponse("no files for upload!")
        else:
            path="./od/static/img/"
            save(file, path, filename)
            filename=filename.replace('.jpg','')
            file="od/static/img/%s.jpg"%filename
            image=Image.open(file)
            image=np.array(image)
            im=detecotr.detect(image)
            im = Image.fromarray(im)
            im.save("od/static/img/%s-d.jpg"%filename)
            # image=cv2.imread(file)
            # img = detecotr.detect(image)
            # cv2.imwrite("od/static/img/%s-d.jpg"%filename,img)
            return render(request, 'index.html',{'img1':"/static/img/%s.jpg"%filename,'img2':"/static/img/%s-d.jpg"%filename})
