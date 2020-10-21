# Nhận diện biển số xe
sử dụng máy tính siêu cùi nhá (win 7 core 2 Due, 2 gb ram,run  bằng CPU :P)
# Thuật toán giải:
- Tìm vị trí biển số
- In ra biển số

# Hướng dẫn cách làm :
*B 1 : get data
*B 2 : cài môi trường
*B 3 : lable image
*B 4 : xuất ra model và test thử 

## B 1 get data
gồm 2 loại ô tô và xe máy :https://www.miai.vn/thu-vien-mi-ai/

## B 2 cài môi trường
### 2.1 cài anaconda (lưu ý lựa chọn anaconda3 version 03/2019 ít lỗi)
link https://repo.anaconda.com/archive/
=>> cứ next next next... là ok (không chỉnh sửa ngoài việc khi cài thay đổi vị trí của nó là C:\Anaconda3 chứ không phải mặc định C:\ProgramData\Anaconda3
### 2.2 cài tensorflow
đầu tiên trong ổ C tạo 1 forder tensorflow1 : copy model trên githup này về ,sửa lại tên bỏ vào trong thư mục vừa tạo
mở anaconda promt -> run Admin -> cd .. cho đến khi dấu nháy về C:\
gõ lệnh sau
```
base C:\> conda create -n tensorflow1 tensorflow=1.13.1 python=3.6.5 (lưu ý tắt fire wall để có thể cài đặt tensorflow)
```
sau khi create xong kiểm tra bằng lệnh
```
base C:\> activate tensorflow1
(tensorflow1) C:\>python

>>>import tensorflow as tf
>>>print(tf.__version__)
```
gõ tiếp các dòng sau để cài môi trường
```
(tensorflow1) C:\> conda install -c anaconda protobuf
(tensorflow1) C:\> pip install pillow
(tensorflow1) C:\> pip install lxml
(tensorflow1) C:\> pip install Cython
(tensorflow1) C:\> pip install contextlib2
(tensorflow1) C:\> pip install jupyter
(tensorflow1) C:\> pip install matplotlib
(tensorflow1) C:\> pip install pandas
(tensorflow1) C:\> pip install opencv-python
```
### 2.3 set pythonpath
```
(tensorflow1) C:\> set PYTHONPATH=C:\tensorflow1\models;C:\tensorflow1\models\research;C:\tensorflow1\models\research\slim
```
lưu ý mỗi lần tắt anaconda promt đều cần chạy lại lệnh này
### 2.4 set protobuf cho môi trường window
di chuyển vào trong thư mục research
```
(tensorflow1) C:\> cd C:\tensorflow1\models\research
```
copy and paste 
```
(tensorflow1) C:\> protoc --python_out=. .\object_detection\protos\anchor_generator.proto .\object_detection\protos\argmax_matcher.proto .\object_detection\protos\bipartite_matcher.proto .\object_detection\protos\box_coder.proto .\object_detection\protos\box_predictor.proto .\object_detection\protos\eval.proto .\object_detection\protos\faster_rcnn.proto .\object_detection\protos\faster_rcnn_box_coder.proto .\object_detection\protos\grid_anchor_generator.proto .\object_detection\protos\hyperparams.proto .\object_detection\protos\image_resizer.proto .\object_detection\protos\input_reader.proto .\object_detection\protos\losses.proto .\object_detection\protos\matcher.proto .\object_detection\protos\mean_stddev_box_coder.proto .\object_detection\protos\model.proto .\object_detection\protos\optimizer.proto .\object_detection\protos\pipeline.proto .\object_detection\protos\post_processing.proto .\object_detection\protos\preprocessor.proto .\object_detection\protos\region_similarity_calculator.proto .\object_detection\protos\square_box_coder.proto .\object_detection\protos\ssd.proto .\object_detection\protos\ssd_anchor_generator.proto .\object_detection\protos\string_int_label_map.proto .\object_detection\protos\train.proto .\object_detection\protos\keypoint_box_coder.proto .\object_detection\protos\multiscale_anchor_generator.proto .\object_detection\protos\graph_rewriter.proto
```
```
(tensorflow1) C:\tensorflow1\models\research> python setup.py build
(tensorflow1) C:\tensorflow1\models\research> python setup.py install
```
### 2.5 test thử với hình ảnh và model đã train sẵn
di chuyển vào trong thư mục object_detection
```
(tensorflow1) C:\tensorflow1\models\research\object_detection> jupyter notebook object_detection_tutorial.ipynb
```
<p align="center">
  <img src="doc/jupyter_notebook_dogs.jpg">
</p>

## B 3 label image
tải lableimg có thể search google và bật lableImg.exe chọn dir\forder chứa ảnh cần label
nếu không thể bật lableImg.exe
có thể clone từ https://github.com/tzutalin/labelImg làm theo hướng dẫn
Lưu ý ở đây đang sử dụng nhận diện cho 1 class 'bienso' 
nếu muốn sử dụng cho các class khác ví dụ 'quan' 'ao' 'non'
thì vui lòng xóa hết tất cả cả các file trong thư mục(lưu ý chỉ xóa files không xóa thư mục
- tất cả file trong \object_detection\images\train  \object_detection\images\test
- 2 tệp test_labels.csv  train_labels.csv trong  \object_detection\images
- tất cả file trong  \object_detection\training
- tất cả file trong  \object_detection\inference_graph
## B 4 Trainning
### 4.1 tạo data train
tại vị trí \object_detection
run dòng code
```
(tensorflow1) C:\tensorflow1\models\research\object_detection> python xml_to_csv.py
```
để chuyển các file xml trong label thành file csv
mở file generate_tfrecord.py bằng cách gì cũng được chỉnh sửa chỗ này 
```
def class_text_to_int(row_label):
    if row_label == 'bienso':
        return 1
    else:
        None
```        
thành các class mà bạn muốn detection ví dụ 3 class quần áo nón
```def class_text_to_int(row_label):
    if row_label == 'quan':
        return 1
    elif row_label == 'ao':
        return 2
    elif row_label == 'non':
        return 3
    else:
        None
```

sau đó run 2 dòng code
```
python generate_tfrecord.py --csv_input=images\train_labels.csv --image_dir=images\train --output_path=train.record
python generate_tfrecord.py --csv_input=images\test_labels.csv --image_dir=images\test --output_path=test.record
```
### 4.2 tạo label map và cấu hình Training
tạo 1 file bỏ vào  C:\tensorflow1\models\research\object_detection\training folder
với tên là labelmap.pbtxt lưu ý định dạng pbtxt chứ không phải txt
```
item {
  id: 1
  name: 'quan'
}

item {
  id: 2
  name: 'ao'
}

item {
  id: 3
  name: 'non'
}
```
### 4.3 chỉnh sửa thông số cho việc train
mở thư mục C:\tensorflow1\models\research\object_detection\samples\configs và copy file faster_rcnn_inception_v2_pets.config
sau đó qua thư mục \object_detection\training dán vào
dùng visual code để dễ chỉnh sửa
- tại dòng số 9 num_class : chuyển đổi thành số class mà bạn muốn train
- tại dòng số 106 chỉnh sửa thành : fine_tune_checkpoint : "C:/tensorflow1/models/research/object_detection/faster_rcnn_inception_v2_coco_2018_01_28/model.ckpt"
- tại dòng 123 125
input_path : "C:/tensorflow1/models/research/object_detection/train.record"
label_map_path: "C:/tensorflow1/models/research/object_detection/training/labelmap.pbtxt"
- tại dòng 130 : nhập số ảnh dùng dể test trong thư mục \images\test vào vị trí num_examples
- tại dòng 135 137 
input_path : "C:/tensorflow1/models/research/object_detection/test.record"
label_map_path: "C:/tensorflow1/models/research/object_detection/training/labelmap.pbtxt"
Sau đó save và bắt đầu run nào
## 5 Train
tại vị trí \Object_detection  run dòng code
```
python train.py --logtostderr --train_dir=training/ --pipeline_config_path=training/faster_rcnn_inception_v2_pets.config
```
## 6 xuất ra model
sau khi chạy 1 số bước (ít nhất vài trăm step .tốt nhất đến khi loss ~ 0.05 và ít thay đổi)
crl + C để dừng lại
xuất ra dòng code 
```
python export_inference_graph.py --input_type image_tensor --pipeline_config_path training/faster_rcnn_inception_v2_pets.config --trained_checkpoint_prefix training/model.ckpt-XXXX --output_directory inference_graph
```
trong đó XXXX thay thế bằng số model to nhất trong thư mục object_detection\trainning
ví dụ chạy 350 step sẽ có file lưu là model.ckpt-350-....
thì ta run lệnh
```
python export_inference_graph.py --input_type image_tensor --pipeline_config_path training/faster_rcnn_inception_v2_pets.config --trained_checkpoint_prefix training/model.ckpt-350 --output_directory inference_graph
```

## 7 Test thử model mới nào
copy file ảnh hoặc video vào trong thư mục Object_detection chỉnh lại tên trùng với tên trong file Object_detection_image.py đối với ảnh
video đối với file Object_detection_video.py hoặc sử dụng webcam Object_detection_webcam.py

Sau đó tại thư mục Object_detection 
run code 
```
python Object_detection_image.py/Object_detection_video.py/Object_detection_webcam.py
```
![test](https://user-images.githubusercontent.com/61773507/96670449-86ec1c00-1389-11eb-9081-a4cf8176d54a.jpg)
ok
