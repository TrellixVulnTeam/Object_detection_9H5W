# -license-plate-identification

# Thuật toán giải:
- Tìm vị trí biển số
- In ra biển số

# Hướng dẫn cách làm :
B 1 : get data
B 2 : cài môi trường
B 3 : lable image
B 4 : xuất ra model và test thử 
B 5 : xuất flask app

## B 1 get data
gồm 2 loại ô tô và xe máy :https://www.miai.vn/thu-vien-mi-ai/

## B 2 cài môi trường
### 2.1 cài anaconda (lưu ý lựa chọn anaconda3 version 03/2019 ít lỗi)
link https://repo.anaconda.com/archive/
=>> cứ next next next... là ok (không chỉnh sửa ngoài việc khi cài thay đổi vị trí của nó là C:\Anaconda3 chứ không phải mặc định C:\ProgramData\Anaconda3
### 2.2 cài tensorflow
=>> đầu tiên trong ổ C tạo 1 forder tensorflow1 : copy model trên githup này về ,sửa lại tên bỏ vào trong thư mục vừa tạo
=>> mở anaconda promt -> run Admin -> cd .. cho đến khi dấu nháy về C:\
=>> gõ lệnh sau
base C:\> conda create -n tensorflow1 tensorflow=1.13.1 python=3.6.5 (lưu ý tắt fire wall để có thể cài đặt tensorflow)
sau khi create xong kiểm tra bằng lệnh
base C:\> activate tensorflow1
(tensorflow1) C:\>python
>>>import tensorflow as tf
>>>print(tf.__version__)
=>> gõ tiếp các dòng sau để cài môi trường
(tensorflow1) C:\> conda install -c anaconda protobuf
(tensorflow1) C:\> pip install pillow
(tensorflow1) C:\> pip install lxml
(tensorflow1) C:\> pip install Cython
(tensorflow1) C:\> pip install contextlib2
(tensorflow1) C:\> pip install jupyter
(tensorflow1) C:\> pip install matplotlib
(tensorflow1) C:\> pip install pandas
(tensorflow1) C:\> pip install opencv-python
### 2.3 set pythonpath
(tensorflow1) C:\> set PYTHONPATH=C:\tensorflow1\models;C:\tensorflow1\models\research;C:\tensorflow1\models\research\slim
lưu ý mỗi lần tắt anaconda promt đều cần chạy lại lệnh này
### 2.4 set protobuf cho môi trường window
di chuyển vào trong thư mục research
(tensorflow1) C:\> cd C:\tensorflow1\models\research
copy and paste 
(tensorflow1) C:\> protoc --python_out=. .\object_detection\protos\anchor_generator.proto .\object_detection\protos\argmax_matcher.proto .\object_detection\protos\bipartite_matcher.proto .\object_detection\protos\box_coder.proto .\object_detection\protos\box_predictor.proto .\object_detection\protos\eval.proto .\object_detection\protos\faster_rcnn.proto .\object_detection\protos\faster_rcnn_box_coder.proto .\object_detection\protos\grid_anchor_generator.proto .\object_detection\protos\hyperparams.proto .\object_detection\protos\image_resizer.proto .\object_detection\protos\input_reader.proto .\object_detection\protos\losses.proto .\object_detection\protos\matcher.proto .\object_detection\protos\mean_stddev_box_coder.proto .\object_detection\protos\model.proto .\object_detection\protos\optimizer.proto .\object_detection\protos\pipeline.proto .\object_detection\protos\post_processing.proto .\object_detection\protos\preprocessor.proto .\object_detection\protos\region_similarity_calculator.proto .\object_detection\protos\square_box_coder.proto .\object_detection\protos\ssd.proto .\object_detection\protos\ssd_anchor_generator.proto .\object_detection\protos\string_int_label_map.proto .\object_detection\protos\train.proto .\object_detection\protos\keypoint_box_coder.proto .\object_detection\protos\multiscale_anchor_generator.proto .\object_detection\protos\graph_rewriter.proto
(tensorflow1) C:\tensorflow1\models\research> python setup.py build
(tensorflow1) C:\tensorflow1\models\research> python setup.py install
### 2.5 test thử với hình ảnh và model đã train sẵn
di chuyển vào trong thư mục object_detection
(tensorflow1) C:\tensorflow1\models\research\object_detection> jupyter notebook object_detection_tutorial.ipynb
## B 3 label image
tải lableimg có thể search google và bật lableImg.exe chọn dir\forder chứa ảnh cần label
nếu không thể bật lableImg.exe
có thể clone từ https://github.com/tzutalin/labelImg làm theo hướng dẫn
Lưu ý ở đây đang sử dụng nhận diện cho 1 class 'bienso' 
nếu không muốn sử dụng cho các class khác ví dụ 'quan' 'ao' 'non'
thì vui lòng xóa hết tất cả cả các file trong thư mục(lưu ý chỉ xóa files không xóa thư mục
tất cả file trong \object_detection\images\train  \object_detection\images\test
2 tệp test_labels.csv  train_labels.csv trong  \object_detection\images
tất cả file trong  \object_detection\training
tất cả file trong  \object_detection\inference_graph
