# CNN-Visualization
CNN featuremaps, filters Visualization with pytorch  
dataset: MNIST  
model: very simple conv net

내가 한 이상한 짓:
1. test_loader의 dataset에 train 넣고 validation을 함(근데 왜 train, test 그래프가 다르게 나오지)
2. 모델 conv_layer에 activation 함수 안넣고 학습함(아마 그래서 학습 전 후 filter 이미지가 유사하게 나온듯?)

결과:  
filter img의 경우 처음 init 상태와 학습 후에 큰 차이가 없어 보인다. 또한 gabor filter나 다른 자료에 나온 것 처럼 눈에 띄는 필터의 모양이 보이진 않았다(아마 mnist의 경우 conv없는 MLP 뉴럴넷으로도 충분히 근사가 가능해서(모델의 파라미터가 데이터셋에 비해 너무 많아서) 학습을 제대로 안한 것 아닐까, 그래서 학습 전후의 차이가 크게 없는 느낌).  
feature map의 경우 첫 레이어는 입력 이미지의 형상이 잘 보이지만 두번째 레이어에서는 입력 이미지의 형상을 찾아보기 어려웠다. 하지만 학습 후의 두번째 레이어 feature map들은 숫자의 형상이 약간 보였다.

# train result
![image](/MNIST/train%20result/result.png)

# filter imgs
![image](/filter%20imgs/model0%200.png)  
epoch 0, 1st layer
![image](/filter%20imgs/model0%201.png)  
epoch 0, 2nd layer
![image](/filter%20imgs/model10%200.png)  
epoch 10, 1st layer
![image](/filter%20imgs/model10%201.png)  
epoch 10, 2nd layer
![image](/filter%20imgs/model100%200.png)  
epoch 100, 1st layer
![image](/filter%20imgs/model100%201.png)  
epoch 100, 2nd layer

# feature maps
![image](/feature%20map/fm1_0.png)
epoch 0, 1st layer
![image](/feature%20map/fm2_0.png)  
epoch 0, 2nd layer
![image](/feature%20map/fm1_10.png) 
epoch 10, 1st layer
![image](/feature%20map/fm2_10.png)  
epoch 10, 2nd layer
![image](/feature%20map/fm1_100.png)  
epoch 100, 1st layer
![image](/feature%20map/fm2_100.png)  
epoch 100, 2nd layer
