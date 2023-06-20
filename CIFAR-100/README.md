# CNN-Visualization
CNN featuremaps, filters Visualization with pytorch  
dataset: CIFAR-100  
model: very simple conv net

내가 한 이상한 짓:

결과: MNIST에 비해 filter img의 학습 전후 변화량이 커보인다. 또한 학습 후 filter imgs에서 gabor filter와 같은 선형 필터도 많이 보였음. MNIST에서 말한 것처럼 복잡한 데이터셋에서 학습을 하니 conv filter들이 학습이 된 것 같다.  
feature map의 경우 크게 눈에 띄는 결과를 찾지 못했다. 아마 모델이 데이터셋에 비해 under parameterized되 있기 때문 아닐까.

# train result
![image](/CIFAR-100/train%20result/result.png)

# filter imgs
![image](/CIFAR-100/filter%20imgs/model0%200.png)  
epoch 0, 1st layer
![image](/CIFAR-100/filter%20imgs/model0%201.png)  
epoch 0, 2nd layer
![image](/CIFAR-100/filter%20imgs/model10%200.png)  
epoch 10, 1st layer
![image](/CIFAR-100/filter%20imgs/model10%201.png)  
epoch 10, 2nd layer
![image](/CIFAR-100/filter%20imgs/model100%200.png)  
epoch 100, 1st layer
![image](/CIFAR-100/filter%20imgs/model100%201.png)  
epoch 100, 2nd layer

# feature maps
![image](/CIFAR-100/feature%20map/fm1_0.png)
epoch 0, 1st layer
![image](/CIFAR-100/feature%20map/fm2_0.png)  
epoch 0, 2nd layer
![image](/CIFAR-100/feature%20map/fm1_10.png) 
epoch 10, 1st layer
![image](/CIFAR-100/feature%20map/fm2_10.png)  
epoch 10, 2nd layer
![image](/CIFAR-100/feature%20map/fm1_100.png)  
epoch 100, 1st layer
![image](/CIFAR-100/feature%20map/fm2_100.png)  
epoch 100, 2nd layer
