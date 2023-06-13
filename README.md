# CNN-Visualization
CNN featuremaps, filters Visualization with pytorch  
dataset: MNIST  
model: very simple conv net

내가 한 이상한 짓:
1. test_loader의 dataset에 train 넣고 validation을 함(근데 왜 train, test 그래프가 다르게 나오지)
2. 모델 conv_layer에 activation 함수 안넣고 학습함(아마 그래서 학습 전 후 filter 이미지가 유사하게 나온듯?)

결과:
filter img의 경우 처음 init 상태와 학습 후에 큰 차이가 없어 보인다.  
feature map의 경우 첫 레이어는 입력 이미지의 형상이 잘 보이지만 두번째 레이어에서는 입력 이미지의 형상을 찾아보기 어려웠다.  
