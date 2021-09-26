# IConvLSTM
一個融合Inception機制於ConvLSTM卷積長短期網路卷積運算的模型演算法。
使用Pytorch 編寫而成，傳統CNN與LSTM模型只能捉取純圖像或時間上的特徵，而ConvLSTM將CNN卷積運算融合至LSTM門閥運算中，使其擁有同時捉取時間與空間特徵，然而與CNN有同樣的問題-固定的filter大小，因此對於空間上的特徵捕取有限，因此本模型融入了Inception機制，使其能同時捕捉多種尺度特徵，使捕捉能力更加全面。
其中Inception區塊部分使用了1x1、3x3、5x5、7x7尺寸卷積運算，並使用堆疊卷積與深度可分離卷積（Depthwise Separable Convolution )達到減少參數量的效果，由Inception區塊取代原始ConvLSTM卷積運算部分，並透過1個1X1CNN輸出門閥狀態。
![image](https://github.com/andy970139/IConvLSTM/blob/main/Inception.PNG)
調整後的公式參考如下:
![image](https://github.com/andy970139/IConvLSTM/blob/main/%E5%85%AC%E5%BC%8F.PNG)
