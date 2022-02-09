# Reverse-Image-Search-for-Fashion
Upload your image related with Fashion maybe cloth, watch, shoes and this WebApp will return five most similar items 

[Dataset Link](https://www.kaggle.com/paramaggarwal/fashion-product-images-dataset)

[Tensorflow Installation Guide](https://medium.com/@arsanatladkat/how-to-setup-tensorflow-2-3-1-cpu-gpu-windows-10-e000e7811e2b)

<center><h2>Prerequisite</h2></center>

1. `CNN`
2. `Transferlearning` ==> technique to use pretrained models
    1. `ResNET` model ==> trained on imageNET data

<center><h2>Required Crude Steps</h2></center>

1. `Import Model` ==> will import CNN model named as ResNET, this is trained on imageNET dataset
2. `Extract Features` ==> using ResNET model will extract features
    1. Let say we have n-number of images, now we have to compare our image with these n-number of images and model will give back say 8 or any number of required images which are most similar to our quary image
3. `Export Features` ==> save in some file to use in future
4. `Generate Recommendation` ==> using Euclidian Distance using Nearest Neighbour of sklearn
