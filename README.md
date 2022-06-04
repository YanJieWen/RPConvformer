# RPConvformer
Improved Transformer based on Tensoflow implementation for traffic flow predictive modeling


## Contents

- [Background](#background)
- [Preliminary](#preliminary)
	- [Dataset](#dataset)
	- [Weight](#weight)
- [Training](#training)
- [Testing](#testing)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Background

  Accurate prediction of traffic flow can significantly improve the operational efficiency of traffic.
  Previous modeling models of the timing of traffic flow have the problems of inability to parallelize processing sequence nodes, inability to expand the fixation of input sequences, and insufficient interpretability. 
  Thus, we propose a novel framework called RPConvformer to solve the above problems, and it is an improved structure of the transformer. 
  Specifically, RPConvformer has an encoding-decoding structure, in the sequence embedding module we adopt causal 1Dconvolution for capturing the local correlation of time series, the encoder module is responsible for encoding historical traffic flow information,
  and information of any length of input sequence can be processed via the key mask (KME), and the decoder autoregressively predicts the future traffic state. 
  RPConvformer has three attention modules, which are divided into input sequence self-attention, output sequence self-attention, and input-output interactive-attention. They all have multiple headers, and the calculation method is scaled dot-product attention (SDPA) In addition, when calculating the attention score, we introduce the relative position bias to consider the relative position information of the internal nodes in the sequence. The calculated attention score matrix is interpretable.
  Extensive experiments on two real-world traffic flow datasets demonstrate the superiority of our model over several state-of-art methods.
## Preliminary

### Dataset

Before training, please run the [data_store.py](data_store.py) to generate PEMS04 and PEMSD8 data files respectively.pkl.
Moreover, if you want to change something, you can change the hyperparameters in [Hyperparameter.py](Hyperparameter.py) (Line 16& Line17), and you also
can change the [data_store.py](data_store.py)(Line 67& Line 68), the second parameters for different links and the third parameters for different traffic features,
such as flow, occupy, and speed.
```
https://drive.google.com/drive/folders/1HUs8BI9rMqP8PGABVzVsQc4o2Tcf6ki2?usp=sharing
```
With tree datasets: `chunk_occlusion_voc`, `HiEve_test`,`VOCdevkit`.Please put them in the root directory of the project

### Weight
We have trained a model for PEMS04 and PEMD8, You can download the weight of the traffic flow prediction model from 
```
https://drive.google.com/drive/folders/1-_66IvjOCkkDUquVEqbJ4FaOZPKi1uw6?usp=sharing
```
Three folders: `ckpt_pems04new`(trained for PEMS04),`ckpt_pems08new`(trained for PEMSD8),`logdir`(running record).When you want to test the PEMS04 testing datasets.
please copy the files in `ckpt_pems04new` to a brand new folder named `ckpt`, the same operation as the PEMSD8. In addtion, if you want to see the runningr record.
you can open the terminal and run the following statement:
```
cd RPConvformer
tensorboard --logdir=logdir
```
and then copy the  the address to open in Google Chrome.However，sometimes you may be failure to open it, just relax, and close network connection.,replay it.


## Training
The backbone RPconvformer
![image](pc/strut.png)

The [moudels.py](moudels.py) and [framework.py](framework.py) are the most important componets in this project. Moerover, You can come up with some innovative and great ideas if you like.So you can finally train the network by running the following command:
```
python [train.py]train.py
```


If you want to training your own model, you need to change the [train,py](train.py), the line 34,35(classes), line 119(which model to use,we apply two,one is original YOLOv3 and DLA). if you want to use original YOLOv3, you can change the line  as :
```
model_body = yolo_body(image_input, num_anchors//3, num_classes,'orginal_yolo')
```

## Testing 

After a long and hard training, you will get a good pedestrian detection model, stored in file logs/000/, and you need to copy it to the file [model_data](model_data).

We wrote a test file [test.py](yolo3/test.py), which matches the grount truth through IOU and confidence. It will generate a table file under the project folder.Your weight file (`line 23 `) should correspond to the model structure file (`line 46`). At last, We run the [cal_ap.py](yolo3/cal_ap.py), it will generate a complete AP record excel file and output the value of AP.

**ap_eval.csv**

TP | FP | Confi | iou
----|----|----|----
0 |	1| 0.3176| 0.2955
0 |	1| 0.4064| 0.3597
1 |	0| 0.4765| 0.8158

**output_yolo3coco.csv**

TP | FP | Confi | iou|acc_tp | acc_fp |precision |recall
----|----|----|----|----|----|----|----
0 |	1| 0.9865| 0.6260| 0| 1| 0| 0
0 |	1| 0.9854| 0.7169| 0| 2| 0| 0
0 |	1| 0.9854| 0.5567| 0| 3| 0| 0
1 |	0| 0.9846| 0.7818| 1| 3| 0.25| 0.001031
## Contributing

Most of the code comes from [qqwweee/keras-yolo3](https://github.com/qqwweee/keras-yolo3)

The image annotation tool we use is [labelImg](https://github.com/tzutalin/labelImg)

The inspiration of this article comes from this [paper](https://arxiv.org/abs/1707.06484)

At the same time, we are also very grateful to [Lin et al](https://arxiv.org/abs/2005.04490). For collecting pedestrian data of complex events

At last, thank you very much for the contribution of the co-author in the article, and also thank my girlfriend for giving me the courage to pursue for a Ph.d.


![image](picture/FIG3.png)

## License

[MIT](LICENSE) © YanjieWen

