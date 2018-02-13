# Traffic Flow Prediction
Traffic Flow Prediction with Neural Networks(SAEs、LSTM、GRU).

## Requirement
- Python 3.5    
- Tensorflow-gpu 1.2.0  
- Keras 2.1.3
- scikit-learn 0.18

## Train the model

**Run command below to train the model:**

```
python train.py --model model_name
```

You can choose "lstm", "gru" or "saes" as arguments. The ```.h5``` weight file was saved at model folder.


## Experiment

Data are obtained from the Caltrans Performance Measurement System (PeMS). Data are collected in real-time from individual detectors spanning the freeway system across all major metropolitan areas of the State of California.
	
	device: Tesla K80
	dataset: PeMS 5min-interval traffic flow data
	optimizer: RMSprop(lr=0.001, rho=0.9, epsilon=1e-06)
	batch_szie: 256 


**Run command below to run the program:**

```
python main.py
```

These are the details for the traffic flow prediction experiment.


| Metrics | MAE | MSE | RMSE | MAPE |  R2  | Explained variance score |
| ------- |:---:| :--:| :--: | :--: | :--: | :----------------------: |
| LSTM | 7.16 | 94.20 | 9.71 | 21.25% | 0.9420 | 0.9421 |
| GRU | 7.18 | 95.01 | 9.75| 17.42% | 0.9415 | 0.9415 |
| SAEs | 7.71 | 106.46 | 10.32 | 25.62% | 0.9344 | 0.9352 |

![evaluate](/images/eva.png)

## Reference

	@article{SAEs,  
	  title={Traffic Flow Prediction With Big Data: A Deep Learning Approach},  
	  author={Y Lv, Y Duan, W Kang, Z Li, FY Wang},
	  journal={IEEE Transactions on Intelligent Transportation Systems, 2015, 16(2):865-873},
	  year={2015}
	}
	
	@article{RNN,  
	  title={Using LSTM and GRU neural network methods for traffic flow prediction},  
	  author={R Fu, Z Zhang, L Li},
	  journal={Chinese Association of Automation, 2017:324-328},
	  year={2017}
	}


## Copyright
See [LICENSE](LICENSE) for details.
