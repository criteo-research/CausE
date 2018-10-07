# CausE

Python code for the RecSys 2018 paper entitled 'Causal Embeddings for Recommendation' using Tensorflow. A pre-print version of the paper can be found here - https://arxiv.org/abs/1706.07639


## Code Organisation
The code is organized as follows:
- **causal_prod2vec.py** - Used to run the CausE-avg method from the paper, where users response under the uniform exposure is averaged into a single vector.
- **causal_prod2vec2i.py** - Used to run the CausE-prod-T and CausE-prod-C methods from the paper, where users response under the uniform exposure is mapped into a separate product matrix.
- **models.py** - SupervisedProd2Vec and CausalProd2Vec as tensorflow model classes.
- **dataset_loading.py** - File to load Movielens/Netflix datasets and convert to user/product matrix.
- **utils.py** - Various helper methods.

## Dependencies and Requirements
The code has been designed to support python 3.6+ only. The project has the following dependences and version requirements:

- python3 - 3.6.5+
- tensorflow - 1.9.0+
- tensorboard - 1.9.0+
- numpy - 1.14.5+
- scipy - 1.1+
- pandas - 0.22+

## Training a Model

After the data has been downloaded, preprocessed and placed in the Data directory, the models can then be training simply by running **python3 causal_prod2vec.py** or **python3 causal_prod2vec2i.py** from the src directory. The various hyper-parameters of the models can be controlled via arguments detailed below. 

### Optional arguments

The code uses TF flags to manage the command line arguments for running the models, please note that both models have the same parameters.

> Causal Embeddings for Recommendation Parameters 
>
>optional arguments: <br />
> --data_set : The dataset to load <br />
>  --model_name : The name of the model used when saving (default: cp2v) <br />
>  --logging_dir : Where to save tensorboard data (default: /tmp/tensorboard/) <br />
>  --learning_rate : The learning rate for SGD (default: 1.0) <br />
>  --l2_pen : The weight decay regularization amount (default: 0.0) <br />
>  --num_epochs : The number of epochs to train for (default: 10) <br />
>  --batch_size : The batch size for SGD (default: 512) <br />
>  --num_steps : The number of batches after which to print information (default: 500) <br />
>  --early_stopping_enabled : If to use early stopping (default: False) <br />
>  --early_stopping : 'Tolerance for early stopping (# of epochs).' (default: 200) <br />
>  --embedding_size : The embeddings dimension of the product and user embeddings (default: 50) <br />
>  --cf_pen : The weighting for the counter-factual regularization (default: 1.0) <br />
>  --cf_distance : Which distance metric to use for the counter-factual regularization (default: l1) <br />


## Cite

Please cite the associated paper for this work if you use this code:

```
@article{bonner2017causal,
  title={Causal Embeddings for Recommendation},
  author={Bonner, Stephen and Vasile, Flavian},
  journal={arXiv preprint arXiv:1706.07639},
  year={2017}
}
```


## License

Copyright CRITEO

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and limitations under the License.
