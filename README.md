
## About

pyMeta is a library to manage machine learning problems as `Tasks' and to sample from Task distributions. Includes Tensorflow implementation of FOMAML and Reptile. <b>New support has been added for the recently published [implicit-MAML (iMAML) (Rajeswaran, Finn, Kakade and Levine (2019), "Meta-Learning with Implicit Gradients"](https://arxiv.org/abs/1909.04630)!</b>

<b>If you use *pyMeta* in your research and would like to cite the library, we suggest you cite [link](https://arxiv.org/abs/1909.04170):</b>
```
Giacomo Spigler. Meta-learnt priors slow down catastrophic forgetting in neural networks. arXiv preprint arXiv:1909.04170, 2019.
```

Bibtex:
```
@ARTICLE{spigler2019,
  author = {{Spigler}, Giacomo},
  title = "{Meta-learnt priors slow down catastrophic forgetting in neural networks}",
  journal = {arXiv e-prints},
  year = "2019",
  month = "Sep",
  eid = {arXiv:1909.04170},
  pages = {arXiv:1909.04170},
  archivePrefix = {arXiv},
  eprint = {1909.04170},
}
```

## Data

Acquire data using the 'fetch_data.sh' script from https://github.com/openai/supervised-reptile, then process it into a single pkl file using the scripts 'scripts/make_miniimagenet_dataset.py' and 'scripts/make_omniglot_dataset.py'.

## Usage

Broad requirements:
+ tested with Python 3.6 and Tensorflow 1.14
+ For the preliminary distributed implementation: mpi4py

For 5-way, 5-shot training on the Omniglot and MinImageNet datasets, you can try the following configurations. For FOMAML:

```
Sinusoid:
python3 example_metatrain.py --dataset="sinusoid" --metamodel="fomaml" \
    --num_train_samples_per_class=10 --num_test_samples_per_class=100 --num_inner_training_iterations=5 --inner_batch_size=10 \
    --meta_lr=0.001 --inner_lr=0.01 --meta_batch_size=5 --num_validation_batches=10 \
    --model_save_filename="saved/model.h5" --num_outer_metatraining_iterations=10000

Omniglot:
python3 example_metatrain.py --dataset="omniglot" --metamodel="fomaml" \
    --num_output_classes=5 --num_train_samples_per_class=5 --num_test_samples_per_class=15 --num_inner_training_iterations=5 --inner_batch_size=-1 \
    --meta_lr=0.001 --inner_lr=0.01 --meta_batch_size=5 --num_validation_batches=10 \
    --model_save_filename="saved/model.h5" --num_outer_metatraining_iterations=30000

Mini-ImageNet:
python3 example_metatrain.py --dataset="miniimagenet" --metamodel="fomaml" \
    --num_output_classes=5 --num_train_samples_per_class=5 --num_test_samples_per_class=15 --num_inner_training_iterations=5 --inner_batch_size=-1 \
    --meta_lr=0.001 --inner_lr=0.01 --meta_batch_size=5 --num_validation_batches=10 \
    --model_save_filename="saved/model.h5" --num_outer_metatraining_iterations=30000
```

For Reptile:
```
Sinusoid:
python3 example_metatrain.py --dataset="sinusoid" --metamodel="reptile" \
    --num_train_samples_per_class=10 --num_test_samples_per_class=100 --num_inner_training_iterations=5 --inner_batch_size=10 \
    --meta_lr=0.001 --inner_lr=0.01 --meta_batch_size=5 --num_validation_batches=10 \
    --model_save_filename="saved/model.h5" --num_outer_metatraining_iterations=10000

Omniglot:
python3 example_metatrain.py --dataset="omniglot" --metamodel="reptile" \
    --num_output_classes=5 --num_train_samples_per_class=10 --num_test_samples_per_class=10 --num_inner_training_iterations=5 --inner_batch_size=-1 \
    --meta_lr=0.1 --inner_lr=0.001 --meta_batch_size=5 --num_validation_batches=10 \
    --model_save_filename="saved/model.h5" --num_outer_metatraining_iterations=30000

Mini-ImageNet:
python3 example_metatrain.py --dataset="miniimagenet" --metamodel="reptile" \
    --num_output_classes=5 --num_train_samples_per_class=15 --num_test_samples_per_class=15 --num_inner_training_iterations=8 --inner_batch_size=-1 \
    --meta_lr=0.1 --inner_lr=0.001 --meta_batch_size=5 --num_validation_batches=10 \
    --model_save_filename="saved/model.h5" --num_outer_metatraining_iterations=30000
```

For i-MAML:
```
Mini-ImageNet:
python3 example_metatrain.py --dataset="miniimagenet" --metamodel="imaml" \
    --num_output_classes=5 --num_train_samples_per_class=5 --num_test_samples_per_class=15 --num_inner_training_iterations=5 --inner_batch_size=-1 \
    --meta_lr=0.001 --inner_lr=0.01 --meta_batch_size=5 --num_validation_batches=10 \
    --model_save_filename="saved/model.h5" --num_outer_metatraining_iterations=30000
```


## Planned improvements

+ Support for reinforcement learning tasks: "models" (neural networks) should be wrapped by "agents", and a new base task type RLTask should be introduced, that can perform model updates by running interactions between the supplied agent and an environment for a specific number of iterations / episodes. Special RLTask-derived classes could be OpenAIGymTask.


