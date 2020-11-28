![title](img/model_github.png)

Yao Fu, Chuanqi Tan, Bin Bi, Mosha Chen, Yansong Feng, Alexander Rush. Latent Template Induction with Gumbel-CRFs. NeurIPS 2020. ([pdf](https://github.com/FranxYao/Gumbel-CRF/blob/main/doc/gumbel_crf_camera_ready.pdf))

## Implementation 
* Gumbel-FFBS (Algorithm 2 in the paper): `src/modeling/structure/linear_crf.py` function `rsample`.
* Core model: `src/modeling/latent_temp_crf_ar.py`
* Training, validation, evaluation: `src/controller.py`
  * This file also serves as a general-purpose framework. New models can be added with minimum engineering. See `src/rnnlm.py` for an example
* Configuration: `src/config.py`

## Adding new models
* `src/rnnlm.py` gives a minimum example of adding new models with the existing framework

## Experiments

### Text modeling, Table 1 in the paper 

The following experiments are for reproducing Table 1 and Figure 3(B) in the paper. 

#### Text Modeling, Gumbel-CRF

```bash
nohup python main.py --model_name=latent_temp_crf_ar --dataset=e2e --task=density --model_version=1.0.3.1 --gpu_id=6 --latent_vocab_size=20 --z_beta=1e-3 --z_overlap_logits=False --use_copy=False --use_src_info=False --num_epoch=60 --validate_start_epoch=0 --num_sample_nll=100 --x_lambd_start_epoch=10 --x_lambd_anneal_epoch=2 --batch_size_train=100 --inspect_grad=False --inspect_model=True  > ../log/latent_temp_crf_ar.1.0.3.1  2>&1 & tail -f ../log/latent_temp_crf_ar.1.0.3.1
```

#### Text Modeling, REINFORCE

```bash
nohup python main.py --model_name=latent_temp_crf_ar --grad_estimator=score_func --dataset=e2e --task=density --model_version=2.0.0.1 --gpu_id=2 --latent_vocab_size=20 --z_beta=1.05 --z_gamma=0 --z_b0=0.1 --z_overlap_logits=False --use_copy=False --use_src_info=False --num_epoch=60 --validate_start_epoch=0 --batch_size_train=100 --num_sample_nll=100 --x_lambd_start_epoch=10 --x_lambd_anneal_epoch=2 > ../log/latent_temp_crf_ar.2.0.0.1  2>&1 & tail -f ../log/latent_temp_crf_ar.2.0.0.1
```

#### Text Modeling, PM-MRF

```bash
nohup python main.py --model_name=latent_temp_crf_ar --dataset=e2e --task=density --model_version=1.5.0.0 --gpu_id=5 --latent_vocab_size=20 --z_beta=1e-3 --z_sample_method=pm --z_overlap_logits=False --use_copy=False --use_src_info=False --num_epoch=60 --validate_start_epoch=0 --num_sample_nll=100 --tau_anneal_epoch=60 --x_lambd_start_epoch=10 --x_lambd_anneal_epoch=2 --batch_size_train=100 --inspect_grad=False --inspect_model=True  > ../log/latent_temp_crf_ar.1.5.0.0  2>&1 & tail -f ../log/latent_temp_crf_ar.1.5.0.0
```

### Paraphrase Generation, Table 3 in the paper 


#### Paraphrase Generation, Gumbel-CRF
```bash
nohup python main.py --model_name=latent_temp_crf_ar --dataset=mscoco --task=generation --model_version=1.3.1.0 --gpu_id=0 --latent_vocab_size=50 --z_beta=1e-3 --z_overlap_logits=False --use_copy=True --use_src_info=True --num_epoch=40 --validate_start_epoch=0 --validation_criteria=b2 --num_sample_nll=100 --x_lambd_start_epoch=0 --x_lambd_anneal_epoch=10 --batch_size_train=100 --batch_size_eval=100 --inspect_grad=False --inspect_model=True --write_full_predictions=True > ../log/latent_temp_crf_ar.1.3.1.0  2>&1 & tail -f ../log/latent_temp_crf_ar.1.3.1.0
```

#### Paraphrase Generation, REINFORCE
```bash
nohup python main.py --model_name=latent_temp_crf_ar --grad_estimator=score_func --dataset=mscoco --task=generation --model_version=2.5.0.0 --gpu_id=4 --latent_vocab_size=50 --z_beta=1.05 --z_gamma=0 --z_b0=0.1 --use_copy=True --use_src_info=True --num_epoch=40 --validate_start_epoch=0 --batch_size_train=100 --num_sample_nll=100 --x_lambd_start_epoch=10 --x_lambd_anneal_epoch=2 --validation_criteria=b4 --test_validate=true > ../log/latent_temp_crf_ar.2.5.0.0  2>&1 & tail -f ../log/latent_temp_crf_ar.2.5.0.0
```
### Data-to-text, Table 2 in the paper


#### Data-to-text, Gumbel-CRF
```bash
nohup python main.py --model_name=latent_temp_crf_ar --dataset=e2e --task=generation --model_version=1.2.0.1 --gpu_id=4 --latent_vocab_size=20 --z_beta=1e-3 --z_overlap_logits=False --use_copy=True --use_src_info=True --num_epoch=80 --validate_start_epoch=0 --validation_criteria=b2 --num_sample_nll=100 --x_lambd_start_epoch=0 --x_lambd_anneal_epoch=10 --batch_size_train=100 --inspect_grad=False --inspect_model=True --write_full_predictions=True --test_validate > ../log/latent_temp_crf_ar.1.2.0.1  2>&1 & tail -f ../log/latent_temp_crf_ar.1.2.0.1
```

#### Data-to-text, REINFORCE
```bash
nohup python main.py --model_name=latent_temp_crf_ar --grad_estimator=score_func --dataset=e2e --task=generation --model_version=2.2.0.1 --gpu_id=6 --latent_vocab_size=20 --z_beta=1.05 --z_gamma=0 --z_b0=0.1 --z_overlap_logits=False --use_copy=True --use_src_info=True --num_epoch=80 --validate_start_epoch=0 --validation_criteria=b4 --batch_size_train=100 --num_sample_nll=100 --x_lambd_start_epoch=0 --x_lambd_anneal_epoch=10 > ../log/latent_temp_crf_ar.2.2.0.1  2>&1 & tail -f ../log/latent_temp_crf_ar.2.2.0.1
```

### Figure 4 and Figure 5 
Results are produced from `template_manager.py` with some other scripts and the current implementation is slow and a bit of hacky. 
I'm cleaning and speeding it up so currently it is not very well integrated (or not quite runnable). 
You should be able to observe the produced templates during training, also reproduce the results with a bit of engineering.
But if you do encounter problems during coding, please contact me. 