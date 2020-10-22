## Implementation 
* Gumbel-FFBS: `src/modeling/structure/linear_crf.py` line 195
* PM-MRF: `src/modeling/structure/linear_crf.py` line 146
* Our model with reparameterized estimators: `src/modeling/latent_temp_crf.py`
* Our model with score function estimators: `src/modeling/latent_temp_crf_rl.py`

## Experiments

#### Gumbel-CRF, Text Modeling

```bash
python main.py --model_name=latent_temp_crf --dataset=e2e --task=density --model_version=1.0 --num_epoch=40 --gpu_id=0 --z_sample_method=gumbel_ffbs --z_beta=1e-4 --latent_vocab_size=20 --z_gamma=0 --z_overlap_logits=True --z_tau_final=1.0 --gumbel_st=False --use_copy=True --dec_adaptive=False --auto_regressive=False --post_process_start_epoch=50 --temp_rank_strategy=random --validate_start_epoch=0 --post_process_sampling_enc=True --num_sample_nll=5
```

#### Gumbel-CRF, Paraphrase Generation 
```bash
python main.py  --model_name=latent_temp_crf  --dataset=mscoco  --model_version=2.0  --gpu_id=0  --latent_vocab_size=50  --z_beta=1e-4  --z_gamma=0.  --z_overlap_logits=True  --z_sample_method=gumbel_ffbs  --gumbel_st=False  --z_tau_final=1.  --use_copy=True  --dec_adaptive=False  --auto_regressive=True  --temp_rank_strategy=topk  --num_sample=3  --validate_start_epoch=5  --post_process_sampling_enc=True  --post_process_start_epoch=0  --post_noise_p=0.5  --x_lambd_start_epoch=10  --validation_criteria=b4  --write_full_predictions=True 
```


#### Gumbel-CRF, Data-to-text
```bash
python main.py  --model_name=latent_temp_crf  --dataset=e2e  --model_version=3.0  --gpu_id=5  --latent_vocab_size=50  --z_beta=1e-4  --z_gamma=0.  --z_overlap_logits=True  --z_sample_method=gumbel_ffbs  --gumbel_st=False  --z_tau_final=1.  --use_copy=True  --dec_adaptive=False  --auto_regressive=True  --temp_rank_strategy=topk  --num_sample=3  --validate_start_epoch=5  --post_process_sampling_enc=True  --post_process_start_epoch=0  --post_noise_p=0.5  --x_lambd_start_epoch=10  --validation_criteria=post_b4  --write_full_predictions=True 
```