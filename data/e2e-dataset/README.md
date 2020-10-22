
The E2E Challenge Dataset
=========================

_Authors: Jekaterina Novikova, Ondrej Dusek and Verena Rieser_

Description
-----------

The E2E dataset is a new dataset for training end-to-end, data-driven natural 
language generation systems in the restaurant domain, which is ten times bigger 
than existing, frequently used datasets in this area. 

The E2E dataset poses new challenges: 
1) its human reference texts show more lexical richness and syntactic variation, 
   including discourse phenomena;
2) generating from this set requires content selection. 

As such, learning from this dataset promises more natural, varied and less 
template-like system utterances.

The E2E set was used in the [E2E NLG Challenge](http://www.macs.hw.ac.uk/InteractionLab/E2E/),
which provides an extensive list of results achieved on this data.

Please refer to [our SIGDIAL2017 paper](https://arxiv.org/abs/1706.09254) for 
a detailed description of the dataset.

Contents
--------

### Files ###

* trainset.csv – the training set
* devset.csv – the development set
* testset.csv – the challenge test set (meaning representations only)
* testset_w_refs.csv – evaluation test set with reference natural language 
    utterances

### CSV Data Fields ###

- **mr** – textual meaning representation (MR)
- **ref** – corresponding natural language utterance (human reference)

Note that several human references correspond to a single MR, i.e., multiple 
lines contain the same MR.

Citing
------

If you use this dataset in your work, please cite the following paper:

```
@inproceedings{novikova2017e2e,
  title={The {E2E} Dataset: New Challenges for End-to-End Generation},
  author={Novikova, Jekaterina and Du{\v{s}}ek, Ondrej and Rieser, Verena},
  booktitle={Proceedings of the 18th Annual Meeting of the Special Interest 
             Group on Discourse and Dialogue},
  address={Saarbr\"ucken, Germany},
  year={2017},
  note={arXiv:1706.09254},
  url={https://arxiv.org/abs/1706.09254},
}
```

License
-------

Distributed under the [Creative Commons 4.0 Attribution-ShareAlike license
(CC4.0-BY-SA)](https://creativecommons.org/licenses/by-sa/4.0/).


Acknowledgements
----------------

This research received funding from the EPSRC projects DILiGENt (EP/M005429/1) and MaDrIgAL (EP/N017536/1).
