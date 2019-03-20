Text Summarization
++++++++++++++++++

This repository contains the code for our work:

Kurniawan, K., & Louvan, S. (2018). IndoSum: A New Benchmark Dataset for Indonesian Text Summarization. In 2018 International Conference on Asian Language Processing (IALP) (pp. 215–220). Bandung, Indonesia: IEEE. https://doi.org/10.1109/IALP.2018.8629109

Requirements
============

Create a virtual environment from ``environment.yml`` file using conda::

    $ conda env create -f environment.yml

To run experiments with NeuralSum [CL16]_, Tensorflow is also required.

Dataset
=======

Get the dataset from https://drive.google.com/file/d/1OgYbPfXFAv3TbwP1Qcwt_CC9cVWSJaco/view.

Preprocessing for NeuralSum
---------------------------

For NeuralSum, the dataset should be further preprocessed using ``prep_oracle_neuralsum.py``::

    $ ./prep_oracle_neuralsum.py -o neuralsum train.01.jsonl

The command will put the oracle files for NeuralSum under ``neuralsum`` directory. Invoke the script with ``-h/--help`` to see its other options.

Running experiments
===================

The scripts to run the experiments are named ``run_<model>.py``. For instance, to run an experiment using LEAD, the script to use is ``run_lead.py``. All scripts use `Sacred <https://sacred.readthedocs.io>`_ so you can invoke each with ``help`` command to see its usage. The experiment configurations are fully documented. Run ``./run_<model>.py print_config`` to print all the available configurations and their docs.

Training a model
----------------

To train a model, for example the naive Bayes model, run ``print_config`` command first to see the available configurations::

    $ ./run_bayes.py print_config

This command will give an output something like::

    INFO - summarization-bayes-testrun - Running command 'print_config'
    INFO - summarization-bayes-testrun - Started
    Configuration (modified, added, typechanged, doc):
      cutoff = 0.1                       # proportion of words with highest TF-IDF score to be considered important words
      idf_path = None                    # path to a pickle file containing the IDF dictionary
      model_path = 'model'               # where to load or save the trained model
      seed = 313680915                   # the random seed for this experiment
      corpus:
        dev = None                       # path to dev oracle JSONL file
        encoding = 'utf-8'               # file encoding
        lower = True                     # whether to lowercase words
        remove_puncts = True             # whether to remove punctuations
        replace_digits = True            # whether to replace digits
        stopwords_path = None            # path to stopwords file, one per each line
        test = 'test.jsonl'              # path to test oracle JSONL file
        train = 'train.jsonl'            # path to train oracle JSONL file
      eval:
        delete_temps = True              # whether to delete temp files after finishes
        on = 'test'                      # which corpus set the evaluation should be run on
        size = 3                         # extract at most this number of sentences as summary
      summ:
        path = 'test.jsonl'              # path to the JSONL file to summarize
        size = 3                         # extract at most this number of sentences as summary
    INFO - summarization-bayes-testrun - Completed after 0:00:00

So, to train the model on a train corpus in ``/tmp/train.jsonl`` and save the model to ``/tmp/models/bayes.model``, simply run::

    $ ./run_bayes.py train with corpus.train=/tmp/train.jsonl model_path=/tmp/models/bayes.model

Evaluating a model
------------------

Evaluating an unsupervised model is simple. For example, to evaluate a LEAD-N summarizer::

    $ ./run_lead.py evaluate with corpus.test=/tmp/test.jsonl

This command will print an output like this::

    INFO - run_experiment - Running command 'evaluate'
    INFO - run_experiment - Started
    INFO - read_jsonl - Reading test JSONL file from /tmp/test.jsonl
    INFO - evaluate - References directory: /var/folders/p9/4pp5smf946q9xtdwyx792cn40000gn/T/tmp7jct3ede
    INFO - evaluate - Hypotheses directory: /var/folders/p9/4pp5smf946q9xtdwyx792cn40000gn/T/tmpnaqoav4o
    INFO - evaluate - ROUGE scores: {'ROUGE-1-R': 0.71752, 'ROUGE-1-F': 0.63514, 'ROUGE-2-R': 0.62384, 'ROUGE-2-F': 0.5502, 'ROUGE-L-R': 0.70998, 'ROUGE-L-F': 0.62853}
    INFO - evaluate - Deleting temporary files and directories
    INFO - run_experiment - Result: 0.63514
    INFO - run_experiment - Completed after 0:00:11

Evaluating a trained model is done similarly with ``model_path`` configuration is set to the path to the saved model.

Setting up Mongodb observer
---------------------------

Sacred allows the experiments to be observed and saved to a Mongodb database. The experiment scripts above can readily be used for this, simply set two environment variables ``SACRED_MONGO_URL`` and ``SACRED_DB_NAME`` to your Mongodb authentication string and database name (to save the experiments into) respectively. Once set, the experiments will be saved to the database. Use ``-u`` flag when invoking the experiment script to disable saving.

Reproducing results
-------------------

All best configurations obtained from tuning on the development set are saved as Sacred's named configurations. This makes it easy to reproduce our results. For instance, to reproduce our LexRank result on fold 1, simply run::

    ./run_lexrank.py evaluate with tuned_on_fold1 corpus.test=test.01.jsonl

Since the best configuration is named as ``tuned_on_fold1``, the command above will use that configuration and evaluate the model on the test set. In general, all run scripts have ``tuned_on_foldX`` named configuration, where ``X`` is the fold number. For ``run_neuralsum.py`` though, there are other named configurations, namely ``emb300_on_foldX`` and ``fasttext_on_foldX``, referring to the scenario of using word embedding size of 300 and fastText pretrained embedding respectively. Some run scripts do not have such named configurations; that is because their hyperparameters were not tuned/they do not have any.

License
=======

Apache License, Version 2.0.

Citation
========

If you're using our code or dataset, please cite::

    @inproceedings{kurniawan2018,
      place={Bandung, Indonesia},
      title={IndoSum: A New Benchmark Dataset for Indonesian Text Summarization},
      url={https://ieeexplore.ieee.org/document/8629109},
      DOI={10.1109/IALP.2018.8629109},
      booktitle={2018 International Conference on Asian Language Processing (IALP)},
      publisher={IEEE},
      author={Kurniawan, Kemal and Louvan, Samuel},
      year={2018},
      month={Nov},
      pages={215-220}
    }


.. [CL16] Cheng, J., & Lapata, M. (2016). Neural summarization by extracting sentences and words. In Proceedings of the 54th Annual Meeting of the Association for Computational Linguistics (pp. 484–494). Berlin, Germany: Association for Computational Linguistics. Retrieved from http://www.aclweb.org/anthology/P16-1046
