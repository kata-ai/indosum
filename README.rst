Text Summarization
++++++++++++++++++

This repository contains the code for our work:

Kemal Kurniawan and Samuel Louvan. *IndoSum: A New Benchmark Dataset for Indonesian Text Summarization*.
In Proceedings of the International Conference on Asian Language Processing. 2018.

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

The scripts to run the experiments are named ``run_<model>.py``. For instance, to run an experiment using LEAD, the script to use is ``run_lead.py``. All scripts use `Sacred <https://sacred.readthedocs.io>`_ so you can invoke each with ``help`` command to see its usage. The experiment configurations are fully documented. Run ``./run_<model>.py print_config`` to print all the available configurations and their docs. An example on how to evaluate a LEAD-N summarizer::

    $ ./run_lead.py evaluate with corpus.test=test.01.jsonl

This command will print an output like this::

    INFO - run_experiment - Running command 'evaluate'
    INFO - run_experiment - Started
    INFO - read_jsonl - Reading test JSONL file from test.01.jsonl
    INFO - evaluate - References directory: /var/folders/p9/4pp5smf946q9xtdwyx792cn40000gn/T/tmp7jct3ede
    INFO - evaluate - Hypotheses directory: /var/folders/p9/4pp5smf946q9xtdwyx792cn40000gn/T/tmpnaqoav4o
    INFO - evaluate - ROUGE scores: {'ROUGE-1-R': 0.71752, 'ROUGE-1-F': 0.63514, 'ROUGE-2-R': 0.62384, 'ROUGE-2-F': 0.5502, 'ROUGE-L-R': 0.70998, 'ROUGE-L-F': 0.62853}
    INFO - evaluate - Deleting temporary files and directories
    INFO - run_experiment - Result: 0.63514
    INFO - run_experiment - Completed after 0:00:11

Setting up Mongodb observer
---------------------------

Sacred allows the experiments to be observed and saved to a Mongodb database. The experiment scripts above can readily be used for this, simply set two environment variables ``SACRED_MONGO_URL`` and ``SACRED_DB_NAME`` to your Mongodb authentication string and database name (to save the experiments into) respectively. Once set, the experiments will be saved to the database. Use ``-u`` flag when invoking the experiment script to disable saving.

Reproducing results
-------------------

All best configurations obtained from tuning on the development set are saved as Sacred's named configurations. This makes it easy to reproduce our results. For instance, to reproduce our LexRank result on fold 1, simply run::

    ./run_lexrank.py evaluate with tuned_on_fold1 corpus.test=test.01.jsonl

Since the best configuration is named as ``tuned_on_fold1``, the command above will use that configuration and evaluate the model on the test set. In general, all run scripts have ``tuned_on_foldX`` named configuration, where ``X`` is the fold number. For ``run_neuralsum.py`` though, there are other named configurations, namely ``emb300_on_foldX`` and ``fasttext_on_foldX``, referring to the scenario of using word embedding size of 300 and fastText pretrained embedding respectively. Some run scripts do not have such named configurations; that is because their hyperparameters were not tuned/they do not have any.


.. [CL16] Cheng, J., & Lapata, M. (2016). Neural summarization by extracting sentences and words. In Proceedings of the 54th Annual Meeting of the Association for Computational Linguistics (pp. 484–494). Berlin, Germany: Association for Computational Linguistics. Retrieved from http://www.aclweb.org/anthology/P16-1046
