# When will you do what? - Anticipating Temporal Occurrences of Activities

This repository provides a TensorFlow implementation of the paper [When will you do what? - Anticipating Temporal Occurrences of Activities](https://pages.iai.uni-bonn.de/gall_juergen/download/jgall_anticipation_cvpr18.pdf).

### Qualitative Results:

Click on the image.

<div align="center">
  <a href="https://www.youtube.com/watch?v=xMNYRcVH_oI"><img src="https://img.youtube.com/vi/xMNYRcVH_oI/0.jpg" alt="IMAGE ALT TEXT"></a>
</div>

### Training:

* download the data from https://uni-bonn.sciebo.de/s/3Wyqu3cxYSm47Kg.
* extract it so that you have the `data` folder in the same directory as `main.py`.
* To train the model on split1 of Breakfast dataset run `python main.py --model=MODEL --action=train  --vid_list_file=./data/train.split1.bundle` where `MODEL` is `cnn` or `rnn`.
* To change the default saving directory or the model parameters, check the list of options by running `python main.py -h`.

### Prediction:

* Run `python main.py --model=MODEL --action=predict  --vid_list_file=./data/test.split1.bundle` for evaluating the the model on split1 of Breakfast. 
* To predict from ground truth observation set `--input_type` option to `gt`. 
* To check the list of options run `python main.py -h`.

### Evaluation:

Run `python eval.py --obs_perc=OBS-PERC --recog_dir=RESULTS-DIR`. Where `RESULTS-DIR` contains the output predictions for a specific observation and prediction percentage, and `OBS-PERC` is the corresponding observation percentage. For example `python eval.py --obs_perc=.3 --recog_dir=./save_dir/results/rnn/obs0.3-pred0.5` will evaluate the output corresponding to 0.3 observation and 0.5 prediction.

### Remarks:

If you use the code, please cite

    Y. Abu Farha, A. Richard, J. Gall:
    When will you do what? - Anticipating Temporal Occurrences of Activities
    in IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2018

To download the used features please visit:
[An end-to-end generative framework for video segmentation and recognition](https://hildekuehne.github.io/projects/end2end/index.html).
