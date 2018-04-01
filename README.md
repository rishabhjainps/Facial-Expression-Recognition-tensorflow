# Facial Expression Recognition Using Tensorflow
Used Convolutional neural networks (CNN) for facial expression recognition . The goal is to classify each facial image into one of the seven facial emotion categories considered .


## Data :
We trained and tested our models on the data set from the [Kaggle Facial Expression Recognition Challenge](https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge), which comprises 48-by-48-pixel grayscale images of human faces,each labeled with one of 7 emotion categories:<strong> anger, disgust, fear, happiness, sadness, surprise, and neutral </strong>.
<br><br>
 Image set of 35,887 examples, with training-set : dev-set: test-set as <strong> 80 : 10 : 10 </strong>.

## Dependencies
 Python 2.7, Tensorflow, numpy .

## Library Used:
  <ul>
    <li> Tensorflow </li>
	  <li> numpy </li>
  </ul>

## Train

  To run the code -

  1. Download FER2013 dataset from [Kaggle Facial Expression Recognition Challenge](https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge) and extract in the <strong> data/ </strong> folder.

  2. After downloading dataset separate dataset into different csv for train ,dev and test.
     Run separate_data.py in data folder
    <pre>
    python separate_data.py
    </pre>

  3. To train the model run train.py file
    <pre>
    python train.py
    </pre>
    Separate Model weights for each epoch is saved in model/ folder

  4.  Run evaluate.py to get accuracy on test data. <br><br>
      "./model/model" + str(epochNumber) + ".ckpt" -> load specific epoch Model weight.
      <br>Change this line in evaluate.py to choose which model weights should be loaded
  <pre>
  saver.restore(sess, "./model/model100.ckpt")
  </pre>
