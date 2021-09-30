# Pneumonia_Detection_using_transfer_learning

## Installation Setup

<ul>
  <li>install the following libraries for training purposes</li>
  <ul>
    <li>Keras</li>
    <li>Numpy</li>
    <li>sklearn</li>
  </ul>
  <li>install the following libraries for server</li>
  <ul>
    <li>Flask</li>
    <li>Numpy</li>
    <li>cv2</li>
    <li>keras</li>
  </ul>
</ul>

## Model training

1. Clone this Repository
2. Download the dataset for Chest X-Ray from Kaggle
3. In the "model" folder you will find 3 different folders of the algorithms.
4. Run the .ipynb files of whichever algorithm you wish to run.(Install the libraries if not installed.)
5. Run it's respective "predict" .ipynb file for prediction.

## Website
1. Open "app.html" file from "UI" folder in google chrome.
2. Run "server.py" from "server" folder
3. Upload the chest X-Ray image in "app.html" and click classify button.
<br>Note: VGG16 has been used for website.