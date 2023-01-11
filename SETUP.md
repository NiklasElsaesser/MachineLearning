# Setup File to run the training in Google Colab
1. Clone labelImg from https://github.com/heartexlabs/labelImg
2. cd into the labelImg Repository.
3. Follow the steps as shown in the documentation.
4. Put the images which shall be labeled into a separate folder (ideally already in the build environment).
5. Put the YoloEuroImages/renaming.ipynb file in the folder with the images and run it.
6. Start labelImg and open the folder with the images.
7. Set the file format to *YOLO* before saving the first labeled picture.
8. Label all images coherently.
9. Now you have a *classes.txt* file, save this as *classes.names* in the "All-Types (*. *)"-Format. (ideally with Notepad++)
10. Open the */YoloEuroImages/creating-files-data-and-name.py* and *YoloEuroImages/creating-train-and-test-txt-files.py* files in YoloEuroImages and put the absolute path to the folder to you pictures in the according line.4K
11. Check if *labelled_data.data*, *test.txt* and *train.txt* have the required path to the images inside. It might be necessary to split the pictures manually 80:20 and put the paths to 80% of the pictures in *train.txt* manually. The same goes for *test.txt*. 
12. Now open the *CoinRecognitionF.ipynb* Notebook in Google Colab and follow the steps there.