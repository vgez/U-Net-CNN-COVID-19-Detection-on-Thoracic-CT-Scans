# The Effect of Data Augmentation on COVID-19 Thoracic CT Images for Semantic Segmentation Using a Deep U-net Convolutional Neural Network

Final project in the course DD2424 Deep Learning for Data Science at KTH Royal Institute of Technology. Project was completed in the early months of the COVID-19 pandemic and hence we decided to focus on contributing to the efforts of detecting the virus through thoracic CT-scans using deep learning technologies. Our specific project focused on analyzing the effects of data augmentation in the semantic segmentation task, using a modified U-Net CNN architecture, as the size of the available, annotated datasets at this point in time were very small.

## Team Members

<ul>
    <li>
        <strong>Maximilian Auer</strong> - <i style="text-decoration: none;">maue@kth.se</i>
    </li>
    <li>
        <strong>Kristin Evegård</strong> - <i style="text-decoration: none;">evegard@kth.se</i>
    </li>  
    <li>
        <strong>Lukas Frösslund</strong> - <i style="text-decoration: none;">lukasfro@kth.se</i>
    </li>
    <li>
        <strong>Valdemar Gezelius</strong> - <i style="text-decoration: none;">vgez@kth.se</i>
    </li>  
</ul>

## Technologies

-   [Python 3](https://www.python.org/)
-   [TensorFlow 2](https://www.tensorflow.org/)
-   [Keras](https://keras.io/)
-   [Scikit-learn](https://sklearn.org/)

## Project Details

Basic augmentation methods such as flips, rotations, shifts, scaling and zooms were used and compared, along with slighly more complex elastic deformations. Dropout layers and Batch Normalization were utilized and thoroughly contrasted.

Binary cross-entropy loss, Weighted binary cross-entropy loss and Dice loss were all implemented. Binary cross-entropy loss showed the most promise. Dice score, Sensitivity and Specificity was used as our main evaluation metrics.

Results were encouraging, with a Dice score of ~ 0.886.

More details available in <a href="https://github.com/Frosslund/U-Net-CNN-COVID-19-Detection-on-Thoracic-CT-Scans/blob/main/Project_Report.pdf">project report</a>

## Qualitative Results

![qual_comparison](https://github.com/Frosslund/U-Net-CNN-COVID-19-Detection-on-Thoracic-CT-Scans/blob/main/images/qual_comparison.png?raw=true)
