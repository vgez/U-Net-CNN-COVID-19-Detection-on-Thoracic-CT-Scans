# The Effect of Data Augmentation on COVID-19 Thoracic CT Images for Semantic Segmentation Using a Deep U-net Convolutional Neural Network

The final project in the course **DD2424 Deep Learning for Data Science** at KTH Royal Institute of Technology. The project was completed in the early months of the _COVID-19_ pandemic and hence we decided to focus on contributing to the efforts of detecting the virus through thoracic _CT-scans_ using deep learning technologies. Our specific project focused on analyzing the effects of _data augmentation_ in the semantic segmentation task, using a modified _U-Net CNN_ architecture, as the size of the available, annotated datasets, in those early days of the pandemic, were very small.

## Team Members

<ul>
    <li>
        <strong>Valdemar Gezelius</strong>
    </li>
    <li>
        <strong>Lukas Frösslund</strong>
    </li>
    <li>
        <strong>Maximilian Auer</strong>
    </li>
    <li>
        <strong>Kristin Evegård</strong>
    </li>  
</ul>

## Technologies

-   [Python3](https://www.python.org/)
-   [TensorFlow2](https://www.tensorflow.org/)
-   [Keras](https://keras.io/)
-   [Scikit-learn](https://sklearn.org/)

## Project Details

Basic augmentation methods such as _flips_, _rotations_, _shifts_, _scaling_ and _zooms_ were used and compared, along with slighly more complex _elastic deformations_. Dropout layers and _Batch Normalization_ were utilized and thoroughly contrasted.

_Binary cross-entropy loss_, _Weighted binary cross-entropy loss_ and _Dice loss_ were all implemented. Binary cross-entropy loss showed the most promise. _Dice score_, _Sensitivity_ and _Specificity_ was used as our main evaluation metrics.

Results were encouraging, with a **Dice score of ~ 0.886**.

More details available in <a href="https://github.com/vgez/U-Net-CNN-COVID-19-Detection-on-Thoracic-CT-Scans/blob/main/Project_Report.pdf">project report</a>.

## Qualitative Results

![qual_comparison](https://github.com/vgez/U-Net-CNN-COVID-19-Detection-on-Thoracic-CT-Scans/blob/main/images/qual_comparison.png?raw=true)
