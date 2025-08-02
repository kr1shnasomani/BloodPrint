<h1 align="center">BloodPrint</h1>
<p align="center" style="margin-top:30px;">
  <img src="https://github.com/user-attachments/assets/766cc9e1-6ab8-4bc8-82e8-c59d37135531" height="300cm"/>
</p>
This project detects blood groups from fingerprint images using deep learning models. It employs a custom PyTorch model alongside EfficientNetB0 and ResNet50 built with TensorFlow to enhance accuracy and performance. The system leverages fingerprint patterns to predict blood types efficiently.

## Repository Structure:
The following is the repository structure:
```
BloodPrint/
├── code/
│   ├── PyTorch/
│   │   └── custom.ipynb
│   └── TensorFlow/
│       ├── densenet121.ipynb
│       ├── efficientnetb0.ipynb
│       ├── inceptionv3.ipynb
│       ├── mobilenetv2.ipynb
│       ├── resnet50.ipynb
│       └── vgg16.ipynb
├── dataset/
│   └── Dataset.md
├── README.md
└── requirements.txt
```

## Accuracy & Loss over Epochs:

| Model Name     | Accuracy                                                                                  | Loss                                                                                      |
|----------------|-------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------|
| Custom         | ![image](https://github.com/user-attachments/assets/f3580af2-18dd-495f-b041-2ee27bb07b2f) | ![image](https://github.com/user-attachments/assets/9c210832-d85a-4fe7-9e7c-70ae395b4b1c) |
| DenseNet121    | ![image](https://github.com/user-attachments/assets/31dcabbf-91c1-4b69-8acf-b0f719521d06) | ![image](https://github.com/user-attachments/assets/2bb400f1-21e4-4a79-8202-881561614a77) |
| EfficientNetB0 | ![image](https://github.com/user-attachments/assets/eb66a4bf-957a-49f8-a652-66c2872deb68) | ![image](https://github.com/user-attachments/assets/2d04d85b-23c9-4254-b744-d0e2eeb549f1) |
| InceptionV3    | ![image](https://github.com/user-attachments/assets/16b6a053-2228-49ed-9703-1a37328877f5) | ![image](https://github.com/user-attachments/assets/aefb4f17-d534-48d6-9490-ef4f13157230) |
| MobileNetV2    | ![image](https://github.com/user-attachments/assets/bccf05c2-f19b-4028-a188-045d63b0b445) | ![image](https://github.com/user-attachments/assets/f64b66a6-1bb1-4ea3-9860-803ca2f46492) |
| ResNet50       | ![image](https://github.com/user-attachments/assets/ee1cb5ba-5cb7-4c32-8493-cfb9b05c2d47) | ![image](https://github.com/user-attachments/assets/fce80be6-c54d-40da-8fef-35b89561569e) |
| VGG16          | ![image](https://github.com/user-attachments/assets/762ce0de-8e9e-4dc8-a4d4-1b8de486c81c) | ![image](https://github.com/user-attachments/assets/789ad6d6-7be8-4c13-ba78-2fbbef904226) |

## Classification Report:

#### Custom:
| Blood Type  | Precision   | Recall | F1-Score | Support |
|-------------|-------------|--------|----------|---------|
| A+          | 0.84        | 0.95   | 0.89     | 104     |
| A-          | 0.87        | 0.86   | 0.87     | 168     |
| AB+         | 0.96        | 0.75   | 0.84     | 131     |
| AB-         | 0.81        | 0.83   | 0.82     | 113     |
| B+          | 0.83        | 0.89   | 0.86     | 101     |
| B-          | 0.86        | 0.96   | 0.91     | 132     |
| O+          | 0.81        | 0.89   | 0.85     | 145     |
| O-          | 0.90        | 0.69   | 0.78     | 106     |

#### EfficientNetB0:
| Blood Type  | Precision   | Recall | F1-Score | Support |
|-------------|-------------|--------|----------|---------|
| A+          | 0.80        | 0.87   | 0.83     | 113     |
| A-          | 0.78        | 0.63   | 0.70     | 202     |
| AB+         | 0.74        | 0.85   | 0.79     | 142     |
| AB-         | 0.80        | 0.66   | 0.72     | 152     |
| B+          | 0.76        | 0.80   | 0.78     | 130     |
| B-          | 0.76        | 0.96   | 0.85     | 148     |
| O+          | 0.64        | 0.79   | 0.71     | 171     |
| O-          | 0.89        | 0.56   | 0.68     | 142     |

#### DenseNet121:
| Blood Type  | Precision   | Recall | F1-Score | Support |
|-------------|-------------|--------|----------|---------|
| A+          | 0.99        | 0.75   | 0.85     | 113     |
| A-          | 0.85        | 0.76   | 0.80     | 202     |
| AB+         | 0.74        | 0.85   | 0.79     | 142     |
| AB-         | 0.84        | 0.81   | 0.83     | 152     |
| B+          | 0.76        | 0.85   | 0.80     | 130     |
| B-          | 0.90        | 0.90   | 0.90     | 148     |
| O+          | 0.84        | 0.78   | 0.81     | 171     |
| O-          | 0.67        | 0.83   | 0.74     | 142     |

#### InceptionV3:
| Blood Type  | Precision   | Recall | F1-Score | Support |
|-------------|-------------|--------|----------|---------|
| A+          | 0.99        | 0.75   | 0.85     | 113     |
| A-          | 0.85        | 0.76   | 0.80     | 202     |
| AB+         | 0.74        | 0.85   | 0.79     | 142     |
| AB-         | 0.84        | 0.81   | 0.83     | 152     |
| B+          | 0.76        | 0.85   | 0.80     | 130     |
| B-          | 0.90        | 0.90   | 0.90     | 148     |
| O+          | 0.84        | 0.78   | 0.81     | 171     |
| O-          | 0.67        | 0.83   | 0.74     | 142     |

#### MobileNetV2:
| Blood Type  | Precision   | Recall | F1-Score | Support |
|-------------|-------------|--------|----------|---------|
| A+          | 0.99        | 0.75   | 0.85     | 113     |
| A-          | 0.85        | 0.76   | 0.80     | 202     |
| AB+         | 0.74        | 0.85   | 0.79     | 142     |
| AB-         | 0.84        | 0.81   | 0.83     | 152     |
| B+          | 0.76        | 0.85   | 0.80     | 130     |
| B-          | 0.90        | 0.90   | 0.90     | 148     |
| O+          | 0.84        | 0.78   | 0.81     | 171     |
| O-          | 0.67        | 0.83   | 0.74     | 142     |

#### ResNet50:
| Blood Type  | Precision   | Recall | F1-Score | Support |
|-------------|-------------|--------|----------|---------|
| A+          | 0.99        | 0.75   | 0.85     | 113     |
| A-          | 0.85        | 0.76   | 0.80     | 202     |
| AB+         | 0.74        | 0.85   | 0.79     | 142     |
| AB-         | 0.84        | 0.81   | 0.83     | 152     |
| B+          | 0.76        | 0.85   | 0.80     | 130     |
| B-          | 0.90        | 0.90   | 0.90     | 148     |
| O+          | 0.84        | 0.78   | 0.81     | 171     |
| O-          | 0.67        | 0.83   | 0.74     | 142     |

#### VGG16:
| Blood Type  | Precision   | Recall | F1-Score | Support |
|-------------|-------------|--------|----------|---------|
| A+          | 0.99        | 0.75   | 0.85     | 113     |
| A-          | 0.85        | 0.76   | 0.80     | 202     |
| AB+         | 0.74        | 0.85   | 0.79     | 142     |
| AB-         | 0.84        | 0.81   | 0.83     | 152     |
| B+          | 0.76        | 0.85   | 0.80     | 130     |
| B-          | 0.90        | 0.90   | 0.90     | 148     |
| O+          | 0.84        | 0.78   | 0.81     | 171     |
| O-          | 0.67        | 0.83   | 0.74     | 142     |

## Confusion Matrix:

| Model Name     | Plot                                                                                      |
|----------------|-------------------------------------------------------------------------------------------|
| Custom         | ![image](https://github.com/user-attachments/assets/51c82dfd-e1e3-42a6-8f1f-b2959768826d) |
| DenseNet121    | ![image](https://github.com/user-attachments/assets/6615675c-f607-4025-b3f0-a7bd42b5c2c3) |
| EfficientNetB0 | ![image](https://github.com/user-attachments/assets/95638317-671d-405a-b454-a35ae637028e) |
| InceptionV3    | ![image](https://github.com/user-attachments/assets/79d91ffe-3c92-4a0c-890a-c508fc9b5de9) |
| MobileNetV2    | ![image](https://github.com/user-attachments/assets/058a47d1-2258-4a40-b013-3c2acc2808fa) |
| ResNet50       | ![image](https://github.com/user-attachments/assets/93b79c39-2369-4b26-839b-a8513c94c9c8) |
| VGG16          | ![image](https://github.com/user-attachments/assets/3806036c-d1ed-4d8a-9b7f-7fc0b29d47fb) |

## Accuracy and Size Comparison:

|      Name      | Accuracy | Size (in MB) |
|----------------|----------|--------------|
| Custom         | 85%      | 74.4         |
| DenseNet121    | 86%      | 231.4        |
| EfficientNetB0 | 76%      | 19.1         |
| InceptionV3    | 79%      | 315.2        |
| MobileNetV2    | 84%      | 261.7        |
| ResNet50       | 81%      | 98.3         |
| VGG16          | 85%      | 160          |
