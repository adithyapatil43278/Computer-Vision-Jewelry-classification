# Jewelry Image Classification (Computer Vision)

This project classifies jewelry images into six classes using computer vision.

## Classes
- Anklets
- Bangles
- Earring
- Necklace
- Ring
- Tikka

## Dataset Structure
The dataset is organized into folders by class. The main dataset lives under [image dataset/](image%20dataset/).

```
image dataset/
  CV_Dataset_1800/
    TRAIN_1500/
      Anklets/
      Bangles/
      Earring/
      Necklace/
      Ring/
      Tikka/
    TEST_300/
      Anklets/
      Bangles/
      Earring/
      Necklace/
      Ring/
      Tikka/
  Raw_Dataset_1088/
    Anklets_130/
    Bangles_186/
    Earring_191/
    Necklace_225/
    Ring_196/
    Tikka_160/
```

### Notes
- **CV_Dataset_1800**: Curated dataset split into **TRAIN_1500** and **TEST_300** folders, each containing the six class folders.
- **Raw_Dataset_1088**: Original raw collection organized by class with item counts in folder names.

## Notebooks
- [jewelry_CV.ipynb](jewelry_CV.ipynb)
- [jewelry_CV_final.ipynb](jewelry_CV_final.ipynb)

## How to Use
1. Load images from the class folders in [image dataset/](image%20dataset/).
2. Use the training split in **CV_Dataset_1800/TRAIN_1500** and evaluate on **CV_Dataset_1800/TEST_300**.
3. Run the notebooks to train and evaluate your model.

**Methods**

- **Dataset loading**: uses `tf.keras.utils.image_dataset_from_directory` with `validation_split`, `subset`, `seed`, `image_size=(300,300)`, and `batch_size=16` to create `train`, `validation`, and `test` datasets.
- **Preprocessing**: applies `tensorflow.keras.applications.efficientnet.preprocess_input` in a `map()` step and converts labels to one-hot with `tf.one_hot`.
- **Data augmentation**: uses an augmentation `Sequential` with `layers.RandomFlip("horizontal")`, `layers.RandomRotation(0.1)`, and `layers.RandomZoom(0.1)` applied on the input pipeline.
- **Performance tuning**: uses `tf.data.AUTOTUNE`, `ds.shuffle(1000)`, and `ds.prefetch(AUTOTUNE)` for pipeline performance.
- **Base model**: `EfficientNetB3` from `tf.keras.applications` with `include_top=False`, pretrained `weights='imagenet'`, and input shape `(300,300,3)`.
- **Model architecture**: built with the Keras Functional API: input -> augmentation -> `base_model` -> `GlobalAveragePooling2D` -> `BatchNormalization` -> `Dense(256, relu)` -> `Dropout(0.5)` -> output `Dense(num_classes, softmax)`.
- **Transfer learning & fine-tuning**: initially freezes the base model (`base_model.trainable = False`) then unfreezes and fine-tunes the top layers (unfreeze last ~30 layers) with a lower learning rate.
- **Loss / optimizer / metrics**: `CategoricalCrossentropy(label_smoothing=0.1)`, `Adam` optimizer (initial lr=1e-3, fine-tune lr=1e-5), and metric `accuracy`.
- **Training**: trains for `15` epochs for both initial training and fine-tuning (as implemented), with optional `EarlyStopping` callbacks commented out.
- **Evaluation & reporting**: evaluates on the test set, computes `confusion_matrix` and `classification_report` (from `sklearn`), and plots combined training and fine-tuning histories.

See the notebook for full implementation details: [jewelry_CV_final.ipynb](jewelry_CV_final.ipynb)
