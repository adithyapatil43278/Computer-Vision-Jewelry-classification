from preprocess import get_datasets
from base_model import get_base_model
from add_head import build_model
from compile_fit import train, finetune
from evaluate_visualize import evaluate, plot_history


IMG_SIZE = (300,300)
BATCH = 16
EPOCHS = 15

TRAIN_DIR = "image dataset/CV_Dataset_1800/TRAIN_1500"
TEST_DIR = "image dataset/CV_Dataset_1800/TEST_300"

train_ds, val_ds, test_ds, class_names, num_classes, aug = get_datasets(
    TRAIN_DIR, TEST_DIR, IMG_SIZE, BATCH, 42
)

base = get_base_model(IMG_SIZE)
model = build_model(base, aug, IMG_SIZE, num_classes)

h1 = train(model, train_ds, val_ds, EPOCHS)
h2 = finetune(model, base, train_ds, val_ds, EPOCHS)

evaluate(model, test_ds, class_names)
plot_history(h1, h2, EPOCHS)

#model saving
import os

MODEL_PATH = "models/jewelry_classifier.h5"

os.makedirs("models", exist_ok=True)

model.save(MODEL_PATH)

print(f"Model saved to {MODEL_PATH}")
