import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report

def evaluate(model, test_ds, class_names):

    y_true = np.concatenate(
        [np.argmax(y, axis=1) for x, y in test_ds],
        axis=0
    )

    y_pred = np.argmax(model.predict(test_ds), axis=1)

    print(confusion_matrix(y_true, y_pred))
    print(classification_report(y_true, y_pred, target_names=class_names))


def plot_history(h1, h2, initial_epochs):

    acc = h1.history['accuracy'] + h2.history['accuracy']
    val_acc = h1.history['val_accuracy'] + h2.history['val_accuracy']
    loss = h1.history['loss'] + h2.history['loss']
    val_loss = h1.history['val_loss'] + h2.history['val_loss']

    plt.figure(figsize=(12,6))

    plt.subplot(1,2,1)
    plt.plot(acc)
    plt.plot(val_acc)
    plt.axvline(initial_epochs-1, linestyle='--')
    plt.title("Accuracy")

    plt.subplot(1,2,2)
    plt.plot(loss)
    plt.plot(val_loss)
    plt.axvline(initial_epochs-1, linestyle='--')
    plt.title("Loss")

    plt.show()
