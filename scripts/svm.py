from sklearn.metrics import accuracy_score, plot_confusion_matrix
import matplotlib.pyplot as plt
from thundersvm import SVC
from make_dataset_svm import transform_data

# get dataset for training
(
    train_dataset_array,
    train_dataset_label,
    val_dataset_array,
    val_dataset_label,
    images,
    dataloaders,
    batch_size,
    class_names,
    dataset_sizes,
) = transform_data()
x_train = train_dataset_array
y_train = train_dataset_label
x_val = val_dataset_array
y_val = val_dataset_label

# define model
model = SVC(C=100, kernel="rbf")

# train model
model.fit(x_train, y_train)

# predict output with validation dataset
y_pred = model.predict(x_val)
print(f"The model is {accuracy_score(y_pred,y_val)*100}% accurate")
plot_confusion_matrix(model, x_val, y_val)
plt.show()
