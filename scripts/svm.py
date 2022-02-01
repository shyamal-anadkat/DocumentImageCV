from sklearn.metrics import classification_report,accuracy_score,confusion_matrix,plot_confusion_matrix
import pickle
from thundersvm import SVC
from make_dataset_svm import transform_data
train_dataset_array, train_dataset_label, val_dataset_array, val_dataset_label, images, dataloaders, batch_size, class_names, dataset_sizes = transform_data()
x_train = train_dataset_array
y_train = train_dataset_label
x_val = val_dataset_array
y_val = val_dataset_label
model = SVC(C=100, kernel='rbf')
model.fit(x_train,y_train)
y_pred=model.predict(x_val)
print(f"The model is {accuracy_score(y_pred,y_val)*100}% accurate")
plot_confusion_matrix(model,x_val,y_val)
