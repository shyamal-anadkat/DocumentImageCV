from make_dataset import transform_data
from model import train_model

images, dataloaders, batch_size, class_names, dataset_sizes = transform_data()
model = train_model(images, dataloaders, batch_size, class_names, dataset_sizes, num_epochs=10)