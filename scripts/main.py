from make_dataset import transform_data
from model import train_model

try:
    images, dataloaders, batch_size, class_names, dataset_sizes = transform_data()
    model = train_model(
        images, dataloaders, batch_size, class_names, dataset_sizes, num_epochs=10
    )
except Exception as e:
    print(e)
