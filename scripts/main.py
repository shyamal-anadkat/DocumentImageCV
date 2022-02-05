from make_dataset import transform_data
from model import train_model
import click


@click.command()
@click.option("--epochs", type=int, default=10)
@click.option("--justdataset", is_flag=True)
def run(epochs, justdataset):
    try:
        images, dataloaders, batch_size, class_names, dataset_sizes = transform_data()
        if not justdataset:
            train_model(
                images,
                dataloaders,
                batch_size,
                class_names,
                dataset_sizes,
                num_epochs=epochs,
            )
    except AttributeError as e:
        print(e)


if __name__ == "__main__":
    run()  # pylint: disable=no-value-for-parameter
