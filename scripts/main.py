from make_dataset import transform_data
from model import train_model
import click
import torch


@click.command()
@click.option("--epochs", type=int, default=10)
@click.option("--onlytransform", is_flag=True)
def run(epochs, onlytransform):
    try:
        images, dataloaders, batch_size, class_names, dataset_sizes = transform_data()
        if not onlytransform:
            model = train_model(
                images,
                dataloaders,
                batch_size,
                class_names,
                dataset_sizes,
                num_epochs=epochs,
            )
            print('Saving model in "models" ------')
            torch.save(model.state_dict(), "../models/model.pt")
    except AttributeError as e:
        print(e)


if __name__ == "__main__":
    run()  # pylint: disable=no-value-for-parameter
