import torch
import torch.nn as nn

from models.models import Generator, Discriminator
from models.losses import GeneratorLoss, DiscriminatorLoss


def run(input_params, data_params, model_params):
    batch_size = data_params["batch_size"]
    num_channels = data_params["num_channels"]
    mel_filters = data_params["mel_filters"]
    timeframes = data_params["timeframes"]
    audio_length = data_params["audio_length"]
    shape = (batch_size, num_channels, mel_filters, timeframes)


if __name__ == "__main__":
    # Data Preparation
    data_params = {
        "batch_size": 16,
        "num_channels": 1,
        "mel_filters": 128,
        "timeframes": 100,
        "audio_length": 47998
    }

    batch_size = data_params["batch_size"]
    num_channels = data_params["num_channels"]
    mel_filters = data_params["mel_filters"]
    timeframes = data_params["timeframes"]
    audio_length = data_params["audio_length"]
    shape = (batch_size, num_channels, mel_filters, timeframes)

    # Create random input tensor with the shape (batch_size, num_channels, mel_filters, timeframes)
    x_mel = torch.randn(batch_size, num_channels, mel_filters, timeframes)
    print("Mel Initiator Input Shape: ", x_mel.shape)

    params = {
        "generator_params": {
            "mel_initiator_params": {
                "in_channels": 1,
                "out_channels": 32,
                "kernel_size": 3,
                "lstm_hidden_size": 64,
                "lstm_num_layers": 2,
                "dpn_depth": 3,
                "audio_length": 47998,
                "input_shape": shape
            },
            "meta_initiator_params": {
                "input_size": 10,
                "hidden_size": 64,
                "output_size": 32
            },
        },
        "discriminator_params": {
            "msd_params": {
                "params_avgpool": {
                    "kernel_size": 11,
                    "stride": 4
                },
                "params_distributor": {
                    "defconv_in_channels": 1,
                    "defconv_out_channels": 16,
                    "kernel_size": 3,
                    "kernel_list": [3, 5],
                    "stride": 1,
                    "depth": 4
                },
                "params_final_msd": {
                    "kernel_size": 7,
                    "stride": 4,
                    "hidden_dims": 512
                }
            },
            "mcd_params": {
                "initiator_params": {
                    "height": 103,
                    "width": 466
                },
                "convolver_params": {
                    "in_channels": 1,
                    "out_channels": 8,
                    "kernel_size": 3,
                    "kernel_list": [3, 5],
                    "stride": 1,
                    "depth": 3
                },
                "final_params": {
                    "in_features": 0,  # To be modified later
                    "hidden_features": 512,
                    "out_features": 2
                }
            }
        }

    }

    # Generator Params
    params_generator = params["generator_params"]

    # Discriminator Params
    discriminator_params = params["discriminator_params"]

    # Define model and input parameters
    batch_size = 16
    input_size = 10  # Input size

    # Generate random input
    x_meta = torch.randn(batch_size, input_size)
    print("Meta Initiator Input Shape: ", x_meta.shape)

    generator = Generator(params=params_generator)

    out_gen = generator(x_mel, x_meta)
    print("Generator Output Shape: ", out_gen.shape)

    # Inputs
    generated_samples = torch.randn(batch_size, 1, audio_length)
    real_samples = torch.randn(batch_size, 1, audio_length)

    # Labels
    # labels_generated = torch.zeros(batch_size, dtype=torch.long)  # Label 0
    # labels_real = torch.ones(batch_size, dtype=torch.long)  # Label 1

    discriminator_model = Discriminator(params=discriminator_params)

    discriminator_output_msd_generated, discriminator_output_mcd_generated = discriminator_model(generated_samples)
    discriminator_output_msd_real, discriminator_output_mcd_real = discriminator_output_msd_generated, discriminator_output_mcd_generated  # discriminator_model(generated_samples)
    print("Discriminator MSD Output Shape: ", discriminator_output_msd_generated.shape)
    print("Discriminator MCD Output Shape: ", discriminator_output_mcd_generated.shape)

    # generator_loss = GeneratorLoss(discriminator=discriminator_model)
    # discriminator_loss = DiscriminatorLoss()
    #
    # # Calculate the generator loss
    # gen_loss = generator_loss(real_samples, generated_samples, x_mel, discriminator_output_msd_generated)
    # print(f"Generator Loss: {gen_loss.item()}")
    #
    # # Calculate the discriminator loss
    # disc_loss = discriminator_loss(discriminator_output_msd_real, discriminator_output_msd_generated)
    # print(f"Discriminator Loss: {disc_loss.item()}")
