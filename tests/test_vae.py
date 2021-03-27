import pytest
import torch
from src.models.vae import VAE
import torch.nn as nn

VAE_SIZES = [10, 5, 3]
CLASSIFIER_SIZES = [3, 3, 2]
SAMPLES = 2


@pytest.fixture
def model():
    return VAE(VAE_SIZES, CLASSIFIER_SIZES)


def test_initializer(model):
    encoder = model.encoder

    # Check encoder layer sizes
    for i in range(len(VAE_SIZES) - 2):
        expected_input = VAE_SIZES[i]
        expected_output = VAE_SIZES[i + 1]
        assert encoder[i * 2].in_features == expected_input
        assert encoder[i * 2].out_features == expected_output
        assert isinstance(encoder[i * 2 + 1], nn.ReLU)

    assert model.mean.in_features == VAE_SIZES[-2]
    assert model.mean.out_features == VAE_SIZES[-1]
    assert model.logvar.in_features == VAE_SIZES[-2]
    assert model.logvar.out_features == VAE_SIZES[-1]

    # Check decoder layer sizes
    decoder = model.decoder

    for i in range(1, len(VAE_SIZES)):
        expected_input = VAE_SIZES[-i]
        expected_output = VAE_SIZES[-i - 1]
        assert decoder[(i - 1) * 2].in_features == expected_input
        assert decoder[(i - 1) * 2].out_features == expected_output
        assert isinstance(decoder[(i - 1) * 2 + 1], nn.ReLU)

    # Check classifier layer sizes
    classifier = model.classifier

    assert classifier.layer_sizes == CLASSIFIER_SIZES[:-1]
    assert classifier.output_units == CLASSIFIER_SIZES[-1]


def test_encode(model):
    x = torch.rand(SAMPLES, VAE_SIZES[0])

    mean, logvar = model.encode(x)

    # Check mean sizes
    assert mean.size(0) == SAMPLES
    assert mean.size(1) == VAE_SIZES[-1]

    # Check logvar sizes
    assert logvar.size(0) == SAMPLES
    assert logvar.size(1) == VAE_SIZES[-1]


def test_reparametrize(monkeypatch, model):
    # Set randn_like output to a tensor of ones
    def mock_randn_like(tensor):
        return torch.ones(tensor.size())

    # torch.exp retrieves the input tensor
    def mock_exp(tensor):
        return tensor

    # Set monkeypatchs
    monkeypatch.setattr(torch, "randn_like", mock_randn_like)
    monkeypatch.setattr(torch, "exp", mock_exp)

    # Initialize tensors
    size = [1, 5]
    mean = torch.rand(size)
    logvar = torch.rand(size)

    # Execute reparametrize
    latent_code = model.reparametrize(mean, logvar)

    # Check the result
    is_equal = (latent_code == (mean + 0.5 * logvar)).all()
    assert is_equal


def test_decode(model):
    x = torch.rand(SAMPLES, VAE_SIZES[-1])

    output = model.decode(x)

    # Check output size
    assert output.size(0) == SAMPLES
    assert output.size(1) == VAE_SIZES[0]


def test_classify(model):
    x = torch.rand(SAMPLES, VAE_SIZES[-1])

    output = model.classify(x)

    # Check output size
    assert output.size(0) == SAMPLES
    assert output.size(1) == CLASSIFIER_SIZES[-1]

    # Check the Softmax output
    output_sum = output.sum(dim=1)
    assert torch.allclose(output_sum, torch.ones(output_sum.size()))


def test_forward(monkeypatch, mocker, model):
    # Set randn_like output to a tensor of ones
    def mock_randn_like(tensor):
        return torch.ones(tensor.size())

    # Set monkeypatchs
    monkeypatch.setattr(torch, "randn_like", mock_randn_like)

    x = torch.rand(SAMPLES, VAE_SIZES[0])

    # Get some expected values
    expected_mean, expected_logvar = model.encode(x)
    latent_code = model.reparametrize(expected_mean, expected_logvar)

    # Initialize mocks
    encode_spy = mocker.spy(model, "encode")
    reparametrize_spy = mocker.spy(model, "reparametrize")

    # Call the method that is being currently tested
    decoded, classification, computed_mean, computed_logvar = model.forward(x)

    # Spy assertions
    encode_spy.assert_called_once_with(x)
    reparametrize_spy.assert_called_once()

    # Assertions about the retrieved values
    assert decoded.equal(model.decode(latent_code))
    assert classification.equal(model.classify(latent_code))
    assert expected_mean.equal(computed_mean)
    assert expected_logvar.equal(computed_logvar)


def test_training_step(model):
    # Fake loss function
    def loss_func(decoder_output, data, classifier_output, label, mean, logvar):
        return torch.ones(data.size(1))

    x = torch.rand(SAMPLES, VAE_SIZES[0])
    y = torch.ones(SAMPLES)

    batch = (x, y)

    loss = model.training_step(batch, loss_func)

    assert loss.equal(torch.ones(x.size(1)))


def test_validation_step(model):
    # Fake loss function
    def loss_func(decoder_output, data, classifier_output, label, mean, logvar):
        return torch.ones(data.size(1))

    x = torch.rand(SAMPLES, VAE_SIZES[0])
    y = torch.ones(SAMPLES)

    batch = (x, y)

    loss = model.validation_step(batch, loss_func)

    assert loss.equal(torch.ones(x.size(1)))
