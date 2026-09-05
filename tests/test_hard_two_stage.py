import inspect

import torch

from models.baselines import HardTwoStageModel


def _model() -> HardTwoStageModel:
    return HardTwoStageModel(
        ["DatasetA", "DatasetB"],
        input_dim=5,
        latent_dim=4,
        num_classes=3,
        id_hidden_dims=[7],
        stage_b_hidden_dims=[7],
        activation="tanh",
        dropout=0.0,
    )


def test_forward_is_dataset_blind_and_exposes_predicted_route():
    model = _model().eval()
    params = [name for name in inspect.signature(model.forward).parameters if name != "self"]
    assert params == ["x"]

    out = model(torch.randn(6, 5))
    assert out["logits"].shape == (6, 3)
    assert out["dataset_logits"].shape == (6, 2)
    assert torch.equal(out["dataset_pred"], out["dataset_logits"].argmax(dim=1))


def test_stage_a_and_each_stage_b_submodel_share_no_parameters():
    model = _model()
    parameter_ids = {
        "stage_a": {id(p) for p in (*model.id_encoder.parameters(), *model.id_head.parameters())},
        **{
            name: {
                id(p)
                for p in (
                    *model.stage_b.encoders[name].parameters(),
                    *model.stage_b.heads[name].parameters(),
                )
            }
            for name in model.dataset_names
        },
    }
    groups = list(parameter_ids.items())
    for index, (left_name, left) in enumerate(groups):
        for right_name, right in groups[index + 1 :]:
            assert left.isdisjoint(right), f"{left_name} unexpectedly shares parameters with {right_name}"


def test_argmax_route_executes_only_the_selected_stage_b_submodel():
    model = _model().eval()
    calls = {name: 0 for name in model.dataset_names}
    hooks = []
    for name in model.dataset_names:
        hooks.append(
            model.stage_b.encoders[name].register_forward_hook(
                lambda _module, _inputs, _output, dataset=name: calls.__setitem__(dataset, calls[dataset] + 1)
            )
        )

    # Force every row to DatasetB regardless of x. This is a predicted route,
    # not a ground-truth dataset ID passed to forward.
    with torch.no_grad():
        model.id_head.linear.weight.zero_()
        model.id_head.linear.bias.copy_(torch.tensor([-1.0, 1.0]))
        out = model(torch.randn(8, 5))

    for hook in hooks:
        hook.remove()
    assert out["dataset_pred"].tolist() == [1] * 8
    assert calls == {"DatasetA": 0, "DatasetB": 1}
