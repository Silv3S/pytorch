import pytest
import torch

@pytest.mark.parametrize("device", ["cpu", "xpu"])
def test_data_write_errors_under_transform(device):
        t = torch.randn(3, 3, device=device)

        def fn(t):
            t.data = torch.randn(3, 3)
            return t.sum()

        msg = "mutating directly with `.data` inside functorch transform is not allowed."
        with pytest.raises(RuntimeError, match=msg):
            torch.func.grad(fn)(t)

        with pytest.raises(RuntimeError, match=msg):
            torch.func.vjp(fn, t)

        with pytest.raises(RuntimeError, match=msg):
            torch.func.jvp(fn, (t,), (torch.randn_like(t),))