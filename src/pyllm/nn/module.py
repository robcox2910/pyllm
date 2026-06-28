from pyllm.autograd import Tensor


class Module:
    """A reusable Lego brick of a neural network.

    A Module holds some learnable numbers (its *parameters*) and knows how to
    turn an input into an output (`forward`). Bricks can hold other bricks, so a
    whole network is just one big brick made of smaller ones. `parameters()`
    walks the whole tower and hands back every learnable number so the optimizer
    can nudge them all.
    """

    def parameters(self):
        """Collect every learnable Tensor in this module and its children."""
        found = []
        for value in self.__dict__.values():
            if isinstance(value, Tensor):
                found.append(value)
            elif isinstance(value, Module):
                found.extend(value.parameters())
            elif isinstance(value, (list, tuple)):
                for item in value:
                    if isinstance(item, Module):
                        found.extend(item.parameters())
                    elif isinstance(item, Tensor):
                        found.append(item)
        return found

    def zero_grad(self):
        """Wipe the slate clean before the next learning step."""
        for param in self.parameters():
            param.zero_grad()

    def forward(self, *args, **kwargs):
        """Turn input into output. Subclasses must override this."""
        raise NotImplementedError("Subclasses must implement forward")

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)
