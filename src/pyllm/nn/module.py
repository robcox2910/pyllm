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
        """Collect every learnable Tensor in this module and its children (each once).

        Like taking a register of all the players in the squad -- if the same
        player is listed under two positions, they still only train once. We use
        each Tensor's identity (`id`) to make sure no one gets called up twice.
        """
        found = []
        seen = set()

        def _add(tensor):
            if id(tensor) not in seen:
                seen.add(id(tensor))
                found.append(tensor)

        for value in self.__dict__.values():
            if isinstance(value, Tensor):
                _add(value)
            elif isinstance(value, Module):
                for param in value.parameters():
                    _add(param)
            elif isinstance(value, (list, tuple)):
                for item in value:
                    if isinstance(item, Module):
                        for param in item.parameters():
                            _add(param)
                    elif isinstance(item, Tensor):
                        _add(item)
        return found

    def zero_grad(self):
        """Wipe the slate clean before the next learning step."""
        for param in self.parameters():
            param.zero_grad()

    def train(self, mode=True):
        """Switch this module and all its children into training mode (or eval).

        Like telling the whole team "this is practice" (train) or "this is the
        real match" (eval). Dropout listens: it only sits players out in practice.
        """
        self.training = mode
        for value in self.__dict__.values():
            if isinstance(value, Module):
                value.train(mode)
            elif isinstance(value, (list, tuple)):
                for item in value:
                    if isinstance(item, Module):
                        item.train(mode)
        return self

    def eval(self):
        """Switch to evaluation mode -- the real match, where every player is on."""
        return self.train(False)

    def forward(self, *args, **kwargs):
        """Turn input into output. Subclasses must override this."""
        raise NotImplementedError("Subclasses must implement forward")

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)
