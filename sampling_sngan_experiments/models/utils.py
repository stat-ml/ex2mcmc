from torch import nn


class Discriminator_logits(nn.Module):
    def __init__(self, discriminator_sigmoid, ngpu):
        super().__init__()
        self.ngpu = ngpu
        self.main = discriminator_sigmoid.main[:-1]

    def forward(self, input):
        if input.is_cuda and self.ngpu > 1:
            output = nn.parallel.data_parallel(
                self.main, input, range(self.ngpu)
            )
        else:
            output = self.main(input)

        return output.view(-1, 1).squeeze(1)
