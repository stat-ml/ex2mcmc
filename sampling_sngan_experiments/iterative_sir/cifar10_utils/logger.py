# Code referenced from https://gist.github.com/gyglim/1f8dfb1b5c82627ae3efcfbbadb9f514
import datetime
import sys

import numpy as np
from general_utils import send_file_to_remote
from torchvision.utils import save_image


sys.path.append("../sampling_utils")


class Logger:
    def __init__(
        self,
        log_file=None,
        plot_dir=None,
        port_to_remote=None,
        path_to_save_remote=None,
    ):
        self.log_file = log_file
        self.plot_dir = plot_dir
        self.port_to_remote = port_to_remote
        self.path_to_save_remote = path_to_save_remote

    def scalar_summary(self, tag, value, epoch):
        f = open(self.log_file, "a+")
        f.write(f"epoch = {epoch}, {tag} = {value}\n")
        f.close()

    def images_summary(self, tag, images, step):
        """Log a list of images."""
        cur_time = datetime.datetime.now().strftime("%Y_%m_%d-%H_%M_%S")
        path_to_images = "{}/{}_step_{}.pdf".format(
            self.plot_dir,
            cur_time,
            step,
        )
        save_image(images.data, path_to_images, nrow=8, normalize=True)
        send_file_to_remote(
            path_to_images,
            self.port_to_remote,
            self.path_to_save_remote,
        )
