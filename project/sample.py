import argparse
import os
import time
import numpy as np
import torch as th
import torch.distributed as dist
import project.read_data as read_data
from project.utils import dist_util, logger
from project.utils.script_util_duo import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)
import imageio


def main():
    args = create_argparser().parse_args()
    device = 'cuda' if th.cuda.is_available() else 'cpu'

    if args.data_type == "singlecoil":
        images, masks = read_data.get_us_singlecoil(args.data_path, R=args.R, contrast=args.contrast)
    elif args.data_type == "multicoil":
        images, masks, coil_maps = read_data.get_us_multicoil(args.data_path, R=args.R, contrast=args.contrast)

    dist_util.setup_dist()
    logger.configure(dir=args.save_path)

    logger.log("creating model and diffusion...")
    logger.log("STEP 1")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    logger.log("STEP 2")
    model.load_state_dict(
        dist_util.load_state_dict(args.model_path, map_location="cpu")
    )
    logger.log("STEP 3")
    #model.to(dist_util.dev())
    #model.to(device)
    logger.log("STEP 4")

    model.eval()
    logger.log("STEP 5")
    for index in range(len(images)):
        logger.log("In loop...")
        logger.log("Loop : STEP 1")
        mask = th.from_numpy(masks[np.newaxis,np.newaxis,index])
        # mask = th.from_numpy(masks[np.newaxis,np.newaxis,index]).cuda()
        logger.log("Loop : STEP 2")
        mask = th.cat([mask,mask],1)
        logger.log("Loop : STEP 3")

        kspace = load_data(images, index, args.batch_size, args.data_type)

        logger.log("Loop : STEP 4")

        if args.data_type == "multicoil":
            logger.log("Loop : STEP 5")
            #coil_map = th.from_numpy(coil_maps[np.newaxis, index, :, :, :]).cuda()
            coil_map = th.from_numpy(coil_maps[np.newaxis, index, :, :, :])
        else:
            coil_map = None

        logger.log("Loop : STEP 6")
        start_time = time.time()
        print("Index:", index)
        sample = diffusion.p_sample_loop_condition(
            model,
            (args.batch_size, 2, args.image_size, args.image_size),
            kspace,
            mask,
            coil_map
        )[-1]
        total_time = time.time() - start_time
        print("total time: " + str(total_time))
        logger.log("Loop : STEP 7")
        samples = []
        coarse = []
        samples.append(sample)
        optimizerG = th.optim.Adam(model.parameters(), lr=1e-4)
        logger.log("Loop : STEP 8")
        samples = th.cat(samples)
        coarse.append(samples.contiguous())
        coarse = th.stack(coarse)
        coarse_np = coarse.cpu().data.numpy()
        logger.log("Loop : STEP 9")
        np.save(os.path.join(args.save_path, "coarse" + str(index)  + ".npy"), coarse_np)
        logger.log("Loop : STEP 10")
        vis_1 = np.abs(coarse_np[0,-1,0,:,:] + coarse_np[0,-1,1,:,:]*1j)
        imageio.imsave(os.path.join(args.save_path, "image" + '_' + str(index) + '.png'), vis_1/vis_1.max())
        logger.log("Loop : STEP Finale")
        print(args.save_path)


def load_data(dataset, index, batch_size, data_type):

    img_prior = dataset[index]

    kspace1 = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(img_prior, axes=[-1,-2])), axes=[-1,-2])

    if data_type == "singlecoil":
        kspace = th.FloatTensor(np.stack([np.real(kspace1), np.imag(kspace1)])).cuda().view(1, 2, img_prior.shape[-2], img_prior.shape[-1]).repeat(batch_size, 1, 1, 1).float()
    elif data_type == "multicoil":
        #kspace = th.from_numpy(kspace1).cuda().view(1, 5, img_prior.shape[-2], img_prior.shape[-1]).repeat(batch_size, 1, 1, 1)
        kspace = th.from_numpy(kspace1).view(1, 5, img_prior.shape[-2], img_prior.shape[-1]).repeat(batch_size, 1, 1, 1)
    return kspace

def create_argparser():
    defaults = dict(
        num_samples=100,
        batch_size=5,
        use_ddim=False,
        model_path="",
        data_path="",
        save_path="",
        beta1 = 0.9,
        beta2 = 0.99,
        R = 4,
        contrast = "T1",
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
