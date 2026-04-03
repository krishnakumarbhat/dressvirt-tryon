"""Lightweight smoke-test trainer that simulates epochs without heavy dependencies.

This script is safe to run on CPU and does not import the project's model code.
It generates synthetic images, iterates for N epochs, and writes a small checkpoint file.
"""
import argparse
import os
import time
import numpy as np


def generate_synthetic_image(w=64, h=64):
    return np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_epochs', type=int, default=3, help='number of epochs to simulate')
    parser.add_argument('--num_samples', type=int, default=4, help='number of synthetic samples per epoch')
    parser.add_argument('--batch_size', type=int, default=1, help='batch size (only used for reporting)')
    parser.add_argument('--name', type=str, default='smoke_experiment', help='experiment name')
    args = parser.parse_args()

    ckpt_dir = os.path.join('checkpoints', args.name)
    os.makedirs(ckpt_dir, exist_ok=True)

    total_iters = 0
    print(f"Starting smoke-run: epochs={args.n_epochs}, samples/epoch={args.num_samples}, batch_size={args.batch_size}")
    for epoch in range(1, args.n_epochs + 1):
        epoch_start = time.time()
        for i in range(args.num_samples):
            total_iters += 1
            img = generate_synthetic_image()
            # dummy "training" compute (mean)
            loss = float(img.mean())
            print(f"[{args.name}][epoch {epoch}][iter {total_iters}] loss: {loss:.4f}")
            # optional small sleep to simulate work
            # time.sleep(0.01)

        # write latest_iter checkpoint
        with open(os.path.join(ckpt_dir, 'latest_iter.txt'), 'w') as f:
            f.write(str(total_iters))

        print('End of epoch %d / %d \t Time Taken: %d sec' % (epoch, args.n_epochs, time.time() - epoch_start))

    print('Smoke-run finished. Checkpoint written to: %s' % ckpt_dir)


if __name__ == '__main__':
    main()
