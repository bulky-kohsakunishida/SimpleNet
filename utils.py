import csv
import logging
import os
import random

import matplotlib.pyplot as plt
import numpy as np
import PIL
import torch
import tqdm
import cv2

import metrics

LOGGER = logging.getLogger(__name__)


def plot_segmentation_images(
    savefolder,
    image_paths,
    segmentations,
    anomaly_scores=None,
    label_gts=None,
    masks=None,
    image_transform=lambda x: x,
    mask_transform=lambda x: x,
    save_depth=4,
):
    """Generate anomaly segmentation images.

    Args:
        image_paths: List[str] List of paths to images.
        segmentations: [List[np.ndarray]] Generated anomaly segmentations.
        anomaly_scores: [List[float]] Anomaly scores for each image.
        mask_paths: [List[str]] List of paths to ground truth masks.
        image_transform: [function or lambda] Optional transformation of images.
        mask_transform: [function or lambda] Optional transformation of masks.
        save_depth: [int] Number of path-strings to use for image savenames.
    """
    cutoff= -1
    scores = []
    if anomaly_scores is not None and label_gts is not None:
        scores = np.squeeze(np.array(anomaly_scores))
        min_scores = scores.min(axis=-1)
        max_scores = scores.max(axis=-1)
        scores = (scores - min_scores) / (max_scores - min_scores)
        _, cutoff = metrics.compute_confusion_matrix(scores, label_gts)
    if masks is None:
        masks = ["-1" for _ in range(len(image_paths))]
    masks_provided = masks[0] != "-1"
    if label_gts is None:
        label_gts = ["-1" for _ in range(len(image_paths))]

    os.makedirs(savefolder, exist_ok=True)

    for image_path, mask, score, label_gt, segmentation in tqdm.tqdm(
        zip(image_paths, masks, scores, label_gts, segmentations),
        total=len(image_paths),
        desc="Generating Segmentation Images...",
        leave=False,
    ):
        image = PIL.Image.open(image_path).convert("RGB")
        image = image_transform(image)
        if not isinstance(image, np.ndarray):
            image = np.asarray(image)

        if masks_provided:
            if mask is not None:
                if not isinstance(mask, np.ndarray):
                    mask = np.array(mask)
            else:
                mask = np.zeros_like(image)

        savename = image_path.split("/")
        axes_composite_image = 0
        add_title = ""
        if cutoff != -1:
            predict_label = 1 if score >= cutoff else 0
            if label_gt != predict_label:
                filename_without_extension, extension = os.path.splitext(savename[-1])
                savename[-1] = f"{filename_without_extension}_NG{extension}"
            add_title = f" label_gt,predict_label: {label_gt},{predict_label}"
            axes_composite_image = 2
        savename = "_".join(savename[-save_depth:])
        plot_title = f"Image: {savename}" + add_title
        savename = os.path.join(savefolder, savename)

        f, axes = plt.subplots(1, 2 + axes_composite_image + int(masks_provided))
        f.suptitle(plot_title)
        axes[0].imshow(image)
        axes[1].imshow(mask.squeeze())
        axes[2].imshow(segmentation)
        if axes_composite_image != 0:
            scale_image = cv2.resize(image,segmentation.shape)
            segmentation_scores = segmentation.copy()
            segmentation_scores = cv2.normalize(segmentation_scores, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            segmentation_scores = cv2.applyColorMap(255-segmentation_scores, cv2.COLORMAP_JET)
            back_image = np.full(scale_image.shape,127).astype(np.uint8)
            blended = cv2.addWeighted(src1=back_image, alpha=0.5, src2=segmentation_scores, beta=0.5, gamma=0)
            axes[3].imshow(blended)
            blended = cv2.addWeighted(src1=scale_image, alpha=0.5, src2=segmentation_scores, beta=0.5, gamma=0)
            axes[4].imshow(blended)
        f.set_size_inches(3 * (2 + + int(axes_composite_image) + int(masks_provided)), 3)
        f.tight_layout()
        f.savefig(savename)
        plt.close()


def create_storage_folder(
    main_folder_path, project_folder, group_folder, run_name, mode="iterate"
):
    os.makedirs(main_folder_path, exist_ok=True)
    project_path = os.path.join(main_folder_path, project_folder)
    os.makedirs(project_path, exist_ok=True)
    save_path = os.path.join(project_path, group_folder, run_name)
    if mode == "iterate":
        counter = 0
        while os.path.exists(save_path):
            save_path = os.path.join(project_path, group_folder + "_" + str(counter))
            counter += 1
        os.makedirs(save_path)
    elif mode == "overwrite":
        os.makedirs(save_path, exist_ok=True)

    return save_path


def set_torch_device(gpu_ids):
    """Returns correct torch.device.

    Args:
        gpu_ids: [list] list of gpu ids. If empty, cpu is used.
    """
    if len(gpu_ids):
        # os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        # os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_ids[0])
        return torch.device("cuda:{}".format(gpu_ids[0]))
    return torch.device("cpu")


def fix_seeds(seed, with_torch=True, with_cuda=True):
    """Fixed available seeds for reproducibility.

    Args:
        seed: [int] Seed value.
        with_torch: Flag. If true, torch-related seeds are fixed.
        with_cuda: Flag. If true, torch+cuda-related seeds are fixed
    """
    random.seed(seed)
    np.random.seed(seed)
    if with_torch:
        torch.manual_seed(seed)
    if with_cuda:
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True


def compute_and_store_final_results(
    results_path,
    results,
    row_names=None,
    column_names=[
        "Instance AUROC",
        "Full Pixel AUROC",
        "Full PRO",
        "Anomaly Pixel AUROC",
        "Anomaly PRO",
    ],
):
    """Store computed results as CSV file.

    Args:
        results_path: [str] Where to store result csv.
        results: [List[List]] List of lists containing results per dataset,
                 with results[i][0] == 'dataset_name' and results[i][1:6] =
                 [instance_auroc, full_pixelwisew_auroc, full_pro,
                 anomaly-only_pw_auroc, anomaly-only_pro]
    """
    if row_names is not None:
        assert len(row_names) == len(results), "#Rownames != #Result-rows."

    mean_metrics = {}
    for i, result_key in enumerate(column_names):
        mean_metrics[result_key] = np.mean([x[i] for x in results])
        LOGGER.info("{0}: {1:3.3f}".format(result_key, mean_metrics[result_key]))

    savename = os.path.join(results_path, "results.csv")
    with open(savename, "w") as csv_file:
        csv_writer = csv.writer(csv_file, delimiter=",")
        header = column_names
        if row_names is not None:
            header = ["Row Names"] + header

        csv_writer.writerow(header)
        for i, result_list in enumerate(results):
            csv_row = result_list
            if row_names is not None:
                csv_row = [row_names[i]] + result_list
            csv_writer.writerow(csv_row)
        mean_scores = list(mean_metrics.values())
        if row_names is not None:
            mean_scores = ["Mean"] + mean_scores
        csv_writer.writerow(mean_scores)

    mean_metrics = {"mean_{0}".format(key): item for key, item in mean_metrics.items()}
    return mean_metrics
