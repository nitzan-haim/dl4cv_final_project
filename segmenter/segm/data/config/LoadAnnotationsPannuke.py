import os.path as osp

import mmcv
import numpy as np
# import pycocotools.mask as maskUtils

# from mmdet.core import BitmapMasks, PolygonMasks

from mmseg.datasets import PIPELINES
try:
    from panopticapi.utils import rgb2id
except ImportError:
    rgb2id = None

@PIPELINES.register_module()
class LoadAnnotationsPannuke:
    """Load multiple types of annotations.
    Args:
        with_bbox (bool): Whether to parse and load the bbox annotation.
             Default: True.
        with_label (bool): Whether to parse and load the label annotation.
            Default: True.
        with_mask (bool): Whether to parse and load the mask annotation.
             Default: False.
        with_seg (bool): Whether to parse and load the semantic segmentation
            annotation. Default: False.
        poly2mask (bool): Whether to convert the instance masks from polygons
            to bitmaps. Default: True.
        denorm_bbox (bool): Whether to convert bbox from relative value to
            absolute value. Only used in OpenImage Dataset.
            Default: False.
        file_client_args (dict): Arguments to instantiate a FileClient.
            See :class:`mmcv.fileio.FileClient` for details.
            Defaults to ``dict(backend='disk')``.
    """

    def __init__(self,
                 with_bbox=False,
                 with_label=False,
                 with_mask=False,
                 with_seg=True,
                 poly2mask=False,
                 denorm_bbox=False,
                 file_client_args=dict(backend='disk')):
        self.with_bbox = with_bbox
        self.with_label = with_label
        self.with_mask = with_mask
        self.with_seg = with_seg
        self.poly2mask = poly2mask
        self.denorm_bbox = denorm_bbox
        self.file_client_args = file_client_args.copy()
        self.file_client = None

    def _load_bboxes(self, results):
        """Private function to load bounding box annotations.
        Args:
            results (dict): Result dict from :obj:`mmdet.CustomDataset`.
        Returns:
            dict: The dict contains loaded bounding box annotations.
        """

        ann_info = results['ann_info']
        results['gt_bboxes'] = ann_info['bboxes'].copy()

        if self.denorm_bbox:
            bbox_num = results['gt_bboxes'].shape[0]
            if bbox_num != 0:
                h, w = results['img_shape'][:2]
                results['gt_bboxes'][:, 0::2] *= w
                results['gt_bboxes'][:, 1::2] *= h

        gt_bboxes_ignore = ann_info.get('bboxes_ignore', None)
        if gt_bboxes_ignore is not None:
            results['gt_bboxes_ignore'] = gt_bboxes_ignore.copy()
            results['bbox_fields'].append('gt_bboxes_ignore')
        results['bbox_fields'].append('gt_bboxes')

        gt_is_group_ofs = ann_info.get('gt_is_group_ofs', None)
        if gt_is_group_ofs is not None:
            results['gt_is_group_ofs'] = gt_is_group_ofs.copy()

        return results

    def _load_labels(self, results):
        """Private function to load label annotations.
        Args:
            results (dict): Result dict from :obj:`mmdet.CustomDataset`.
        Returns:
            dict: The dict contains loaded label annotations.
        """

        results['gt_labels'] = results['ann_info']['labels'].copy()
        return results

#    def _poly2mask(self, mask_ann, img_h, img_w):
#        """Private function to convert masks represented with polygon to
#        bitmaps.
#        Args:
#            mask_ann (list | dict): Polygon mask annotation input.
#            img_h (int): The height of output mask.
#            img_w (int): The width of output mask.
#        Returns:
#            numpy.ndarray: The decode bitmap mask of shape (img_h, img_w).
#        """
#
#        if isinstance(mask_ann, list):
#            # polygon -- a single object might consist of multiple parts
#            # we merge all parts into one mask rle code
#            rles = maskUtils.frPyObjects(mask_ann, img_h, img_w)
#            rle = maskUtils.merge(rles)
#        elif isinstance(mask_ann['counts'], list):
#            # uncompressed RLE
#            rle = maskUtils.frPyObjects(mask_ann, img_h, img_w)
#        else:
#            # rle
#            rle = mask_ann
#        mask = maskUtils.decode(rle)
#        return mask

    def process_polygons(self, polygons):
        """Convert polygons to list of ndarray and filter invalid polygons.
        Args:
            polygons (list[list]): Polygons of one instance.
        Returns:
            list[numpy.ndarray]: Processed polygons.
        """

        polygons = [np.array(p) for p in polygons]
        valid_polygons = []
        for polygon in polygons:
            if len(polygon) % 2 == 0 and len(polygon) >= 6:
                valid_polygons.append(polygon)
        return valid_polygons

    # def _load_masks(self, results):
    #     """Private function to load mask annotations.
    #     Args:
    #         results (dict): Result dict from :obj:`mmdet.CustomDataset`.
    #     Returns:
    #         dict: The dict contains loaded mask annotations.
    #             If ``self.poly2mask`` is set ``True``, `gt_mask` will contain
    #             :obj:`PolygonMasks`. Otherwise, :obj:`BitmapMasks` is used.
    #     """

    #     h, w = results['img_info']['height'], results['img_info']['width']
    #     gt_masks = results['ann_info']['masks']
    #     if self.poly2mask:
    #         gt_masks = BitmapMasks(
    #             [self._poly2mask(mask, h, w) for mask in gt_masks], h, w)
    #     else:
    #         # gt_masks = BitmapMasks(
    #         #     [self.process_(mask, h, w) for mask in gt_masks], h, w)
    #         gt_masks = PolygonMasks(
    #             [self.process_polygons(polygons) for polygons in gt_masks], h,
    #             w)
    #     results['gt_masks'] = gt_masks
    #     results['mask_fields'].append('gt_masks')
    #     return results

    def _load_semantic_seg(self, results):
        """Private function to load semantic segmentation annotations.
        Args:
            results (dict): Result dict from :obj:`dataset`.
        Returns:
            dict: The dict contains loaded semantic segmentation annotations.
        """

        # if self.file_client is None:
        #     self.file_client = mmcv.FileClient(**self.file_client_args)

        filename = osp.join(results['seg_prefix'],
                            results['ann_info']['seg_map'])
        # img_bytes = self.file_client.get(filename)
        # results['gt_semantic_seg'] = mmcv.imfrombytes(
        #     img_bytes, flag='unchanged').squeeze()
        segmap = np.load(filename).squeeze()
        # mask_channels, h,w = segmap.shape
        # for i in range(mask_channels):
        #   (segmap[i])[segmap[i] != 0] = i
        # segmap = segmap.sum(axis=0)
        # print("in LoadAnnotationsPannuke.py: segmap shape: ", segmap.shape)
        results['gt_semantic_seg'] = segmap.astype(np.float32)
        results['seg_fields'].append('gt_semantic_seg')
        return results

    def __call__(self, results):
        """Call function to load multiple types annotations.
        Args:
            results (dict): Result dict from :obj:`mmdet.CustomDataset`.
        Returns:
            dict: The dict contains loaded bounding box, label, mask and
                semantic segmentation annotations.
        """

        if self.with_bbox:
            results = self._load_bboxes(results)
            if results is None:
                return None
        if self.with_label:
            results = self._load_labels(results)
        # if self.with_mask:
        #     results = self._load_masks(results)
        if self.with_seg:
            results = self._load_semantic_seg(results)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(with_bbox={self.with_bbox}, '
        repr_str += f'with_label={self.with_label}, '
        repr_str += f'with_mask={self.with_mask}, '
        repr_str += f'with_seg={self.with_seg}, '
        repr_str += f'poly2mask={self.poly2mask}, '
        repr_str += f'file_client_args={self.file_client_args})'
        return repr_str
