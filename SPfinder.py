import os
import shutil
import psutil
import pickle
import multiprocessing as mp
from itertools import combinations

import h5py
import numpy as np
from scipy.stats import gaussian_kde
from scipy import ndimage
from skimage.measure import regionprops, label
from skimage import morphology, feature
import matplotlib.pyplot as plt

from geometry import Geometry
import utils

def reverse_enumerate(iterable):
    """
    Enumerate over an iterable in reverse order while retaining proper indexes
    """
    return zip(reversed(range(len(iterable))), reversed(iterable))


class SPfinder(object):

    plt.rcParams['mathtext.fontset'] = 'cm'
    plt.rcParams['mathtext.rm'] = 'serif'
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    PLOT_PAR = {'family': 'sans-serif',
                'weight': 'normal',
                'size': 14}
    plt.rc('font', **PLOT_PAR)
    plt.rc('text', **{'latex.unicode': True})
    PLOT_NUM = 0

    # -------- #
    PIXEL_SIZE = 110e-6 # m
    COFFSET = 0.568 # 0.568 m -> angstrom
    THRESHOLD = 10
    MIN_INTENSITY = 80
    # SNR = 3
    MAX_CLIM = 20
    OBJECT_MIN_SIZE = 8
    PADDING_SIZE = 20
    SLOPE = 1.113
    INTERCEPT = -85.649 + 22.234

    def __init__(self, cxi_name, geom_name):
        self.cxi_file = h5py.File(cxi_name)
        self.frame_size = self.cxi_file['/LCLS/photon_energy_eV'].shape[0]
        self.geom = Geometry(geom_name, self.PIXEL_SIZE)

    
    def __exit__(self):
        self.cxi_file.close()

    def get_frame_size(self):
        return self.frame_size

    def get_energy(self, frame):
        energy = self.cxi_file['/LCLS/photon_energy_eV'][frame]
        return energy

    def get_working_distance(self, frame):
        working_distance = self.cxi_file['/LCLS/detector_1/EncoderValue'][frame] * 1e-3 + self.COFFSET # m
        return working_distance

    def get_working_distance_shift(self, frame):
        working_distance_shift = self.cxi_file['/LCLS/detector_1/EncoderValue'][frame] # mm 
        # * 1e-3 m
        return working_distance_shift

    def get_frame_map(self, frame):
        working_distance = self.get_working_distance(frame)
        scale_mask = self.geom.generate_intensity_scale_mask(working_distance)
        intensity = self.cxi_file['/entry_1/instrument_1/detector_1/detector_corrected/data'][frame]
        image = self.geom.rearrange(intensity) * scale_mask
        return image

    def is_filtered(self, x, y):
        y_ = self.SLOPE * np.asarray(x) + self.INTERCEPT
        return y_ > np.asarray(y)


    def find_spots(self, frame, inc_intensity=False, reject=True):

        intensity = self.get_frame_map(frame)
        intensity[intensity < self.THRESHOLD] = 0

        object_map = morphology.remove_small_objects(intensity>0, min_size=self.OBJECT_MIN_SIZE)
        intensity[~object_map] = 0

        label_image = label(intensity>0)
        regions = regionprops(label_image, intensity_image=intensity)

        if reject:
            for i, props in reverse_enumerate(regions):
                if props.max_intensity < self.MIN_INTENSITY:
                    regions.pop(i)

        if inc_intensity:
            return regions, intensity
        else:
            return regions
    
    def dump_spots(self, save_to_pickles=True, dstpath=None):
        # get_length = lambda regions: len(regions)
        # get_area = lambda props: props.area
        # get_orientation = lambda props: props.orientation
        # get_weighted_centroid = lambda props: props.weighted_centroid
        # get_radius = lambda props: np.sqrt(np.sum(np.asarray(props.weighted_centroid) ** 2, axis=1))
        # spots_group = [self.find_spots(fr) for fr in range(self.frame_size)]
        # spots_group = spots_group = [self.find_spots(fr) for fr in range(self.frame_size)]
        # num_peaks_per_frame = [len(regions) for regions in spots_group]
        # areas = [props.area for regions in spots_group for props in regions]
        # orientations = [props.orientation for regions in spots_group for props in regions]
        # weighted_centroids = [props.weighted_centroid for regions in spots_group for props in regions]
        if dstpath:
            pkl_path = os.path.join(dstpath, 'pickles')
            stat_path = os.path.join(dstpath, 'statistics')
            if not os.path.exists(pkl_path):
                os.makedirs(pkl_path, exist_ok=True)
            if not os.path.exists(stat_path):
                os.makedirs(stat_path, exist_ok=True)

            num_peaks_per_frame = list()
            areas = list()
            orientations = list()
            weighted_centroids = list()
            major_axis_length = list()
            minor_axis_length = list()

            for fr in range(self.frame_size):
                spots = self.find_spots(fr)
                if save_to_pickles:
                    with open(os.path.join(pkl_path, 'frame'+str(fr).zfill(4))+'.pkl', 'wb') as pkl_file:
                        pkl_file = pickle.dump(spots, pkl_file)

                num_peaks_per_frame += [len(spots)]
                areas += [props.area for props in spots]
                orientations += [props.orientation for props in spots]
                weighted_centroids += [props.weighted_centroid for props in spots]
                major_axis_length += [props.major_axis_length for props in spots]
                minor_axis_length += [props.minor_axis_length for props in spots]

            with open(os.path.join(stat_path, 'statistics.pkl'), 'wb') as stat_pkl:
                pickle.dump([num_peaks_per_frame, areas, orientations,
                                weighted_centroids, major_axis_length, minor_axis_length],
                                stat_pkl)
        else:
            pass
        
        return num_peaks_per_frame, areas, orientations, \
            weighted_centroids, major_axis_length, minor_axis_length

    def find_split_spots(self, dstpath, display=True, save=False):
            
        for fr in range(self.frame_size):
            
            energy = self.get_energy(fr)
            working_distance_shift = self.get_working_distance_shift(fr)
            intensity_map = self.get_frame_map(fr)
            regions, ori = self.find_spots(fr, inc_intensity=True)

            for j, props in enumerate(regions):
                
                image = props.intensity_image
                label_img = props.image
                minr, minc, maxr, maxc = props.bbox
                cy, cx = props.centroid
                
                local_max = feature.peak_local_max(image, indices=False,
                                                   footprint=np.ones((3, 3)),
                                                   labels=label_img)
                centroids = feature.peak_local_max(image, indices=True,
                                                   footprint=np.ones((3, 3)),
                                                   labels=label_img)
                # coordinate of the highest intensity point
                idx_maximum = np.argmax(image)
                ym, xm = np.unravel_index(idx_maximum, image.shape)

                markers, num_markers = ndimage.label(local_max)

                if num_markers > 1 and not self.is_filtered(xm, ym):
                    # watershed
                    labels = morphology.watershed(-image, markers, mask=label_img)      
                    split = regionprops(labels, intensity_image=image)
                    split_centroids = np.asarray([split_props.weighted_centroid for split_props in split])
                    split_y, split_x = split_centroids[:, 0], split_centroids[:, 1]

                    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(8, 8), sharex=True, sharey=True)

                    ax0, ax1, ax2, ax3 = axes.ravel()

                    im = ax0.imshow(image, clim=(0, image.mean()), cmap='gray', origin='lower')
                    ax0.set_title("Original")
                    y, x = centroids[:, 0], centroids[:, 1]
                    ax0.plot(x, y, 'ro', fillstyle='none')

                    ax1.imshow(markers, cmap='nipy_spectral', origin='lower')
                    ax1.set_title("Markers")
                    for (p1, p2) in combinations(centroids, 2):
                        center = (p1 + p2) / 2
                        dist = np.sqrt( np.sum( (p1 - p2) ** 2 ) )
                        ax1.text(center[1], center[0], '{:.2f}'.format(dist), color='white')
                        ax1.plot((p1[1], p2[1]), (p1[0], p2[0]), 'b')
                    axis_lim = ax1.axis()
                    real_xm = xm + minc
                    real_ym = ym + minr
                    slope, intercept = np.polyfit((real_xm, self.geom.offset_x), (real_ym, self.geom.offset_y), 1)
                    x_line_to_source = np.arange(0, maxc - minc)
                    y_line_to_source = slope * (x_line_to_source + minc) + intercept - minr
                    ax1.plot(x_line_to_source, y_line_to_source, '--', color='magenta')
                    ax1.axis(axis_lim)

                    ax2.imshow(labels, cmap='nipy_spectral', origin='lower')
                    ax2.set_title("Segmented")
                    ax2.plot(split_x, split_y, 'ro', fillstyle='none')

                    ax3.imshow(np.zeros_like(image), cmap='gray', origin='lower')
                    for (p1, p2) in combinations(split_centroids, 2):
                        center = (p1 + p2) / 2
                        dist = np.sqrt( np.sum( (p1 - p2) ** 2 ) )
                        ax3.text(center[1], center[0], '{:.2f}'.format(dist), color='white')
                        ax3.plot((p1[1], p2[1]), (p1[0], p2[0]), 'b')
                    ax3.axis(axis_lim)

                    rho = np.sqrt((real_xm - self.geom.offset_x) ** 2 + (real_ym - self.geom.offset_y) ** 2)
                    DeltaQ = utils.get_DeltaQ(80, rho, energy, working_distance_shift)
                    ax3.set_title(r'$\Delta Q$={:.2f} pixel'.format(DeltaQ))

                    # for ax in axes.ravel():
                    #     ax.set_xlabel('$x$ (pixel)')
                    #     ax.set_ylabel('$y$ (pixel)')
                    #     xticks = ax.get_xticks()
                    #     ax.set_xticklabels(np.asarray(xticks) + minc - self.geom.offset_x)
                    #     yticks = ax.get_yticks()
                    #     ax.set_yticklabels(np.asarray(yticks) + minr - self.geom.offset_y)

                    if save:
                        fig_dir = os.path.join(dstpath, 'split_spots')
                        if not os.path.exists(fig_dir):
                            os.makedirs(fig_dir)
                        filename = 'frame_{0}_label_{1}'.format(fr, j)
                        plt.savefig(os.path.join(fig_dir, filename), dpi=150, bbox_inches='tight')
                        fig2 = plt.figure(2)
                        plt.imshow(intensity_map, clim=(0, 50), cmap='gray', origin='lower')
                        plt.plot(cx, cy, 'ro', fillstyle='none', markersize=10)
                        plt.colorbar()
                        plt.savefig(os.path.join(fig_dir, filename+'_in_pixelmap'), bbox_inches='tight')
                    if display:
                        fig.tight_layout()
                        plt.show()

                    plt.close(fig)
                    plt.close(fig2)

    # ------------------------ #
    def plot_frame_map(self, frames, display=True, save=False, dstpath=None, display_peaks=True):
        if isinstance(frames, int):
            frames = (frames,)
        for fr in frames:
            fig, ax = plt.subplots()
            image = self.get_frame_map(fr)
            im = ax.imshow(image, clim=(0, self.MAX_CLIM), cmap='gray', origin='lower')
            fig.colorbar(im)
            ax.set_xlabel('$x$ (col) (pixel)')
            ax.set_ylabel('$y$ (row) (pixel)')
            ax.set_title('Pixel Map for Frame {0}'.format(fr))

            x_ = np.arange(0, self.geom.nx)
            y_ = self.SLOPE * x_ + self.INTERCEPT
            ax.plot(x_, y_, color='red', linewidth=2.5)
            ax.axis([0, self.geom.nx-1, 0, self.geom.ny-1])
            xticks = ax.get_xticks()
            ax.set_xticklabels(np.asarray(xticks) - self.geom.offset_x)
            yticks = ax.get_yticks()
            ax.set_yticklabels(np.asarray(yticks) - self.geom.offset_y)

            if display_peaks:
                spots = self.find_spots(fr)
                for i, props in enumerate(spots):
                    y0, x0 = props.centroid
                    yw, xw = props.weighted_centroid
                    orientation = props.orientation
                    # x1 = x0 + np.cos(orientation) * 0.5 * props.major_axis_length
                    # y1 = y0 - np.sin(orientation) * 0.5 * props.major_axis_length
                    # x2 = x0 - np.sin(orientation) * 0.5 * props.minor_axis_length
                    # y2 = y0 - np.cos(orientation) * 0.5 * props.minor_axis_length

                    # ax.plot((x0, x1), (y0, y1), '-r', linewidth=2.5)
                    # ax.plot((x0, x2), (y0, y2), '-r', linewidth=2.5)
                    ax.plot(xw, yw, '.g', markersize=10)
                    ax.text(xw+2, yw+2, str(i), fontsize=12, color='green')
                    # minr, minc, maxr, maxc = props.bbox
                    # bx = (minc, maxc, maxc, minc, minc)
                    # by = (minr, minr, maxr, maxr, minr)
                    # ax.plot(bx, by, '-b', linewidth=2.5)

            if display:
                plt.show()
            if save and dstpath:
                fig_dir = os.path.join(dstpath, 'spots', 'frame'+str(fr).zfill(4))
                if not os.path.exists(fig_dir):
                    os.mkdir(fig_dir)
                plt.savefig(os.path.join(fig_dir, 'intensity_map'), dpi=150)
            plt.close(fig)

    def plot_spots_statistics_for_frame(self, frame):
        spots = self.find_spots(frame)

        num_peaks = len(spots)
        print('number of peaks for frame {0} is {1}.'.format(frame, num_peaks))
        areas = [props.area for props in spots]
        orientations = [props.orientation for props in spots]
        weighted_centroids = [props.weighted_centroid for props in spots]
        major_axis_length = [props.major_axis_length for props in spots]
        minor_axis_length = [props.minor_axis_length for props in spots]

        shift_weighted_centroids = np.asarray(weighted_centroids) - (self.geom.offset_y, self.geom.offset_x)
        radius = np.sqrt((shift_weighted_centroids ** 2).sum(axis=1))

        fig, ax = plt.subplots()
        ax.hist(areas, bins=np.arange(0.5, 100.5), edgecolor='white')
        ax.set_xlabel('number of pixels in spot')
        ax.set_ylabel('counts')
        ax.set_title('area histogram for all spots')
        fig, ax = plt.subplots()
        ax.hist(radius, edgecolor='white')
        ax.set_xlabel('radius of peak (weighted centroid)')
        ax.set_ylabel('counts')
        ax.set_title('radius histogram for all spots')
        fig, ax = plt.subplots()
        ax.plot(radius, areas, '.', markersize=1)
        ax.set_xlabel('radius')
        ax.set_ylabel('areas')
        ax.set_title('scatter for radius and areas')
        plt.tight_layout()
        plt.show()

    def plot_spots_statistics(self, dstpath, save=False):
        stat_pkl = os.path.join(dstpath, 'statistics', 'statistics.pkl') 
        if os.path.exists(stat_pkl):
            with open(stat_pkl, 'rb') as pkl_file:
                num_peaks_per_frame, areas, orientations, weighted_centroids, \
                    major_axis_length, minor_axis_length = pickle.load(pkl_file)
        else:
            num_peaks_per_frame, areas, orientations, weighted_centroids, \
                    major_axis_length, minor_axis_length = \
                    self.dump_spots(save_to_pickles=save, dstpath=dstpath)

        shift_weighted_centroids = np.asarray(weighted_centroids) - (self.geom.offset_y, self.geom.offset_x)
        major_axis_length = np.asarray(major_axis_length)
        minor_axis_length = np.asarray(minor_axis_length)
        minor_axis_length[minor_axis_length==0] = 100000000000
        axis_ratio = major_axis_length / minor_axis_length
        radius = np.sqrt((shift_weighted_centroids ** 2).sum(axis=1))

        fig_dir = os.path.join(dstpath, 'statistics')

        fig, ax = plt.subplots()
        ax.hist(areas, bins=np.arange(0.5, 100.5), edgecolor='white')
        ax.set_xlabel('number of pixels in spot')
        ax.set_ylabel('counts')
        ax.set_title('area histogram for all spots')
        plt.tight_layout()
        if save:
            plt.savefig(os.path.join(fig_dir, 'area_histogram'), dpi=150, bbox_inches='tight')

        fig, ax = plt.subplots()
        ratio_kde = gaussian_kde(axis_ratio)
        x = np.arange(0, 10, 0.05)
        ax.plot(x, ratio_kde.evaluate(x))
        ax.set_xlabel('ratio for major axis length and minor axis length')
        ax.set_ylabel('counts')
        ax.set_title('ratio kernel density estimation for all spots')
        if save:
            plt.savefig(os.path.join(fig_dir, 'ratio_kde'), dpi=150, bbox_inches='tight')

        fig, ax = plt.subplots()
        ax.hist(axis_ratio, bins=np.arange(-0.25, 10.5, 0.5), edgecolor='white')
        ax.set_xlabel('ratio for major axis length and minor axis length')
        ax.set_ylabel('counts')
        ax.set_title('ratio histogram for all spots')
        if save:
            plt.savefig(os.path.join(fig_dir, 'ratio_histogram'), dpi=150, bbox_inches='tight')

        fig, ax = plt.subplots()
        ax.hist(radius, bins=np.arange(-50, 1000, step=100), edgecolor='white')
        ax.set_xlabel('radius of peak (weighted centroid)')
        ax.set_ylabel('counts')
        ax.set_title('radius histogram for all spots')
        plt.tight_layout()
        if save:
            plt.savefig(os.path.join(fig_dir, 'radius_histogram'), dpi=150, bbox_inches='tight')
        
        fig, ax = plt.subplots()
        radius_kde = gaussian_kde(radius)
        x = np.arange(0, 1000, 10)
        ax.plot(x, radius_kde.evaluate(x))
        ax.set_xlabel('radius of peak (weighted centroid)')
        ax.set_ylabel('counts')
        ax.set_title('radius kernel density estimation for all spots')
        plt.tight_layout()
        if save:
            plt.savefig(os.path.join(fig_dir, 'radius_kde'), dpi=150, bbox_inches='tight')

        fig, ax = plt.subplots()
        ax.plot(list(range(self.frame_size)), num_peaks_per_frame)
        ax.set_xlabel('frame')
        ax.set_ylabel('number of peaks')
        ax.set_title('number of peaks for each frame')
        plt.tight_layout()
        if save:
            plt.savefig(os.path.join(fig_dir, 'num_peaks_per_frame'), dpi=150, bbox_inches='tight')

        fig, ax = plt.subplots()
        ax.plot(radius, areas, '.', markersize=1)
        ax.set_xlabel('radius')
        ax.set_ylabel('areas')
        ax.set_title('scatter for radius and areas')
        plt.tight_layout()
        if save:
            plt.savefig(os.path.join(fig_dir, 'radius_area'), dpi=150, bbox_inches='tight')

        plt.show()

    def save_spot_region(self, frame, dstpath):

        regions, img_map = self.find_spots(frame, inc_intensity=True)
        fig_dir = os.path.join(dstpath, 'spots', 'frame'+str(frame).zfill(4))
        if os.path.exists(fig_dir):
            shutil.rmtree(fig_dir)
        os.makedirs(fig_dir, exist_ok=True)
        for i, props in enumerate(regions):
            padding_size = self.PADDING_SIZE
            minr, minc, maxr, maxc = props.bbox
            if minr - padding_size < 0:
                minr_padding_size = minr
            else:
                minr_padding_size = padding_size
            if minc - padding_size < 0:
                minc_padding_size = minc
            else:
                minc_padding_size = padding_size
            selected_region = img_map[minr-minr_padding_size:maxr+padding_size,
                                      minc-minc_padding_size:maxc+padding_size]
            fig, ax = plt.subplots()
            im = ax.imshow(selected_region, cmap='gray', clim=(0, self.MAX_CLIM), origin='lower')
            fig.colorbar(im)
            bx = (minc_padding_size, maxc-minc+minc_padding_size, 
                  maxc-minc+minc_padding_size, minc_padding_size, minc_padding_size)
            by = (minr_padding_size, minr_padding_size,
                  maxr-minr+minr_padding_size, maxr-minr+minr_padding_size, minr_padding_size)
            ax.plot(bx, by, '-b', linewidth=2.5)

            yw, xw = props.weighted_local_centroid
            yw += minr_padding_size
            xw += minc_padding_size
            y0, x0 = props.local_centroid
            y0 += minr_padding_size
            x0 += minc_padding_size
            x1 = x0 + np.cos(props.orientation) * 0.5 * props.major_axis_length
            y1 = y0 - np.sin(props.orientation) * 0.5 * props.major_axis_length
            x2 = x0 - np.sin(props.orientation) * 0.5 * props.minor_axis_length
            y2 = y0 - np.cos(props.orientation) * 0.5 * props.minor_axis_length

            ax.plot((x0, x1), (y0, y1), '-r', linewidth=2.5)
            ax.plot((x0, x2), (y0, y2), '-r', linewidth=2.5)
            ax.plot(x0, y0, '.r', markersize=10)
            ax.plot(xw, yw, '.g', markersize=10)
            col_ticks = ax.get_xticks()
            row_ticks = ax.get_yticks()
            ax.set_xticklabels(np.asarray(col_ticks + minc - minc_padding_size, dtype=int))
            ax.set_yticklabels(np.asarray(row_ticks + minr - minr_padding_size, dtype=int))

            plt.savefig(os.path.join(fig_dir, str(i).zfill(3)))
            plt.close(fig)

    def save_spots(self, frames, dstpath):
        if isinstance(frames, int):
            frames = (frames,)
        for fr in frames:
            self.save_spot_region(fr, dstpath)

    def save_all_spots(self, dstpath, parallel=True):
        threads = psutil.cpu_count(logical=False)
        # pool = mp.Pool(threads)
        # partial_spot_save = partialmethod(self.save_spot_region, dstpath=dstpath)
        # pool.map(partial_spot_save, [1])
        for fr in range(self.frame_size):
            print('plot_frames:', fr)
            self.save_spot_region(fr, dstpath)
            self.plot_frame_map(fr, display=False, save=True, dstpath=dstpath)
