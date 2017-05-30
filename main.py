import os
from SPfinder import SPfinder

cxi_name = '/Users/lqhuang/Documents/CSRC/Data/Two-color/cxi_files_new/cxij6916-r0078-c00.cxi'
geom_name = '/Users/lqhuang/Documents/CSRC/Data/Two-color/geometry/geometry.h5'

run_dir = os.path.abspath('data/run78_minsize8')

run78 = SPfinder(cxi_name, geom_name)



# a, b = run78.find_spots(757)



# run78_size = run78.get_frame_size()
# run78.save_all_spots(run_dir)

# run78.save_spots([756, 757, 758], run_dir)

# run78.plot_spots_statistics(dstpath='data/run78_minsize8', save=True)
# run78.plot_spots_statistics(dstpath='data/run78', save=True)

frame = 0
# # frame = 756
run78.plot_frame_map(frame, display_peaks=True)


# run78.plot_spots_statistics_for_frame(frame)


# run78.find_split_spots(run_dir, display=False, save=True)